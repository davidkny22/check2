"""A frozen forming update attached to the model, so that it acts on every path that runs the formed model.

The update acts only while its own control state is set. The pipeline's one forward call
sets it. Two other paths run the model without that call, and each gets a hook on the
text decoder that sets the state for each call from what the call already carries:

- the training forms set only the consumer adapter's control state, whose mask is 1 at real
  tokens after the read token; the forming mask is then the call's real tokens less that
  mask (`follow_consumer_state`);
- the library's own `generate`, which target sampling uses, runs whole left-padded prompts
  and then one token at a time; in the call over the whole prompts the read token is the
  template tail's token at the read position, in one column of every row, and in a later
  call every token lies after it (`follow_prompt_tail`).

`load_frozen` builds the update with the forming training's factors, frozen and installed.
"""

from __future__ import annotations

import torch

from check2.data.episodes import TAIL_LENGTH, check_tail_position
from check2.mechanism.control import ControlState
from check2.mechanism.generate import Pipeline
from check2.mechanism.loading import Layout
from check2.mechanism.model_parts import text_model
from check2.training.forming.update import FormingError, FormingUpdate, layers_for


def load_frozen(model: torch.nn.Module, layout: Layout, implementation: str, factors: dict) -> FormingUpdate:
    """The forming update with the given factors, frozen and installed."""
    update = FormingUpdate(model, implementation, *layers_for(layout))
    update.load_factors(factors)
    update.freeze()
    update.install()
    return update


class FormedPipeline(Pipeline):
    """The pipeline of a formed model: it sets the forming update's state in its calls, and on close it removes
    the update's hooks and any hooks attached for other paths (`attached`)."""

    attached: list = []

    def close(self) -> None:
        super().close()
        for hooks in self.attached:
            hooks.remove()
        self.attached = []
        if self.forming is not None and self.forming.installed:
            self.forming.remove()


def _call_shape(args, kwargs) -> tuple[torch.Tensor, int]:
    attention = kwargs.get("attention_mask")
    ids = kwargs.get("input_ids", args[0] if args else None)
    if attention is None or ids is None:
        raise FormingError("a call on the formed model passed no attention mask or no input ids")
    return attention, ids.shape[1]


class _Attached:
    """Hook handles that set and clear the update's state around each call of the text decoder."""

    def __init__(self, update: FormingUpdate, model: torch.nn.Module, set_state) -> None:
        body = text_model(model)
        self.update = update

        def before(module, args, kwargs):
            attention, length = _call_shape(args, kwargs)
            set_state(attention, length)

        def after(module, args, kwargs, output):
            update.clear()

        self.handles = [body.register_forward_pre_hook(before, with_kwargs=True),
                        body.register_forward_hook(after, with_kwargs=True)]

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def follow_consumer_state(update: FormingUpdate, model: torch.nn.Module, consumer: ControlState) -> _Attached:
    """For the training forms: each call's forming mask is its real tokens less the consumer adapter's mask."""
    def set_state(attention: torch.Tensor, length: int) -> None:
        if consumer.mask is None:
            raise FormingError("a training call ran without the consumer adapter's control state set")
        real = attention[:, attention.shape[1] - length:].to(torch.float32)
        mask = real - consumer.mask.to(real.device)
        if bool((mask < 0).any()):
            raise FormingError("the consumer adapter's mask covers padding")
        update.control.set(mask.to(update.device), adapter_active=True)
    return _Attached(update, model, set_state)


def follow_prompt_tail(update: FormingUpdate, model: torch.nn.Module, tail_position: int) -> _Attached:
    """For the library's `generate` on whole left-padded prompts, with the read token at `tail[tail_position]`."""
    check_tail_position(tail_position)
    prompt_width: list[int] = []

    def set_state(attention: torch.Tensor, length: int) -> None:
        width = attention.shape[1]
        if length == width:
            prompt_width[:] = [width]
            read_column = torch.full((attention.shape[0],), width - TAIL_LENGTH + tail_position,
                                     device=attention.device)
            update.set_call(attention, read_column, length)
        elif prompt_width and width - length >= prompt_width[0]:
            update.control.set(torch.zeros((attention.shape[0], length), device=update.device),
                               adapter_active=True)
        else:
            raise FormingError("a call continues inside the prompt; its read token cannot be placed")
    return _Attached(update, model, set_state)
