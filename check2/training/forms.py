"""The two forms a training step can run in, behind one interface.

Both run a forward batch through the pipeline's model and hooks and return logits at the
reply positions only. The batch is laid out as the measured shared-prefix check laid it
out: the tokens `0..r` left-padded, so that every row's read token sits in one column, and
the tokens after `r` right-padded.

- `SharedPrefix` runs tokens `0..r` once, under `torch.no_grad()`, when the batch is
  prepared. Each pass then starts from its own deep copy of that cache, made to store its
  new recurrent state by rebinding, and runs over the tokens after `r`; its attention mask
  covers the cached tokens and the new ones.
- `WholeSequence` runs the whole padded sequence in every pass.

A pass sets the control state when it runs and leaves it set: the caller runs the pass's
backward pass and then calls `end_pass`, so that the state a pass ran under is the one in
force until its backward pass has finished.
"""

from __future__ import annotations

import copy
import types
from dataclasses import dataclass

import torch

from check2.mechanism import masks
from check2.mechanism.generate import Pipeline, _positions
from check2.training.examples import TrainingExample

# The write's strength in training.
TRAINING_ALPHA = 1.0


class FormError(RuntimeError):
    """A pass asked for an addition the pipeline does not have, or a batch the form cannot lay out."""


def rebind_recurrent_state(cache):
    for layer in cache.layers:
        if hasattr(layer, "recurrent_states") and hasattr(layer, "update_recurrent_state"):
            def upd(self, recurrent_states, state_idx=0, **kwargs):
                if not self.is_recurrent_states_initialized[state_idx]:
                    self.lazy_initialization(recurrent_states=recurrent_states, state_idx=state_idx)
                self.recurrent_states[state_idx] = recurrent_states
                return self.recurrent_states[state_idx]
            layer.update_recurrent_state = types.MethodType(upd, layer)


def continuation_cache(prefix_cache):
    """A pass's own copy of the prefix cache, storing each new recurrent state by rebinding.

    The library's cache layer overwrites its Gated DeltaNet recurrent state in place after
    the kernel has saved it for the backward pass, which makes that backward pass fail;
    rebinding stores the new state as a new tensor instead.
    """
    cache = copy.deepcopy(prefix_cache)
    rebind_recurrent_state(cache)
    return cache


@dataclass
class Prepared:
    """A forward batch laid out for one form, with what its passes share."""

    examples: list[TrainingExample]
    ids: torch.Tensor          # [B, C], the tokens each pass feeds
    attention: torch.Tensor    # [B, columns], 1 at real tokens, covering every column up to the end of a pass
    read_column: torch.Tensor  # [B]
    reply_start: torch.Tensor  # [B], the column of `ids` whose output predicts the first reply token
    labels: torch.Tensor       # [B, L], the reply's tokens and the end-of-turn token, left-aligned
    reply_mask: torch.Tensor   # [B, L], True where a position predicts one of them
    cache: object = None       # the cache of tokens 0..r, for the shared-prefix form

    @property
    def batch_size(self) -> int:
        return len(self.examples)


@dataclass
class _Layout:
    prefix_ids: torch.Tensor
    prefix_attention: torch.Tensor
    continuation_ids: torch.Tensor
    continuation_attention: torch.Tensor
    read_column: torch.Tensor
    labels: torch.Tensor
    reply_mask: torch.Tensor


def _layout(examples: list[TrainingExample], pad_id: int, device) -> _Layout:
    if not examples:
        raise FormError("an empty batch")
    size = len(examples)
    width_prefix = max(e.prefix_length for e in examples)
    width_continuation = max(len(e.continuation_ids) for e in examples)
    width_reply = max(len(e.target_ids) for e in examples)
    prefix_ids = torch.full((size, width_prefix), pad_id, dtype=torch.long)
    prefix_attention = torch.zeros((size, width_prefix), dtype=torch.long)
    continuation_ids = torch.full((size, width_continuation), pad_id, dtype=torch.long)
    continuation_attention = torch.zeros((size, width_continuation), dtype=torch.long)
    labels = torch.zeros((size, width_reply), dtype=torch.long)
    reply_mask = torch.zeros((size, width_reply), dtype=torch.bool)
    for b, e in enumerate(examples):
        prefix = e.prompt_ids[:e.prefix_length]
        prefix_ids[b, width_prefix - len(prefix):] = torch.tensor(prefix)
        prefix_attention[b, width_prefix - len(prefix):] = 1
        rest = e.continuation_ids
        continuation_ids[b, :len(rest)] = torch.tensor(rest)
        continuation_attention[b, :len(rest)] = 1
        labels[b, :len(e.target_ids)] = torch.tensor(e.target_ids)
        reply_mask[b, :len(e.target_ids)] = True
    first = torch.tensor([width_prefix - e.prefix_length for e in examples])
    read_column = masks.read_columns(first, torch.tensor([e.read_index for e in examples]))
    return _Layout(prefix_ids.to(device), prefix_attention.to(device), continuation_ids.to(device),
                   continuation_attention.to(device), read_column.to(device), labels.to(device),
                   reply_mask.to(device))


class TrainingForm:
    """A way of running a training step's forward batches through the pipeline's model and hooks."""

    name = ""

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline
        self.control = pipeline.control

    def prepare(self, batch: list[TrainingExample]) -> Prepared:
        raise NotImplementedError

    def batch_tokens(self, batch: list[TrainingExample]) -> int:
        """The tokens a forward batch carries gradient over: its rows times its padded width."""
        raise NotImplementedError

    def _call(self, prepared: Prepared, adapter_on: bool, write_on: bool, p: torch.Tensor | None):
        """Set the control state for this pass; returns the write vectors and the call's mask."""
        pipeline = self.pipeline
        if adapter_on and pipeline.adapter is None:
            raise FormError("a pass with the adapter on, but the pipeline has no adapter")
        vectors = {}
        if write_on:
            if pipeline.writers is None:
                raise FormError("a pass with the write on, but the pipeline has no writers")
            if p is None or tuple(p.shape) != (prepared.batch_size, pipeline.writers.input_size):
                raise FormError(f"a pass with the write on needs a signal of shape "
                                f"[{prepared.batch_size}, {pipeline.writers.input_size}]")
            vectors = pipeline.writers(p.to(pipeline.device))
        length = prepared.ids.shape[1]
        mask = masks.after_read_mask(prepared.attention, prepared.read_column, length)
        self.control.set(mask.to(pipeline.device), vectors, TRAINING_ALPHA, (), adapter_active=adapter_on)
        return _positions(prepared.attention, length)

    def _reply_logits(self, hidden: torch.Tensor, prepared: Prepared) -> torch.Tensor:
        """Logits `[B, L, V]` in float32 at the positions that predict the reply and the end-of-turn token."""
        width = prepared.labels.shape[1]
        offsets = torch.arange(width, device=hidden.device)
        columns = (prepared.reply_start.unsqueeze(1) + offsets.unsqueeze(0)).clamp(max=hidden.shape[1] - 1)
        rows = torch.arange(hidden.shape[0], device=hidden.device).unsqueeze(1)
        return self.pipeline.model.lm_head(hidden[rows, columns]).to(torch.float32)

    def forward(self, prepared: Prepared, adapter_on: bool, write_on: bool, p: torch.Tensor | None) -> torch.Tensor:
        """One pass; returns logits `[B, L, V]` at the reply positions. The control state stays set until `end_pass`."""
        raise NotImplementedError

    def end_pass(self) -> None:
        """Clear the control state, once the pass's backward pass has finished."""
        self.control.clear()


class SharedPrefix(TrainingForm):
    """Tokens `0..r` run once without gradient; each pass continues from its own copy of their cache."""

    name = "shared_prefix"

    def prepare(self, batch: list[TrainingExample]) -> Prepared:
        pipeline = self.pipeline
        layout = _layout(batch, pipeline.pad_id, pipeline.device)
        width_prefix = layout.prefix_ids.shape[1]
        mask = masks.after_read_mask(layout.prefix_attention, layout.read_column, width_prefix)
        self.control.set(mask.to(pipeline.device), {}, TRAINING_ALPHA, (), adapter_active=False)
        try:
            with torch.no_grad():
                out = pipeline.body(input_ids=layout.prefix_ids, attention_mask=layout.prefix_attention,
                                    position_ids=_positions(layout.prefix_attention, width_prefix), use_cache=True)
        finally:
            self.control.clear()
        attention = torch.cat([layout.prefix_attention, layout.continuation_attention], 1)
        reply_start = torch.tensor([e.rest_length - 1 for e in batch], device=pipeline.device)
        return Prepared(examples=list(batch), ids=layout.continuation_ids, attention=attention,
                        read_column=layout.read_column, reply_start=reply_start, labels=layout.labels,
                        reply_mask=layout.reply_mask, cache=out.past_key_values)

    def batch_tokens(self, batch: list[TrainingExample]) -> int:
        return len(batch) * max(len(e.continuation_ids) for e in batch)

    def forward(self, prepared: Prepared, adapter_on: bool, write_on: bool, p: torch.Tensor | None) -> torch.Tensor:
        cache = continuation_cache(prepared.cache)
        positions = self._call(prepared, adapter_on, write_on, p)
        out = self.pipeline.body(input_ids=prepared.ids, attention_mask=prepared.attention, position_ids=positions,
                                 past_key_values=cache, use_cache=True)
        return self._reply_logits(out.last_hidden_state, prepared)


class WholeSequence(TrainingForm):
    """Every pass runs the whole padded sequence."""

    name = "whole_sequence"

    def prepare(self, batch: list[TrainingExample]) -> Prepared:
        layout = _layout(batch, self.pipeline.pad_id, self.pipeline.device)
        ids = torch.cat([layout.prefix_ids, layout.continuation_ids], 1)
        attention = torch.cat([layout.prefix_attention, layout.continuation_attention], 1)
        width_prefix = layout.prefix_ids.shape[1]
        reply_start = torch.tensor([width_prefix + e.rest_length - 1 for e in batch], device=self.pipeline.device)
        return Prepared(examples=list(batch), ids=ids, attention=attention, read_column=layout.read_column,
                        reply_start=reply_start, labels=layout.labels, reply_mask=layout.reply_mask)

    def batch_tokens(self, batch: list[TrainingExample]) -> int:
        return len(batch) * (max(e.prefix_length for e in batch) + max(len(e.continuation_ids) for e in batch))

    def forward(self, prepared: Prepared, adapter_on: bool, write_on: bool, p: torch.Tensor | None) -> torch.Tensor:
        positions = self._call(prepared, adapter_on, write_on, p)
        out = self.pipeline.body(input_ids=prepared.ids, attention_mask=prepared.attention, position_ids=positions,
                                 use_cache=False)
        return self._reply_logits(out.last_hidden_state, prepared)


FORMS = {form.name: form for form in (SharedPrefix, WholeSequence)}
