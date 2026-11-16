"""The passes and backward passes of one forward batch, for each kind of training.

Each function prepares the batch in the given form, runs the passes its objective names,
back-propagates the sum of the batch's episode losses divided by the step's episode count
after each pass that carries gradient, and returns the batch's episode losses. The control
state set for a pass stays in force until that pass's backward pass has finished. The
optimiser and the loop belong elsewhere: gradients accumulate on the trained parameters
across the forward batches of a step.

- Connector: the writers, with the write on and no adapter.
  `L = CE(model(seq; write on), reply)`.
- Consumer: the adapter and the writers.
  `L = CE(logits_on, reply) + beta * KL(softmax(logits_orig) || softmax(logits_off))`, where
  `logits_orig` has no adapter and the write off and carries no gradient, `logits_off` has
  the adapter on and the write off, and `logits_on` has both on. The write-off pass and the
  write-on pass are each followed by their own backward pass. In the answer-weighted
  form, the cross-entropy of a short-format episode gives its answer token a weight of its
  own; this is the same in both training forms, which lay the reply positions out alike.
- Adapter-only: the adapter alone. `L = CE(model(seq; adapter on, write off), reply)`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from check2.training.examples import TrainingExample
from check2.training.forms import TrainingForm
from check2.training.answer_token import AnswerWeighting
from check2.training.loss import answer_weighted_losses, batch_objective, episode_kl, episode_losses


class ObjectiveError(RuntimeError):
    """A step asked of a pipeline that lacks the parts its objective trains."""


@dataclass
class EpisodeLosses:
    """One forward batch's losses per episode, detached: the total, its cross-entropy and its KL term."""

    total: torch.Tensor
    cross_entropy: torch.Tensor
    kl: torch.Tensor | None = None


def _backward(form: TrainingForm, objective: torch.Tensor) -> None:
    """Run the pass's backward pass, then clear the control state the pass ran under."""
    try:
        objective.backward()
    finally:
        form.end_pass()


def _cross_entropy_pass(form: TrainingForm, prepared, adapter_on: bool, write_on: bool, p, step_size: int,
                        answer: AnswerWeighting | None = None):
    try:
        logits = form.forward(prepared, adapter_on, write_on, p)
        if answer is None:
            losses = episode_losses(logits, prepared.labels, prepared.reply_mask)
        else:
            positions = answer.positions_of(prepared.examples).to(logits.device)
            losses = answer_weighted_losses(logits, prepared.labels, prepared.reply_mask, positions, answer.weight)
        del logits
    except BaseException:
        form.end_pass()
        raise
    _backward(form, batch_objective(losses, step_size))
    return losses.detach()


def connector_step(form: TrainingForm, batch: list[TrainingExample], p: torch.Tensor,
                   step_size: int) -> EpisodeLosses:
    """The writers' loss with the write on, the model frozen and no adapter."""
    if form.pipeline.writers is None:
        raise ObjectiveError("connector training needs writers")
    if form.pipeline.adapter is not None:
        raise ObjectiveError("connector training has no adapter")
    prepared = form.prepare(batch)
    ce = _cross_entropy_pass(form, prepared, adapter_on=False, write_on=True, p=p, step_size=step_size)
    return EpisodeLosses(total=ce, cross_entropy=ce)


def consumer_step(form: TrainingForm, batch: list[TrainingExample], p: torch.Tensor, beta: float,
                  step_size: int, answer: AnswerWeighting | None = None) -> EpisodeLosses:
    """The consumer objective: cross-entropy with the write on, plus `beta` times the KL of the write-off pass.

    With `answer`, the cross-entropy of a short-format episode gives its answer token the
    weighting's weight; the KL term is the same with it and without it.

    Without an adapter the write-off pass is the reference pass, so its KL term is zero
    and carries no gradient, and that pass has no backward pass; this case exists only to
    time a step with no adapter.
    """
    if form.pipeline.writers is None:
        raise ObjectiveError("consumer training needs writers")
    prepared = form.prepare(batch)
    try:
        with torch.no_grad():
            reference = form.forward(prepared, adapter_on=False, write_on=False, p=None)
    finally:
        form.end_pass()

    has_adapter = form.pipeline.adapter is not None
    try:
        off = form.forward(prepared, adapter_on=has_adapter, write_on=False, p=None)
        kl = episode_kl(reference, off, prepared.reply_mask)
        del off, reference
    except BaseException:
        form.end_pass()
        raise
    if has_adapter:
        _backward(form, batch_objective(beta * kl, step_size))
    else:
        form.end_pass()
    kl = kl.detach()

    ce = _cross_entropy_pass(form, prepared, adapter_on=has_adapter, write_on=True, p=p, step_size=step_size,
                             answer=answer)
    return EpisodeLosses(total=ce + beta * kl, cross_entropy=ce, kl=kl)


def adapter_only_step(form: TrainingForm, batch: list[TrainingExample], step_size: int) -> EpisodeLosses:
    """The adapter's loss with the write off and no writers."""
    if form.pipeline.adapter is None:
        raise ObjectiveError("adapter-only training needs an adapter")
    if form.pipeline.writers is not None:
        raise ObjectiveError("adapter-only training has no writers")
    prepared = form.prepare(batch)
    ce = _cross_entropy_pass(form, prepared, adapter_on=True, write_on=False, p=None, step_size=step_size)
    return EpisodeLosses(total=ce, cross_entropy=ce)
