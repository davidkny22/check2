"""One optimiser step over a step's episodes, and the two ways a step stops the run.

A step runs its episodes as forward batches within the token budget, so that gradients
accumulate across them, clips the gradients of every trained parameter together, and
takes one optimiser step. Its loss does not depend on how it is split into forward
batches.

When a forward batch runs out of memory, the step frees what it holds, clears the
gradients it has accumulated, halves the token budget for this step only, and runs again
from its start with the same episodes; the optimiser has not stepped. It may halve again.
When a batch of one row runs out of memory, or a row no longer fits the halved budget, the
run stops. A loss or a gradient norm that is not finite also stops the run, before the
optimiser step.
"""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass, field

import torch

from check2.training.answer_token import AnswerWeighting
from check2.training.batches import BatchError, plan_batches
from check2.training.examples import TrainingExample
from check2.training.forms import TrainingForm
from check2.training.kinds import Kind, TrainingError
from check2.training.objectives import adapter_only_step, connector_step, consumer_step
from check2.training.schedule import clip


class StepStop(TrainingError):
    """A step that stops the run; the last checkpoint is kept."""


class NonFiniteLoss(StepStop):
    """A step whose loss or gradient norm is not finite."""


class OutOfMemory(StepStop):
    """A step that runs out of memory with one row in a forward batch."""


@dataclass
class StepLosses:
    """One step's losses, each the mean over its episodes, the gradient norm before clipping,
    and each out-of-memory event: the budget and the rows of the batch that ran out."""

    total: float
    cross_entropy: float
    kl: float | None
    gradient_norm: float
    out_of_memory: list[dict] = field(default_factory=list)


def _attempt(kind: Kind, form: TrainingForm, examples: list[TrainingExample], signals: dict | None, beta: float,
             token_budget: float, failed: dict, answer: AnswerWeighting | None = None) -> dict:
    """Run the step's forward batches, accumulating gradients; returns the summed episode losses."""
    size = len(examples)
    sums = {"total": 0.0, "cross_entropy": 0.0, "kl": 0.0}
    for part in plan_batches(examples, token_budget, form.batch_tokens):
        failed["rows"] = len(part)
        p = None
        if signals is not None:
            p = torch.stack([signals[e.episode_id] for e in part]).to(form.pipeline.device)
        if kind.name == "connector":
            losses = connector_step(form, part, p, size)
        elif kind.name == "consumer":
            losses = consumer_step(form, part, p, beta, size, answer)
        else:
            losses = adapter_only_step(form, part, size)
        sums["total"] += float(losses.total.double().sum())
        sums["cross_entropy"] += float(losses.cross_entropy.double().sum())
        if losses.kl is not None:
            sums["kl"] += float(losses.kl.double().sum())
    return sums


def _free(form: TrainingForm, optimizer: torch.optim.Optimizer) -> None:
    """Clear the step's accumulated gradients and the control state, and return the freed memory to the card."""
    optimizer.zero_grad(set_to_none=True)
    form.end_pass()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_step(kind: Kind, form: TrainingForm, examples: list[TrainingExample], signals: dict | None, beta: float,
             token_budget: float, optimizer: torch.optim.Optimizer, schedule,
             answer: AnswerWeighting | None = None) -> StepLosses:
    """One optimiser step over a step's episodes, in forward batches within the budget.

    The budget is halved for this step each time a forward batch runs out of memory. With
    `answer`, consumer training's cross-entropy is the answer-weighted one.
    """
    size = len(examples)
    budget = token_budget
    events: list[dict] = []
    optimizer.zero_grad(set_to_none=True)
    while True:
        failed: dict = {}
        out_of_memory = False
        try:
            sums = _attempt(kind, form, examples, signals, beta, budget, failed, answer)
        except torch.OutOfMemoryError:
            out_of_memory = True
        # Memory is freed outside the handler, so that the error's frames no longer hold it.
        if not out_of_memory:
            break
        _free(form, optimizer)
        events.append({"token_budget": budget, "rows": failed.get("rows")})
        if failed.get("rows") == 1:
            raise OutOfMemory(f"a forward batch of one row ran out of memory at a token budget of {budget}")
        budget = budget / 2
        try:
            plan_batches(examples, budget, form.batch_tokens)
        except BatchError as err:
            raise OutOfMemory(f"after running out of memory, a row does not fit half the budget: {err}") from err
    out = StepLosses(total=sums["total"] / size, cross_entropy=sums["cross_entropy"] / size,
                     kl=sums["kl"] / size if kind.name == "consumer" else None, gradient_norm=math.nan,
                     out_of_memory=events)
    if not math.isfinite(out.total):
        raise NonFiniteLoss(f"the step's loss is {out.total}")
    out.gradient_norm = float(clip(optimizer))
    if not math.isfinite(out.gradient_norm):
        raise NonFiniteLoss(f"the step's gradient norm is {out.gradient_norm}")
    optimizer.step()
    schedule.step()
    return out
