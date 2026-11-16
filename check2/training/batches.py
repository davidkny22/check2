"""Forward batches of a training step, sorted by length and sized to a token budget, and that budget's measurement.

One optimiser step covers its episodes in forward batches sorted by length, each holding
as many rows as fit the token budget, with gradients accumulated across them. A batch's
tokens are its rows times its padded width in the form training uses. The budget is the
largest such count that fits the card, measured on the session in that form with the
consumer step, which holds the reference logits and the KL term's buffers as training
does:

    python -m check2.training.batches --model full --form shared_prefix [--adapter hooks]

The measurement runs consumer steps on the longest short-format training episodes,
doubling the rows until the card runs out of memory, then halving the gap, and writes the
largest count that fitted as a measurement record.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable

import torch

from check2.record import write_record
from check2.training.examples import TrainingExample

NAME = "token_budget"
# The measurement's ceiling on rows; far above what a card of 96 GB holds at the longest replies.
MAX_ROWS = 256


class BatchError(ValueError):
    """An episode that does not fit the budget even alone."""


def plan_batches(episodes: list[TrainingExample], token_budget: float,
                 batch_tokens: Callable[[list[TrainingExample]], int] | None = None,
                 max_rows: int | None = None) -> list[list[TrainingExample]]:
    """The step's episodes in forward batches, sorted by length, each within the budget and the row limit.

    `batch_tokens` gives a batch's rows times its padded width in the form training uses;
    by default, the whole padded sequence.
    """
    count = batch_tokens or (lambda batch: len(batch) * max(len(e.sequence_ids) for e in batch))
    ordered = sorted(episodes, key=lambda e: (count([e]), e.episode_id))
    out: list[list[TrainingExample]] = []
    current: list[TrainingExample] = []
    for e in ordered:
        if count([e]) > token_budget:
            raise BatchError(f"{e.episode_id} carries {count([e])} tokens alone, above the budget of {token_budget}")
        candidate = current + [e]
        if current and (count(candidate) > token_budget or (max_rows is not None and len(candidate) > max_rows)):
            out.append(current)
            candidate = [e]
        current = candidate
    if current:
        out.append(current)
    return out


def _clear(parameters) -> None:
    for parameter in parameters:
        parameter.grad = None
    gc.collect()
    torch.cuda.empty_cache()


def measure_budget(form, examples: list[TrainingExample], beta: float, max_rows: int = MAX_ROWS) -> dict:
    """The largest token count of a consumer step's forward batch that fits the card, in this form."""
    from check2.training.objectives import consumer_step

    if not examples:
        raise BatchError("no examples to measure with")
    ordered = sorted(examples, key=lambda e: (-form.batch_tokens([e]), e.episode_id))
    pipeline = form.pipeline
    trained = list(pipeline.writers.parameters()) + (pipeline.adapter.parameters() if pipeline.adapter else [])
    trials = []

    def batch_of(rows: int) -> list[TrainingExample]:
        return [ordered[i % len(ordered)] for i in range(rows)]

    def fits(rows: int) -> bool:
        batch = batch_of(rows)
        p = torch.full((rows, 3), 1.0 / 3, device=pipeline.device)
        _clear(trained)
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        try:
            consumer_step(form, batch, p, beta, rows)
            torch.cuda.synchronize()
            ok = True
        except torch.cuda.OutOfMemoryError:
            ok = False
        trials.append({"rows": rows, "tokens": form.batch_tokens(batch), "fits": ok,
                       "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                       "seconds": round(time.time() - started, 3)})
        _clear(trained)
        return ok

    if not fits(1):
        raise BatchError("the longest episode does not fit the card alone")
    low, high = 1, None
    while high is None and low < max_rows:
        rows = min(2 * low, max_rows)
        if fits(rows):
            low = rows
        else:
            high = rows
    while high is not None and high - low > 1:
        middle = (low + high) // 2
        if fits(middle):
            low = middle
        else:
            high = middle
    best = batch_of(low)
    return {"form": form.name, "token_budget": form.batch_tokens(best), "rows": low,
            "width": form.batch_tokens(best) // low, "first_failing_rows": high,
            "first_failing_tokens": None if high is None else form.batch_tokens(batch_of(high)),
            "reached_row_ceiling": high is None, "trials": trials}


def main(argv=None) -> int:
    from check2.gates import mechanism_setup as setup
    from check2.training import bench as training_bench
    from check2.training.forms import FORMS

    args = training_bench.parser(__doc__.splitlines()[0]).parse_args(argv)
    started = time.time()
    loaded = setup.load(args.model, exact=False)
    pipeline = training_bench.training_pipeline(loaded, args.adapter)
    episodes = training_bench.candidate_episodes(args, "short")
    examples = training_bench.examples(episodes, loaded.tokenizer, args.targets)
    form = FORMS[args.form](pipeline)
    values = measure_budget(form, examples, training_bench.BETA)
    values["adapter"] = args.adapter
    record = write_record("measurement", NAME, values, extra={
        "model_kind": args.model, "dtype": str(loaded.dtype).replace("torch.", ""),
        "builtin_path": loaded.builtin_path, "tail_position": args.tail_position,
        "seconds": round(time.time() - started, 1)})
    print({k: v for k, v in values.items() if k != "trials"})
    for trial in values["trials"]:
        print(trial)
    print(f"record: {record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
