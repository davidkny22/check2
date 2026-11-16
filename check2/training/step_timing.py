"""The time of one consumer training step of 16 short-format episodes, with a given adapter and form.

    python -m check2.training.step_timing --adapter none|hooks|wrappers|library --form <name>
        [--model full] [--token-budget N]

The episodes are `named` and `reasoned` `train` episodes with a target reply, drawn as a
gate draws from a split. The step runs as forward batches within the token budget (one
batch when none is given), with the frozen model in bf16 on the DeltaNet path of this
process and the writers and adapter at their initial values. Two warm-up steps run first;
the third is timed and written as a measurement record.
"""

from __future__ import annotations

import time

import torch

from check2.config.schema import Config
from check2.gates import mechanism_setup as setup
from check2.record import write_record
from check2.training import bench as training_bench
from check2.training.batches import plan_batches
from check2.training.forms import FORMS
from check2.training.objectives import consumer_step

NAME = "training_step_time"
EPISODES = Config().training.episodes_per_step
WARM_UP_STEPS = 2


def _synchronise() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_step(form, examples, beta: float, token_budget: float) -> dict:
    """Run two warm-up steps and time a third; returns the seconds, the batches and the peak memory."""
    batches = plan_batches(examples, token_budget, form.batch_tokens)
    pipeline = form.pipeline
    trained = list(pipeline.writers.parameters()) + (pipeline.adapter.parameters() if pipeline.adapter else [])
    # The signal's values change no cost; every row gets the uniform one.
    signals = [torch.full((len(b), 3), 1.0 / 3, device=pipeline.device) for b in batches]
    seconds = []
    for _ in range(WARM_UP_STEPS + 1):
        for parameter in trained:
            parameter.grad = None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        _synchronise()
        started = time.perf_counter()
        for batch, p in zip(batches, signals):
            consumer_step(form, batch, p, beta, len(examples))
        _synchronise()
        seconds.append(time.perf_counter() - started)
    return {"seconds": round(seconds[-1], 4), "warm_up_seconds": [round(s, 4) for s in seconds[:-1]],
            "batches": [{"rows": len(b), "tokens": form.batch_tokens(b)} for b in batches],
            "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None}


def main(argv=None) -> int:
    parser = training_bench.parser(__doc__.splitlines()[0])
    parser.add_argument("--token-budget", type=int, default=None, help="the forward batches' token budget")
    args = parser.parse_args(argv)
    started = time.time()
    loaded = setup.load(args.model, exact=False)
    pipeline = training_bench.training_pipeline(loaded, args.adapter)
    episodes = training_bench.drawn(training_bench.candidate_episodes(args, "short"), EPISODES)
    examples = training_bench.examples(episodes, loaded.tokenizer, args.targets)
    form = FORMS[args.form](pipeline)
    budget = float("inf") if args.token_budget is None else args.token_budget
    values = time_step(form, examples, training_bench.BETA, budget)
    values.update(adapter=args.adapter, form=args.form, episodes=len(examples), token_budget=args.token_budget,
                  episode_ids=[e.episode_id for e in examples])
    record = write_record("measurement", NAME, values, extra={
        "model_kind": args.model, "dtype": str(loaded.dtype).replace("torch.", ""),
        "builtin_path": loaded.builtin_path, "tail_position": args.tail_position,
        "seconds": round(time.time() - started, 1)})
    print({k: v for k, v in values.items() if k != "episode_ids"})
    print(f"record: {record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
