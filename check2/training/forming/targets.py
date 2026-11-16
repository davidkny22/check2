"""Target replies sampled from the formed model, for the second stage of signal forming.

    python -m check2.training.forming.targets <config> [--limit N] [--samples N] [--cells FILE]
        [--epoch K] [--scratch-tag TAG] [--batch N] [--batch-tokens FMT=N,...]

The sampling and the selection of the original model's targets run unchanged, with the
frozen forming update attached to the model so that it acts on the tokens of each prompt
up to the read token: the same episodes, sampler settings, batches and seeds, into
`data/targets/formed/`. `--cells` samples the listed (situation, format) cells again, as the
original sampling does for a cell that drops too many, into the resampled file. The
selection then writes `targets.jsonl` and `report.json` there.
"""

from __future__ import annotations

import argparse
import json

from check2.data import paths, sampling, select_targets
from check2.mechanism.loading import FULL_LAYOUT

FOLDER = paths.TARGETS.parent / "formed"
SAMPLES = FOLDER / sampling.SAMPLES.name
RESAMPLED = FOLDER / sampling.RESAMPLED.name
TARGETS = FOLDER / select_targets.TARGETS_NAME


def sample_formed(model, tokenizer, update, tail_position: int, batches: list[list[dict]], n_samples: int,
                  seed: int, out=SAMPLES) -> dict:
    """The sampler's `sample_all` with the forming update acting on every prompt; returns its timing."""
    from check2.training.forming.attach import follow_prompt_tail

    hooks = follow_prompt_tail(update, model, tail_position)
    try:
        return sampling.sample_all(model, tokenizer, batches, n_samples, seed, out)
    finally:
        hooks.remove()


def main(argv: list[str] | None = None) -> int:
    from check2.config.schema import load_config
    from check2.record import write_record
    from check2.training.forming import training
    from check2.training.forming.attach import load_frozen

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", help="a config of signal forming")
    parser.add_argument("--limit", type=int, default=None, help="keep the first N episodes of each format")
    parser.add_argument("--samples", type=int, default=None, help="replies per episode; the config's by default")
    parser.add_argument("--cells", default=None, help="JSON list of {situation, format} to sample again")
    parser.add_argument("--epoch", type=int, default=training.EPOCHS, help="the forming checkpoint to load")
    parser.add_argument("--scratch-tag", default=None, help="load a development run's forming checkpoint")
    parser.add_argument("--batch", type=int, default=sampling.BATCH_EPISODES)
    parser.add_argument("--batch-tokens", type=sampling.parse_budgets, default=None, metavar="FMT=N,...")
    args = parser.parse_args(argv)
    config = training.forming_config(load_config(args.config))
    factors = training.frozen_factors(config, args.epoch, training.storage_for(args.scratch_tag))
    cells = sampling.load_cells(args.cells) if args.cells else None
    episodes = sampling.training_episodes(cells=cells)
    n_samples = args.samples if args.samples is not None else config.data.target_samples
    batches = sampling.plan_batches(episodes, args.batch, args.limit, args.batch_tokens)
    revision = sampling.model_revision()
    model, tokenizer = sampling.load_original_model(revision)
    update = load_frozen(model, FULL_LAYOUT, config.adapter.implementation, factors)
    try:
        per_format = sample_formed(model, tokenizer, update, config.reader.read_position, batches, n_samples,
                                   config.seed, RESAMPLED if cells else SAMPLES)
    finally:
        update.remove()
    select_targets.main(["--folder", str(FOLDER)])
    values = {"per_format": per_format, "training_id": training.training_id(config), "forming_epoch": args.epoch,
              "update_factors_sha256": update.factors_sha256(), "limit": args.limit,
              "cells": sorted(map(list, cells)) if cells else None,
              "report": json.loads((FOLDER / select_targets.REPORT_NAME).read_text(encoding="utf-8"))}
    record = write_record("measurement", "formed_target_sampling", values,
                          config={"model_revision": revision, "temperature": sampling.TEMPERATURE,
                                  "top_p": sampling.TOP_P, "top_k": sampling.TOP_K, "samples": n_samples,
                                  "seed": config.seed, "batch": args.batch, "batch_tokens": args.batch_tokens})
    print(f"wrote {RESAMPLED if cells else SAMPLES} and {TARGETS}; record {record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
