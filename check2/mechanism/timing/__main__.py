"""The command line of the timing pass; see the package's description."""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch

from check2.mechanism.adapter.implementations import IMPLEMENTATIONS
from check2.mechanism.timing import __doc__ as DESCRIPTION
from check2.mechanism.timing.batch import CAP, final_answers, first_batch, generate, load, make_pipeline, repeats
from check2.mechanism.timing.settings import SETTINGS, setting_on
from check2.mechanism.timing.step_profile import profile_decoding_step
from check2.record import write_record

RECORD_NAMES = {"generation": "generation_timing", "profile": "decoding_step_profile", "setting": "decode_setting"}


def run_generation(model: str, implementation: str, limit: int | None) -> dict:
    loaded, tokenizer, layout = load(model)
    batch = first_batch(limit)
    pipeline = make_pipeline(loaded, tokenizer, layout, implementation)
    try:
        with torch.inference_mode():
            values, _ = repeats(pipeline, batch)
    finally:
        pipeline.close()
    return {"implementation": implementation, "episodes": len(batch), **values}


def run_profile(model: str, limit: int | None) -> dict:
    """The decoding step in the middle of the reply cap, after a warm-up run of the batch, with no adapter."""
    loaded, tokenizer, layout = load(model)
    batch = first_batch(limit)
    pipeline = make_pipeline(loaded, tokenizer, layout, "none")
    try:
        generate(pipeline, batch)
        values = profile_decoding_step(pipeline, batch, CAP // 2, CAP)
    finally:
        pipeline.close()
    return {"implementation": "none", "episodes": len(batch), **values}


def run_setting(model: str, setting: str, limit: int | None) -> dict:
    loaded, tokenizer, layout = load(model)
    batch = first_batch(limit)
    pipeline = make_pipeline(loaded, tokenizer, layout, "none")
    values: dict = {"setting": setting, "episodes": len(batch)}
    try:
        with torch.inference_mode():
            values["off"], default_replies = repeats(pipeline, batch)
            default_answers = final_answers(pipeline, batch, default_replies)
            try:
                with setting_on(setting, pipeline, batch, CAP) as room:
                    values["on"], replies = repeats(pipeline, batch)
                    answers = final_answers(pipeline, batch, replies, room)
            except Exception as error:  # a setting that cannot run is a result, recorded with its error
                values["error"] = f"{type(error).__name__}: {error}"
                return values
    finally:
        pipeline.close()
    differing = [e["episode_id"] for e, x, y in zip(batch, default_answers, answers) if x != y]
    values.update(final_answers_differing=len(differing), differing_episodes=differing,
                  replies_identical_to_default=values["on"]["replies_sha256"] == values["off"]["replies_sha256"],
                  time_change=(values["on"]["mean_seconds"] - values["off"]["mean_seconds"])
                  / values["off"]["mean_seconds"])
    return values


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=DESCRIPTION.splitlines()[0])
    commands = out.add_subparsers(dest="command", required=True)
    for name in RECORD_NAMES:
        command = commands.add_parser(name)
        command.add_argument("--model", choices=("tiny", "full"), default="full")
        command.add_argument("--limit", type=int, default=None, help="take only the first N episodes of the batch")
        if name == "generation":
            command.add_argument("--implementation", choices=["none", *sorted(IMPLEMENTATIONS)], required=True)
        if name == "setting":
            command.add_argument("--setting", choices=SETTINGS, required=True)
    return out


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    started = time.time()
    if args.command == "generation":
        values = run_generation(args.model, args.implementation, args.limit)
    elif args.command == "profile":
        values = run_profile(args.model, args.limit)
    else:
        values = run_setting(args.model, args.setting, args.limit)
    extra = {"model_kind": args.model, "limit": args.limit, "seconds": round(time.time() - started, 1)}
    record = write_record("measurement", RECORD_NAMES[args.command], values, extra=extra)
    print(json.dumps({"measurement": RECORD_NAMES[args.command], **extra, "values": values}, indent=1, default=str))
    print(f"record: {record}")
    return 1 if "error" in values else 0


if __name__ == "__main__":
    sys.exit(main())
