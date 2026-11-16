"""Saved state equals live state: the state training read from the cache is the state a run computes live.

    python -m check2.gates.saved_state --run <heldout run folder> [--saved <states file>] [--plant]

For every episode of a complete `heldout` run of the wide-signal variant, the cosine between
the state saved from the reader's cache (`python -m check2.reader.wide_states --split
heldout`) and the state the run computed live and saved for the same episode, in float32.
The lowest cosine is recorded. Green when it is at least 0.99 and every episode of the run
has both states. The planted fault rotates the saved file's rows by one episode, so that
each episode is compared with another's saved state; the gate must then go red.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from check2.config.schema import WIDE_WRITER_INPUT
from check2.evaluation.runs import execution
from check2.evaluation.tables import load_run
from check2.reader import wide_states
from check2.record import write_record

NAME = "saved_state"
LOWEST_COSINE = 0.99


class SavedStateError(ValueError):
    """A run that is not a complete held-out run of the wide-signal variant."""


def _rotated(saved: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    ids = sorted(saved)
    return {eid: saved[ids[(i + 1) % len(ids)]] for i, eid in enumerate(ids)}


def check(run_folder: Path, saved_file: Path, plant: bool = False) -> dict:
    meta, records = load_run(run_folder)
    settings = meta["config"].get("check2", meta["config"])
    if meta.get("split") != "heldout" or settings.get("writer", {}).get("input") != WIDE_WRITER_INPUT:
        raise SavedStateError(f"{run_folder.name} is not a heldout run of the wide-signal variant")
    live = execution.load_wide_states(run_folder.name)
    saved = wide_states.load_states(saved_file)
    if plant:
        saved = _rotated(saved)
    ids = [r["episode_id"] for r in records]
    missing_saved = sorted(e for e in ids if e not in saved)
    missing_live = sorted(e for e in ids if e not in live)
    lowest, lowest_id, compared = None, None, 0
    for eid in ids:
        if eid in saved and eid in live:
            a, b = saved[eid].to(torch.float32), live[eid].to(torch.float32)
            cosine = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
            compared += 1
            if lowest is None or cosine < lowest:
                lowest, lowest_id = cosine, eid
    green = (not missing_saved and not missing_live and compared == len(ids) and lowest is not None
             and lowest >= LOWEST_COSINE)
    return {"run": run_folder.name, "saved_file": str(saved_file), "episodes": len(ids), "compared": compared,
            "lowest_cosine": lowest, "episode_of_lowest": lowest_id, "threshold": LOWEST_COSINE,
            "missing_saved": missing_saved[:20], "missing_saved_count": len(missing_saved),
            "missing_live": missing_live[:20], "missing_live_count": len(missing_live), "green": green}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True, help="the folder of a complete heldout run of the variant")
    parser.add_argument("--saved", type=Path, default=wide_states.states_file("heldout"))
    parser.add_argument("--plant", action="store_true", help="rotate the saved file's rows; the gate must go red")
    args = parser.parse_args(argv)
    values = check(args.run, args.saved, args.plant)
    verdict = "green" if values["green"] else "red"
    record = write_record("gate", NAME, values, verdict=verdict, extra={"planted": bool(args.plant)})
    print(json.dumps(values, indent=1))
    print(f"verdict: {verdict}  record: {record}")
    return 0 if values["green"] else 1


if __name__ == "__main__":
    sys.exit(main())
