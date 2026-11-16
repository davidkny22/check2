"""The `p` a run computed live against the `p` saved from the reader's cached states, episode by episode.

    python -m check2.evaluation.runs.live_signal --run <run folder> [--saved data/reader/p_heldout.jsonl]

The live window is cut from the run's own call over tokens `0..r`, the saved one from the
reader's cache; they are built by different code, so they must agree. The record holds the
largest absolute difference over every output of every episode the two share, the number of
episodes beyond the tolerance, and whether every episode of the run had a saved `p`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from check2.data import paths
from check2.evaluation.tables import load_run
from check2.record import write_record

TOLERANCE = 1e-3
SAVED_HELDOUT = paths.DATA / "reader" / "p_heldout.jsonl"
LISTED = 20


def compare(records: Sequence[Mapping], saved: Mapping[str, Sequence[float]]) -> dict:
    missing = [r["episode_id"] for r in records if r["episode_id"] not in saved]
    worst, beyond = 0.0, []
    for r in records:
        if r["episode_id"] not in saved:
            continue
        difference = max(abs(a - b) for a, b in zip(r["p"], saved[r["episode_id"]]))
        worst = max(worst, difference)
        if difference > TOLERANCE:
            beyond.append({"episode_id": r["episode_id"], "difference": difference})
    beyond.sort(key=lambda x: -x["difference"])
    return {"episodes": len(records), "without_saved_p": len(missing), "max_abs_difference": worst,
            "beyond_tolerance": len(beyond), "largest": beyond[:LISTED], "tolerance": TOLERANCE,
            "agree": not missing and not beyond}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m check2.evaluation.runs.live_signal",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--saved", type=Path, default=SAVED_HELDOUT)
    args = parser.parse_args(argv)
    meta, records = load_run(args.run)
    with args.saved.open(encoding="utf-8") as f:
        saved = {row["episode_id"]: row["p"] for row in (json.loads(line) for line in f if line.strip())}
    values = compare(records, saved)
    path = write_record("measurement", "live_signal", values, config={"run": meta["run_id"], "saved": str(args.saved)},
                        extra={"run": {k: meta.get(k) for k in ("run_id", "commit", "order_sha256", "split", "gpu")}})
    print(f"{values['episodes']} episodes, {values['without_saved_p']} without a saved p; largest difference "
          f"{values['max_abs_difference']:.3g}; {values['beyond_tolerance']} beyond {TOLERANCE}; record {path}")
    return 0 if values["agree"] else 1


if __name__ == "__main__":
    sys.exit(main())
