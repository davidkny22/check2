"""Gate Scorer fallback: every reply has a final answer, and the appended read agrees with stated answers.

    python -m check2.gates.scorer_fallback --run <run folder>

On the run's checked sample (200 stated replies per situation in each format, or all there
are, drawn by `random.Random(2034)`), the appended read must give the stated answer on at
least 97% of the sample. The agreement per format and situation, and the share of replies
scored each way per format, are recorded. The sample is drawn again from the run's own
records, so the gate does not rely on the run's summary.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from check2.evaluation.rates import rate
from check2.evaluation.runs import appended_check
from check2.evaluation.runs.appended_check import LEAST_AGREEMENT
from check2.evaluation.tables import load_run
from check2.record import write_record

GATE_NAME = "scorer_fallback"


def evaluate(records: Sequence[Mapping[str, Any]]) -> dict:
    """The gate's values and verdict on one run's records."""
    missing = [r["episode_id"] for r in records if r.get("a2") is None]
    check = appended_check.agreement(records)
    scored: dict[str, list[int]] = {}
    for r in records:
        n = scored.setdefault(r["format"], [0, 0])
        n[0] += r["scored_by"] == "stated"
        n[1] += 1
    shares = {fmt: {"stated": rate(n[0], n[1]), "appended": rate(n[1] - n[0], n[1])} for fmt, n in sorted(scored.items())}
    overall = check["all"]["rate"]
    green = not missing and overall is not None and overall >= LEAST_AGREEMENT
    return {"replies": len(records), "replies_without_a2": len(missing), "first_without_a2": missing[:50],
            "appended_read_check": check, "scored_each_way": shares, "least_agreement": LEAST_AGREEMENT,
            "verdict": "green" if green else "red"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m check2.gates.scorer_fallback", description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, type=Path, help="the run's folder")
    args = parser.parse_args(argv)
    meta, records = load_run(args.run)
    result = evaluate(records)
    verdict = result.pop("verdict")
    path = write_record("gate", GATE_NAME, result, verdict=verdict, config={"run": meta["run_id"]},
                        extra={"run": {k: meta.get(k) for k in ("run_id", "commit", "config_sha256", "order_sha256",
                                                                 "split", "gpu")}})
    check = result["appended_read_check"]["all"]
    print(f"replies {result['replies']}, without a2 {result['replies_without_a2']}")
    for fmt, share in result["scored_each_way"].items():
        print(f"{fmt}: stated {share['stated']['numerator']}/{share['stated']['denominator']}, "
              f"appended {share['appended']['numerator']}/{share['appended']['denominator']}")
    for g in result["appended_read_check"]["groups"]:
        a = g["agreement"]
        print(f"{g['format']} {g['situation']}: appended read agrees on {a['numerator']}/{a['denominator']} "
              f"of {g['stated_replies']} stated")
    shown = "undefined" if check["rate"] is None else f"{check['rate']:.4f}"
    print(f"agreement on the sample: {check['numerator']}/{check['denominator']} ({shown}); "
          f"gate {GATE_NAME}: {verdict}; record {path}")
    return 0 if verdict == "green" else 1


if __name__ == "__main__":
    sys.exit(main())
