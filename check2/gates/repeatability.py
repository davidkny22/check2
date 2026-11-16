"""Gate Repeatability: the Original run executed twice gives the same final answer on every episode.

    python -m check2.gates.repeatability --run <run folder> --repeat <run folder>

The two folders must hold two complete Original runs, not one run given twice, over the
same episodes in the same order, from the same settings. The gate is green when every episode's `a2` is the same in both. The
agreement, the share of episodes with the same `a2`, is recorded overall and per format,
with the episodes that differ; it is reported beside every paired difference.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from check2.evaluation.rates import rate
from check2.evaluation.tables import load_run, run_role
from check2.record import write_record

GATE_NAME = "repeatability"
LISTED = 50


def _run_facts(meta: Mapping[str, Any]) -> dict:
    return {k: meta.get(k) for k in ("run_id", "commit", "config_sha256", "order_sha256", "split", "gpu")}


def evaluate(meta_a: Mapping[str, Any], records_a: Sequence[Mapping[str, Any]], meta_b: Mapping[str, Any],
             records_b: Sequence[Mapping[str, Any]]) -> dict:
    """The agreement of the two runs' final answers and the gate's verdict."""
    if meta_a["run_id"] == meta_b["run_id"]:
        raise ValueError(f"both folders hold the run {meta_a['run_id']}; this gate compares the Original run with "
                         "its repeat")
    for meta in (meta_a, meta_b):
        role, variant = run_role(meta["config"])
        if role != "original" or variant:
            raise ValueError(f"run {meta['run_id']} is {role!r} {variant!r}; this gate compares two Original runs")
    if meta_a["config"] != meta_b["config"]:
        raise ValueError("the two runs were made from different settings")
    if meta_a["order_sha256"] != meta_b["order_sha256"]:
        raise ValueError("the two runs ran their episodes in different orders")
    ids_a = [r["episode_id"] for r in records_a]
    if ids_a != [r["episode_id"] for r in records_b]:
        raise ValueError("the two runs hold different episodes")
    differ = [a["episode_id"] for a, b in zip(records_a, records_b) if a["a2"] != b["a2"]]
    by_format: dict[str, list[int]] = {}
    for a, b in zip(records_a, records_b):
        n = by_format.setdefault(a["format"], [0, 0])
        n[0] += a["a2"] == b["a2"]
        n[1] += 1
    return {
        "agreement": rate(len(ids_a) - len(differ), len(ids_a)),
        "agreement_by_format": {fmt: rate(*n) for fmt, n in sorted(by_format.items())},
        "episodes_differing": len(differ),
        "first_differing": differ[:LISTED],
        "verdict": "green" if not differ else "red",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m check2.gates.repeatability", description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, type=Path, help="the first Original run's folder")
    parser.add_argument("--repeat", required=True, type=Path, help="the second Original run's folder")
    args = parser.parse_args(argv)
    meta_a, records_a = load_run(args.run)
    meta_b, records_b = load_run(args.repeat)
    result = evaluate(meta_a, records_a, meta_b, records_b)
    verdict = result.pop("verdict")
    path = write_record("gate", GATE_NAME, result, verdict=verdict,
                        config={"run": meta_a["run_id"], "repeat": meta_b["run_id"]},
                        extra={"runs": [_run_facts(meta_a), _run_facts(meta_b)]})
    agreement = result["agreement"]
    print(f"same a2 on {agreement['numerator']} of {agreement['denominator']} episodes "
          f"({agreement['rate']:.4f}); by format: "
          + ", ".join(f"{fmt} {r['numerator']}/{r['denominator']}" for fmt, r in result["agreement_by_format"].items()))
    if result["first_differing"]:
        print("differing: " + ", ".join(result["first_differing"]))
    print(f"gate {GATE_NAME}: {verdict}; record {path}")
    return 0 if verdict == "green" else 1


if __name__ == "__main__":
    sys.exit(main())
