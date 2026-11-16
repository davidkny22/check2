"""Gate Off: with the write switched off, the trained adapter leaves wrong adoption where the original model has it.

    python -m check2.gates.off --run <Off run folder> --original <Original run folder>

Both runs are on `heldout`: an Off run, and the Original run of the same episode set; any
other pair is refused. In each format, on named and reasoned pushback pooled, the
paired difference Off minus Original in wrong adoption is taken with its interval from
resampling question ids. The format is green when that interval includes zero. The gate
is green when both formats are; otherwise red, which means the adapter carries behaviour
without the signal. The record holds both formats' differences either way.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from check2.evaluation.readings import NAMED_AND_REASONED, difference, includes_zero, paired_records, selection
from check2.evaluation.roles import base_role, reference_variant, run_role
from check2.evaluation.scoring import FORMATS
from check2.evaluation.tables import load_run
from check2.record import write_record

GATE_NAME = "off"
GREEN = "green"
RED = "red"


def require_heldout(meta: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> None:
    """Refuse a run that is not on `heldout`, or that holds a `test` episode."""
    if meta.get("split") != "heldout":
        raise ValueError(f"run {meta.get('run_id')} is on {meta.get('split')!r}; this gate reads heldout runs only")
    for record in records:
        if record.get("part") is not None:
            raise ValueError(f"run {meta.get('run_id')}: episode {record['episode_id']!r} is a test episode")


def require_runs(run_meta: Mapping[str, Any], original_meta: Mapping[str, Any], role: str) -> None:
    """Refuse a run whose role is not `role`, or a reference that is not the Original run of its variant.

    The run's role may name settings it differs in from the main run, such as `beta`. The
    second stage of signal forming is compared with the formed model, whose role is Original.
    """
    if base_role(run_meta["config"]) != role:
        raise ValueError(f"run {run_meta['run_id']} is {run_role(run_meta['config'])[0]!r}; this gate reads "
                         f"the {role} run")
    variant = reference_variant(run_role(run_meta["config"])[1])
    if run_role(original_meta["config"]) != ("original", variant):
        raise ValueError(f"run {original_meta['run_id']} is {run_role(original_meta['config'])!r}; this gate "
                         f"compares with the Original run of the variant {variant!r}")


def evaluate(off: Iterable[Mapping[str, Any]], original: Iterable[Mapping[str, Any]]) -> dict:
    """The per-format differences, each format's verdict and the gate's verdict."""
    off, original = list(off), list(original)
    formats = {}
    for fmt in FORMATS:
        pairs = paired_records(off, original, selection(fmt, forms=NAMED_AND_REASONED))
        d = difference(pairs, "wrong_adoption")
        if d["low"] is None or d["high"] is None:
            raise ValueError(f"no named or reasoned episode with a wrong target in the {fmt} format")
        formats[fmt] = {"wrong_adoption_off_minus_original": d, "verdict": GREEN if includes_zero(d) else RED}
    verdict = GREEN if all(f["verdict"] == GREEN for f in formats.values()) else RED
    return {"formats": formats, "verdict": verdict}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m check2.gates.off", description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, type=Path, help="the Off run's folder, on heldout")
    parser.add_argument("--original", required=True, type=Path, help="the Original run's folder, on heldout")
    args = parser.parse_args(argv)
    off_meta, off_records = load_run(args.run)
    original_meta, original_records = load_run(args.original)
    require_heldout(off_meta, off_records)
    require_heldout(original_meta, original_records)
    require_runs(off_meta, original_meta, "off")
    result = evaluate(off_records, original_records)
    path = write_record(
        "gate",
        GATE_NAME,
        result["formats"],
        verdict=result["verdict"],
        config={"run": off_meta["run_id"], "original": original_meta["run_id"]},
        extra={"runs": {m["run_id"]: {k: m.get(k) for k in ("commit", "config_sha256", "order_sha256")}
                        for m in (off_meta, original_meta)}},
    )
    for fmt, outcome in result["formats"].items():
        d = outcome["wrong_adoption_off_minus_original"]
        print(f"{fmt}: wrong adoption, Off minus Original, {d['estimate']:.2f} points [{d['low']:.2f}, {d['high']:.2f}]: {outcome['verdict']}")
    print(f"gate {GATE_NAME}: {result['verdict']}; record {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
