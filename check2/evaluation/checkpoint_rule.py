"""The rule that keeps one checkpoint, or ranks steering strengths, from their `heldout` rates.

    python -m check2.evaluation.checkpoint_rule --original <heldout Original run folder>
        --runs <heldout run folder>... --kind epochs|steering [--write-chosen]

A candidate is a dict `{label, order, wrong_adoption, right_adoption_drop}`. `order` is
the epoch or the steering strength. `wrong_adoption` is the candidate's wrong adoption on
`heldout`, reasoned pushback, both formats pooled, as a proportion. `right_adoption_drop`
is the Original run's right adoption minus the candidate's on the same episodes, in
points; a negative drop is a gain.

A candidate qualifies when its drop is at most the pass bar's limit. The qualifying
candidates come first, by wrong adoption ascending, then the others, by drop ascending.
A tie goes to the smaller `order`: the earlier epoch, the smaller strength.

The command applies the rule to `heldout` run folders and writes a measurement record of
every candidate's rates and drop, the ranking, what is kept and the recorded check that
applies. With `--kind epochs` the runs are one training's checkpoints, each named
`<config>-heldout-epoch<k>`. The training's name is computed from each run's saved settings
by the checkpoints' own naming, and runs of two trainings are refused. The first of the
ranking is kept, the check is whether it is the last epoch's, and `--write-chosen` writes
the training's name, the kept epoch and the record's file name to that training's
`data/trainings/<training-id>/chosen.json`, the file the checkpoint loader reads. With
`--kind steering` the runs are the steering strengths; the
first three of the ranking are kept, and the check is whether all three sit at one end
of the grid.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

from check2.config.schema import Baseline, Config
from check2.evaluation.readings import (
    REASONED,
    RIGHT_ADOPTION_DROP_LIMIT,
    counts,
    paired_records,
    selection,
)
from check2.evaluation.roles import base_role
from check2.evaluation.run_folders import load_run
from check2.record import write_record
from check2.training import checkpoints

MEASUREMENT_NAME = "checkpoint_rule"
EPOCH_IN_RUN_ID = re.compile(r"-epoch(\d+)$")
KEPT_STRENGTHS = 3
# The steering grid: the strengths the configuration allows.
STEERING_GRID = tuple(sorted(v for v in next(f for f in fields(Baseline) if f.name == "steering_strength")
                             .metadata["allowed"] if v is not None))
LAST_EPOCH_CHECK = "the kept checkpoint is the last epoch's"
GRID_END_CHECK = "all three kept steering strengths sit at one end of the grid"

Candidate = Mapping[str, Any]


def qualifies(candidate: Candidate) -> bool:
    """Whether the candidate's drop in right adoption is at most the limit, in points."""
    return candidate["right_adoption_drop"] <= RIGHT_ADOPTION_DROP_LIMIT


def rank(candidates: Iterable[Candidate]) -> list:
    """The candidates in the rule's order: qualifying ones by wrong adoption, then the rest by drop."""
    candidates = list(candidates)
    qualifying = sorted(
        (c for c in candidates if qualifies(c)),
        key=lambda c: (c["wrong_adoption"], c["order"]),
    )
    others = sorted(
        (c for c in candidates if not qualifies(c)),
        key=lambda c: (c["right_adoption_drop"], c["order"]),
    )
    return qualifying + others


def choose(candidates: Iterable[Candidate]) -> Candidate:
    """The candidate the rule keeps: the first in its ranking."""
    ranked = rank(candidates)
    if not ranked:
        raise ValueError("the checkpoint rule needs at least one candidate")
    return ranked[0]


def _require_not_test(records: Sequence[Mapping[str, Any]], label: str) -> None:
    for record in records:
        if record.get("part") is not None:
            raise ValueError(
                f"{label}: episode {record['episode_id']!r} is a test episode; the rule is applied on heldout only"
            )


def _pooled(records: Iterable[Mapping[str, Any]], rate_name: str) -> tuple[int, int]:
    numerator = denominator = 0
    for record in records:
        n, d = counts(record, rate_name)
        numerator += n
        denominator += d
    return numerator, denominator


def candidates_from_runs(runs: Iterable[Mapping[str, Any]], original: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Candidates from runs on `heldout`, each `{label, order, records}`, against the Original run's records.

    Each candidate is paired with Original on the same reasoned episodes of both formats;
    an episode missing from either is an error. Rates with no episode to count are refused.
    Each candidate also carries its right adoption, as a proportion.
    """
    original = list(original)
    _require_not_test(original, "original")
    candidates = []
    for run in runs:
        records = list(run["records"])
        _require_not_test(records, run["label"])
        pairs = paired_records(records, original, selection(forms=REASONED))
        wrong = _pooled((a for a, _ in pairs), "wrong_adoption")
        right = _pooled((a for a, _ in pairs), "right_adoption")
        right_original = _pooled((b for _, b in pairs), "right_adoption")
        if wrong[1] == 0 or right[1] == 0:
            raise ValueError(f"{run['label']}: no reasoned episode counts in wrong or right adoption")
        candidates.append(
            {
                "label": run["label"],
                "order": run["order"],
                "wrong_adoption": wrong[0] / wrong[1],
                "right_adoption": right[0] / right[1],
                "right_adoption_drop": 100 * right_original[0] / right_original[1] - 100 * right[0] / right[1],
            }
        )
    return candidates


def _candidate_run(meta: Mapping[str, Any], records: list[dict], kind: str) -> dict:
    """The run as a candidate of its kind, `{label, order, records}`; a run of the other kind is refused."""
    run_id = meta["run_id"]
    if meta.get("split") != "heldout":
        raise ValueError(f"run {run_id} is on {meta.get('split')!r}; the rule is applied to heldout runs")
    strength = meta["config"]["check2"]["baseline"]["steering_strength"]
    epoch = EPOCH_IN_RUN_ID.search(run_id)
    if kind == "epochs":
        if strength is not None or epoch is None:
            raise ValueError(f"run {run_id} is not one epoch's checkpoint of a training")
        return {"label": run_id, "order": int(epoch.group(1)), "records": records}
    if strength is None or epoch is not None:
        raise ValueError(f"run {run_id} is not a steering run")
    return {"label": run_id, "order": strength, "records": records}


def _training_name(settings: Mapping[str, Any]) -> str:
    """The name of the training whose checkpoint a run loaded, from the settings its `run.json` holds.

    A run's saved settings are resolved, so they are not read as a config file: the settings
    the name uses are taken from the mapping, on the schema's defaults, and named by the
    checkpoints' own rule.
    """
    try:
        raw = settings["check2"]
        writer, adapter = raw["writer"], raw["adapter"]
        base = Config()
        config = replace(base, signal=raw["signal"], seed=raw["seed"], signal_forming=raw["signal_forming"],
                         writer=replace(base.writer, alpha=writer["alpha"]),
                         adapter=replace(base.adapter, kind=adapter["kind"], rank=adapter["rank"],
                                         beta=adapter["beta"]))
    except KeyError as err:
        raise ValueError(f"a run's saved settings lack {err}, which the training's name uses") from err
    return checkpoints.training_id(config)


def write_chosen(training_id: str, epoch: int, record_name: str) -> Path:
    """Write the kept epoch to the training's chosen-epoch file, the one the checkpoint loader reads; its path."""
    path = checkpoints.CHOSEN_ROOT / training_id / checkpoints.CHOSEN_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"training_id": training_id, "epoch": epoch, "record": record_name}) + "\n",
                    encoding="utf-8", newline="\n")
    return path


def apply_rule(original: tuple[Mapping[str, Any], list[dict]], runs: Sequence[tuple[Mapping[str, Any], list[dict]]],
               kind: str) -> dict:
    """The rule on `heldout` runs of one kind against the Original run: candidates, ranking, what is kept, the check."""
    original_meta, original_records = original
    if original_meta.get("split") != "heldout" or base_role(original_meta["config"]) != "original":
        raise ValueError(f"run {original_meta['run_id']} is not an Original run over heldout")
    chosen = [_candidate_run(meta, records, kind) for meta, records in runs]
    orders = [c["order"] for c in chosen]
    if len(set(orders)) != len(orders):
        raise ValueError(f"two runs share an {'epoch' if kind == 'epochs' else 'strength'}: {sorted(orders)}")
    names = sorted({_training_name(meta["config"]) for meta, _ in runs}) if kind == "epochs" else []
    if len(names) > 1:
        raise ValueError(f"the runs are checkpoints of more than one training: {names}")
    if kind == "epochs" and len({json.dumps(meta["config"], sort_keys=True) for meta, _ in runs}) != 1:
        raise ValueError("the runs are checkpoints of more than one training")
    candidates = candidates_from_runs(chosen, original_records)
    ranking = rank(candidates)
    values: dict[str, Any] = {
        "kind": kind,
        "original": original_meta["run_id"],
        "candidates": [{**c, "qualifies": qualifies(c)} for c in candidates],
        "ranking": [c["label"] for c in ranking],
    }
    if kind == "epochs":
        values["training_id"] = names[0]
        kept = ranking[0]
        values["kept"] = {"label": kept["label"], "epoch": kept["order"]}
        values["recorded_check"] = {"check": LAST_EPOCH_CHECK, "holds": kept["order"] == max(orders)}
    else:
        if len(ranking) < KEPT_STRENGTHS:
            raise ValueError(f"the rule keeps {KEPT_STRENGTHS} steering strengths; {len(ranking)} were given")
        kept = ranking[:KEPT_STRENGTHS]
        strengths = sorted(c["order"] for c in kept)
        ends = (list(STEERING_GRID[:KEPT_STRENGTHS]), list(STEERING_GRID[-KEPT_STRENGTHS:]))
        values["kept"] = {"labels": [c["label"] for c in kept], "strengths": strengths}
        values["recorded_check"] = {"check": GRID_END_CHECK, "holds": strengths in ends}
    return values


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m check2.evaluation.checkpoint_rule",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--original", required=True, type=Path, help="the Original run's folder, on heldout")
    parser.add_argument("--runs", required=True, nargs="+", type=Path, help="the candidates' run folders, on heldout")
    parser.add_argument("--kind", required=True, choices=("epochs", "steering"))
    parser.add_argument("--write-chosen", action="store_true",
                        help="with --kind epochs: write the kept epoch to the chosen-epoch file of the runs' training")
    args = parser.parse_args(argv)
    if args.write_chosen and args.kind != "epochs":
        parser.error("--write-chosen names the kept epoch; it is given with --kind epochs only")
    original = load_run(args.original)
    runs = [load_run(folder) for folder in args.runs]
    values = apply_rule(original, runs, args.kind)
    facts = {meta["run_id"]: {k: meta.get(k) for k in ("commit", "config_sha256", "order_sha256", "gpu")}
             for meta, _ in [original, *runs]}
    path = write_record("measurement", MEASUREMENT_NAME, values,
                        config={"kind": args.kind, "original": original[0]["run_id"],
                                "runs": [meta["run_id"] for meta, _ in runs]},
                        extra={"runs": facts})
    for c in values["candidates"]:
        print(f"{c['label']}: wrong adoption {c['wrong_adoption']:.4f}, right adoption {c['right_adoption']:.4f}, "
              f"drop {c['right_adoption_drop']:.2f} points{'' if c['qualifies'] else ', does not qualify'}")
    print("ranking: " + ", ".join(values["ranking"]))
    check = values["recorded_check"]
    print(f"kept: {values['kept']}; {check['check']}: {check['holds']}; record {path}")
    if args.write_chosen:
        print(f"wrote {write_chosen(values['training_id'], values['kept']['epoch'], path.name)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
