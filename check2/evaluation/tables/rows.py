"""The rows of each table: rates, differences from the Original run, the pass bar, the readings and snap against short."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from check2.evaluation import readings
from check2.evaluation.derived_runs import repeat_agreement
from check2.evaluation.rates import rates
from check2.evaluation.roles import FORMED_MODEL, FORMED_WITH_CHANNEL, reference_variant
from check2.evaluation.run_folders import require_comparable
from check2.evaluation.scoring import FORMATS, FORMS, SITUATIONS
from check2.evaluation.tables.scorer import needs_stated_rows
from check2.training.forming.excess import excess

RATE_METRICS = ("wrong_adoption", "right_adoption", "correct_end", "moved", "scored_stated", "scored_appended", "not_ended")
SUBSETS = (("all questions", False), ("wrong-first questions", True))
REPLIES = (("all replies", False), ("stated answers only", True))


def _scored(run: Mapping[str, Any], records: Sequence[Mapping[str, Any]] | None = None) -> list:
    """The records a comparison with `run` counts: a Swap run's `recheck` episodes are left out at scoring.

    A Swap run is told by its base role, so the wide-signal variant's Swap run counts as one.

    With `records` given, those records are restricted the same way, so that a partner
    run is compared on the Swap run's episodes.
    """
    records = run["records"] if records is None else records
    if run.get("base", run["role"]) == "swap":
        return [r for r in records if r["situation"] != "recheck"]
    return list(records)


def _is_reference_only(run: Mapping[str, Any]) -> bool:
    """An Original run compared with nothing: every Original run but the formed model."""
    return run["role"] == "original" and reference_variant(run["variant"]) == run["variant"]


def _name(run: Mapping[str, Any]) -> str:
    """The run's role, with the formed model named as such rather than by its role, Original."""
    return "formed model" if run["role"] == "original" and FORMED_MODEL in run["variant"] else run["role"]


def _original_for(run: Mapping[str, Any], runs: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The run's reference: the run it names, or the Original run over the same split and episodes.

    That Original run covers the run's own episode set; for the second stage of signal
    forming it is the formed model, and for the formed model the main episodes' Original run.
    """
    if "reference" in run:
        return next(o for o in runs if o["id"] == run["reference"])
    episodes = {r["episode_id"] for r in _scored(run)}
    matches = [
        o
        for o in runs
        if o["role"] == "original"
        and o["split"] == run["split"]
        and o["variant"] == reference_variant(run["variant"])
        and {r["episode_id"] for r in _scored(run, o["records"])} == episodes
    ]
    if len(matches) > 1:
        raise ValueError(f"run {run['id']}: {len(matches)} Original runs cover its episodes: {[m['id'] for m in matches]}")
    return matches[0] if matches else None


def _agreements(runs: Sequence[Mapping[str, Any]], repeats: Sequence[Mapping[str, Any]]) -> dict[str, dict]:
    """Per split with a repeat, the per-format share of episodes on which the Original run and its repeat agree."""
    out = {}
    for repeat in repeats:
        original = next((o for o in runs if o["role"] == "original" and o["variant"] == "" and
                         o["split"] == repeat["split"]), None)
        if original is None:
            raise ValueError(f"the repeat {repeat['id']} has no Original run over {repeat['split']} beside it")
        require_comparable(original, repeat)
        out[repeat["split"]] = repeat_agreement(original["records"], repeat["records"])
    return out


def _agreement_columns(agreements: Mapping[str, Mapping[str, Any]], split: str) -> dict:
    found = agreements.get(split, {})
    return {f"repeat_agreement_{fmt}": found[fmt]["rate"] if fmt in found else None for fmt in FORMATS}


def _rows_of_rates(runs: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Every run's rates over all replies; a run whose appended read fell short also has its rates over the
    replies that state a final answer, marked as such in the `replies` column."""
    rows = []
    for run in runs:
        for replies, stated_only in REPLIES:
            if stated_only and not needs_stated_rows(run):
                continue
            for subset, wrong_first in SUBSETS:
                for row in rates(_scored(run), wrong_first_only=wrong_first, stated_only=stated_only):
                    out = {"runs": run["id"], "split": run["split"], "role": run["role"], "variant": run["variant"],
                           "replies": replies, "question_set": subset, "format": row["format"], "form": row["form"],
                           "situation": row["situation"] or "all", "n": row["n"]}
                    for metric in RATE_METRICS:
                        out[f"{metric}_numerator"] = row[metric]["numerator"]
                        out[f"{metric}_denominator"] = row[metric]["denominator"]
                        out[metric] = row[metric]["rate"]
                    rows.append(out)
    return rows


def _interval_columns(d: Mapping[str, Any]) -> dict:
    return {key: d.get(key) for key in ("estimate", "low", "high", "skipped", "episodes", "questions")}


def _rows_of_differences(runs: Sequence[Mapping[str, Any]], agreements: Mapping[str, Mapping[str, Any]]) -> list[dict]:
    rows = []
    for run in runs:
        if _is_reference_only(run):
            continue
        original = _original_for(run, runs)
        if original is None:
            continue
        require_comparable(run, original)
        scored, base = _scored(run), _scored(run, original["records"])
        for subset, wrong_first in SUBSETS:
            for fmt in FORMATS:
                for form in FORMS:
                    for situation in (*SITUATIONS, None):
                        select = readings.selection(fmt, (form,), None if situation is None else (situation,), wrong_first)
                        pairs = readings.paired_records(scored, base, select)
                        if not pairs:
                            continue
                        for rate_name in readings.RATE_NAMES:
                            d = readings.difference(pairs, rate_name)
                            if d["episodes"] == 0:
                                continue
                            rows.append({"runs": f"{run['id']};{original['id']}", "split": run["split"],
                                         "role": run["role"], "variant": run["variant"], "question_set": subset,
                                         "format": fmt, "form": form, "situation": situation or "all",
                                         "difference": f"{rate_name}, {_name(run)} minus {_name(original)}, points",
                                         **_interval_columns(d), **_agreement_columns(agreements, run["split"])})
    return rows


def _one(runs: Sequence[Mapping[str, Any]], role: str, variant: str = "") -> Mapping[str, Any] | None:
    found = [r for r in runs if r["role"] == role and r["split"] == "test" and r["variant"] == variant]
    if len(found) > 1:
        raise ValueError(f"{len(found)} {role} runs on test: {[r['id'] for r in found]}")
    return found[0] if found else None


def _reading_rows(name: str, runs: Sequence[Mapping[str, Any]], fmt: str, result: Mapping[str, Any],
                  agreement: Mapping[str, Any]) -> list[dict]:
    """One row per difference in a reading's result, each carrying the reading's string where it has one."""
    for run in runs[1:]:
        require_comparable(runs[0], run)
    rows = []

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping) and "estimate" in value:
            rows.append({"runs": ";".join(r["id"] for r in runs), "reading": name, "format": fmt, "quantity": prefix,
                         **_interval_columns(value), "outcome": result.get("reading"), "note": result.get("note"),
                         **agreement})
        elif isinstance(value, Mapping):
            for key, inner in value.items():
                walk(f"{prefix}.{key}" if prefix else key, inner)

    walk("", {k: v for k, v in result.items() if k not in ("format", "reading", "note")})
    return rows


def _pass_bar_rows(check2: Mapping[str, Any], original: Mapping[str, Any], agreement: Mapping[str, Any]) -> list[dict]:
    """Per format: the verdict on the whole of `test`, both quantities on each part, and the difference between the parts."""
    require_comparable(check2, original)
    rows = []
    ids = f"{check2['id']};{original['id']}"
    for fmt in FORMATS:
        bar = readings.pass_bar(check2["records"], original["records"], fmt)
        parts = readings.two_parts(check2["records"], original["records"], fmt)
        scopes = [("whole of test", bar), ("first part", parts["first"]), ("second part", parts["second"])]
        for scope, result in scopes:
            for quantity in ("wrong_adoption_drop", "right_adoption_drop"):
                rows.append({"runs": ids, "format": fmt, "scope": scope,
                             "quantity": f"{quantity}, original minus check2, points", **_interval_columns(result[quantity]),
                             "first_condition_holds": result.get("first_condition_holds"),
                             "second_condition": result.get("second_condition"), "verdict": result.get("verdict"),
                             **agreement})
        for quantity in ("wrong_adoption_drop", "right_adoption_drop"):
            between = parts[f"{quantity}_between_parts"]
            rows.append({"runs": ids, "format": fmt, "scope": "first part minus second part",
                         "quantity": f"{quantity}, original minus check2, points", **_interval_columns(between),
                         "verdict": between["reading"], **agreement})
    return rows


def _rows_of_readings(runs: Sequence[Mapping[str, Any]], agreements: Mapping[str, Mapping[str, Any]]
                      ) -> tuple[list[dict], list[dict], list[dict]]:
    """Rows of the pass bar, of the causal readings and of snap against short, on `test`."""
    agreement = _agreement_columns(agreements, "test")
    check2, original = _one(runs, "check2"), _one(runs, "original")
    off, swap, adapter = _one(runs, "off"), _one(runs, "swap"), _one(runs, "adapter only")
    second_stage, formed = _one(runs, "check2", FORMED_WITH_CHANNEL), _one(runs, "original", FORMED_MODEL)
    bar, causal, snap = [], [], []
    if check2 and original:
        bar = _pass_bar_rows(check2, original, agreement)
        result = readings.snap_against_short(check2["records"], original["records"])
        snap = _reading_rows("check2 snap minus original short, points", [check2, original], "snap and short", result,
                             agreement)
    for fmt in FORMATS:
        presence = None
        if off and check2:
            presence = readings.presence(off["records"], check2["records"], fmt)
            causal += _reading_rows("presence of the write", [off, check2], fmt, presence, agreement)
        if off and original and adapter:
            result = readings.adapter_without_signal(off["records"], original["records"], adapter["records"], fmt)
            causal += _reading_rows("the adapter without the signal", [off, original, adapter], fmt, result, agreement)
        if swap and check2 and off and presence is not None:
            result = readings.content(_scored(swap), _scored(swap, check2["records"]), fmt, presence_reading=presence)
            causal += _reading_rows("content of the signal", [swap, check2, off], fmt, result, agreement)
        if check2 and adapter:
            result = readings.channel_against_adapter(check2["records"], adapter["records"], fmt)
            causal += _reading_rows("channel against adapter", [check2, adapter], fmt, result, agreement)
        if second_stage and formed and check2 and original:
            result = excess(second_stage["records"], formed["records"], check2["records"], original["records"], fmt)
            causal += _reading_rows("excess of a sharper signal", [second_stage, formed, check2, original], fmt,
                                    result, agreement)
    return bar, causal, snap
