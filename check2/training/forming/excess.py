"""Whether a sharper signal helps: the second stage's gain over the formed model, against Check²'s over Original.

On reasoned `test` episodes of one format, for each of wrong and right adoption, the excess
is the second stage's paired difference from the formed model's run minus Check²'s paired
difference from the Original run, in points, the four runs on the same episodes. Its
interval resamples question ids, the same ids for the four runs in each resample, 2,000
times from one `random.Random(2033)`.

Lower wrong adoption and higher right adoption are favourable. A sharper signal helps only
if one rate's excess has its interval wholly on the favourable side and the other rate's
excess does not lie wholly on the other side; otherwise the stage could not tell a sharper
signal that does not help from a consumer that over-trusted it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from check2.evaluation import intervals
from check2.evaluation.readings.differences import (REASONED, Record, counts, paired_records, points,
                                                    require_interval, require_test, selection)

HELPS = "a sharper signal helps"
COULD_NOT_TELL = ("the stage could not tell a sharper signal that does not help from a consumer that "
                  "over-trusted it")
# The sign of a favourable excess in each rate.
FAVOURABLE = {"wrong_adoption": -1, "right_adoption": 1}


def _aligned(second_stage, formed, check2, original, select) -> list[tuple[Record, Record, Record, Record]]:
    """The four runs' selected records, matched by episode id, in the second stage's order."""
    first = paired_records(second_stage, formed, select)
    second = {a["episode_id"]: (a, b) for a, b in paired_records(check2, original, select)}
    paired_records([a for a, _ in first], [a for a, _ in second.values()], lambda r: True)
    return [(a, b, *second[a["episode_id"]]) for a, b in first]


def _by_question(rows, rate: str) -> tuple[list[dict], int]:
    """Per run, per question id, the summed (numerator, denominator) of the episodes the rate counts."""
    sums: list[dict[str, list[int]]] = [{}, {}, {}, {}]
    episodes = 0
    for records in rows:
        parts = [counts(r, rate) for r in records]
        if len({d for _, d in parts}) != 1:
            raise ValueError(f"episode {records[0]['episode_id']!r} counts in {rate} in some runs only")
        if parts[0][1] == 0:
            continue
        episodes += 1
        for run_sums, (n, d) in zip(sums, parts):
            cell = run_sums.setdefault(records[0]["qid"], [0, 0])
            cell[0] += n
            cell[1] += d
    return [{q: [tuple(v)] for q, v in s.items()} for s in sums], episodes


def _excess(by_qid: list[dict], drawn: Sequence[str]) -> float | None:
    values = [points([item for q in drawn for item in run[q]]) for run in by_qid]
    if any(v is None for v in values):
        return None
    second_stage, formed, check2, original = values
    return (second_stage - formed) - (check2 - original)


def rate_excess(rows, rate: str) -> dict:
    """One rate's excess in points, with its interval from resampling question ids."""
    by_qid, episodes = _by_question(rows, rate)
    ids = sorted(by_qid[0])
    values, skipped = [], 0
    for drawn in intervals.resamples(ids):
        value = _excess(by_qid, drawn)
        if value is None:
            skipped += 1
        else:
            values.append(value)
    result = intervals._summarise(_excess(by_qid, ids), values, skipped, intervals.DEFAULT_LEVEL)
    return {"estimate": result.estimate, "low": result.low, "high": result.high, "skipped": result.skipped,
            "episodes": episodes, "questions": len(ids)}


def _wholly(d: Mapping[str, Any], sign: int) -> bool:
    """The interval lies wholly on the side of zero that `sign` names."""
    return d["low"] > 0 if sign > 0 else d["high"] < 0


def reading(wrong: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    """Helps if one rate's excess is wholly favourable and the other's is not wholly unfavourable."""
    excesses = {"wrong_adoption": wrong, "right_adoption": right}
    for rate, other in (("wrong_adoption", "right_adoption"), ("right_adoption", "wrong_adoption")):
        if _wholly(excesses[rate], FAVOURABLE[rate]) and not _wholly(excesses[other], -FAVOURABLE[other]):
            return HELPS
    return COULD_NOT_TELL


def excess(second_stage: Iterable[Record], formed: Iterable[Record], check2: Iterable[Record],
           original: Iterable[Record], fmt: str) -> dict:
    """The excess in each adoption rate on reasoned `test` episodes of one format, and the reading."""
    rows = _aligned(list(second_stage), list(formed), list(check2), list(original), selection(fmt, forms=REASONED))
    require_test([(r[0], r[1]) for r in rows])
    out: dict[str, Any] = {"format": fmt}
    for rate in FAVOURABLE:
        out[f"{rate}_excess"] = rate_excess(rows, rate)
        require_interval(out[f"{rate}_excess"], f"the excess in {rate}")
    out["reading"] = reading(out["wrong_adoption_excess"], out["right_adoption_excess"])
    return out
