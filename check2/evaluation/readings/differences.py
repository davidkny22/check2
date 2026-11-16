"""Paired differences between the scored records of two runs, in points, with intervals by question id.

A run is given as its scored records: the records of `check2.evaluation.rates`, each with
`p`, the reader's three probabilities, and `part`, `first` or `second`, on a `test`
episode. Both runs must hold the same episodes for a comparison, and the same question
ids are drawn for both in each resample. An episode present in one run and missing from
the other is an error.

A difference is in points: a rate times 100. It is returned as a dict with its
`estimate`, `low`, `high` (the 95% interval), `skipped` (resamples on which a rate was
undefined), and the counts of `episodes` and `questions` it was computed on. The
resampled question ids are those of the episodes the rate is taken over.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from check2.evaluation import intervals
from check2.evaluation.scoring import fields

Record = Mapping[str, Any]
Selection = Callable[[Record], bool]

RATE_NAMES = ("wrong_adoption", "right_adoption", "correct_end", "moved")
PARTS = ("first", "second")

# The fields that describe an episode and not a reply; two runs must agree on them.
EPISODE_FIELDS = ("qid", "format", "form", "situation", "target", "gold", "a1", "part")
# A pair's episodes in the two formats agree on these.
PAIR_FIELDS = ("qid", "form", "situation", "target", "gold", "part")

NAMED_AND_REASONED = ("named", "reasoned")
REASONED = ("reasoned",)


def selection(
    fmt: str | None = None,
    forms: Sequence[str] | None = None,
    situations: Sequence[str] | None = None,
    wrong_first: bool = False,
    part: str | None = None,
) -> Selection:
    """A predicate on records by format, forms, situations, first answer and part of `test`."""

    def keep(record: Record) -> bool:
        if fmt is not None and record["format"] != fmt:
            return False
        if forms is not None and record["form"] not in forms:
            return False
        if situations is not None and record["situation"] not in situations:
            return False
        if wrong_first and record["a1"] == record["gold"]:
            return False
        if part is not None and record.get("part") != part:
            return False
        return True

    return keep


def _index(records: Iterable[Record], select: Selection, key: Callable[[Record], Any], name: str) -> dict:
    indexed: dict = {}
    for record in records:
        if not select(record):
            continue
        k = key(record)
        if k in indexed:
            raise ValueError(f"{name}: two records share the key {k!r}")
        indexed[k] = record
    return indexed


def paired_records(
    run_a: Iterable[Record],
    run_b: Iterable[Record],
    select: Selection,
    key: Callable[[Record], Any] = lambda r: r["episode_id"],
    same: Sequence[str] = EPISODE_FIELDS,
    select_b: Selection | None = None,
) -> list[tuple[Record, Record]]:
    """The selected records of two runs, matched by `key`, in the first run's order.

    `select` picks the records of the first run, and of the second unless `select_b` is
    given. Both runs must hold exactly the same keys after selection, and each matched
    pair must agree on the fields named in `same`. Anything else is an error.
    """
    a = _index(run_a, select, key, "first run")
    b = _index(run_b, select_b or select, key, "second run")
    only_a = [k for k in a if k not in b]
    only_b = [k for k in b if k not in a]
    if only_a or only_b:
        raise ValueError(
            "a paired comparison needs the same episodes in both runs; "
            f"{len(only_a)} only in the first (e.g. {only_a[:3]}), "
            f"{len(only_b)} only in the second (e.g. {only_b[:3]})"
        )
    for k, record_a in a.items():
        record_b = b[k]
        for name in same:
            if record_a.get(name) != record_b.get(name):
                raise ValueError(
                    f"episode {k!r}: the runs disagree on {name} ({record_a.get(name)!r} against {record_b.get(name)!r})"
                )
    return [(a[k], b[k]) for k in a]


def counts(record: Record, rate_name: str) -> tuple[int, int]:
    """The record's contribution (numerator, denominator) to a rate.

    Wrong adoption counts episodes with a target that is not the answer key's option,
    right adoption those whose target is the answer key's option, `correct_end` every
    episode, and `moved` bare episodes only, as the rate tables count them.
    """
    values = fields(record["a2"], record["a1"], record["target"], record["gold"])
    if rate_name == "wrong_adoption":
        if record["target"] is None or record["target"] == record["gold"]:
            return 0, 0
        return int(values["adopted"]), 1
    if rate_name == "right_adoption":
        if record["target"] is None or record["target"] != record["gold"]:
            return 0, 0
        return int(values["adopted"]), 1
    if rate_name == "correct_end":
        return int(values["correct_end"]), 1
    if rate_name == "moved":
        if record["form"] != "bare":
            return 0, 0
        return int(values["moved"]), 1
    raise ValueError(f"unknown rate {rate_name!r}; expected one of {RATE_NAMES}")


def points(items: Sequence[tuple[int, int]]) -> float | None:
    numerator = sum(item[0] for item in items)
    denominator = sum(item[1] for item in items)
    return 100 * numerator / denominator if denominator else None


def by_question(pairs: Sequence[tuple[Record, Record]], rate_name: str) -> tuple[dict, dict, int]:
    """Per question id, the summed (numerator, denominator) of each run, over the episodes the rate counts."""
    a: dict[str, list[int]] = {}
    b: dict[str, list[int]] = {}
    episodes = 0
    for record_a, record_b in pairs:
        numerator_a, denominator_a = counts(record_a, rate_name)
        numerator_b, denominator_b = counts(record_b, rate_name)
        if denominator_a != denominator_b:
            raise ValueError(f"episode {record_a['episode_id']!r} counts in {rate_name} in one run only")
        if denominator_a == 0:
            continue
        episodes += 1
        sums_a = a.setdefault(record_a["qid"], [0, 0])
        sums_b = b.setdefault(record_a["qid"], [0, 0])
        sums_a[0] += numerator_a
        sums_a[1] += denominator_a
        sums_b[0] += numerator_b
        sums_b[1] += denominator_b
    return (
        {qid: [tuple(sums)] for qid, sums in a.items()},
        {qid: [tuple(sums)] for qid, sums in b.items()},
        episodes,
    )


def _as_dict(result: intervals.Interval, episodes: int, questions: int) -> dict:
    return {
        "estimate": result.estimate,
        "low": result.low,
        "high": result.high,
        "skipped": result.skipped,
        "episodes": episodes,
        "questions": questions,
    }


def difference(pairs: Sequence[tuple[Record, Record]], rate_name: str) -> dict:
    """The paired difference in a rate, first run minus second, in points, with its interval.

    The interval is `check2.evaluation.intervals.paired_difference` over the question ids
    of the episodes the rate counts.
    """
    a, b, episodes = by_question(pairs, rate_name)
    result = intervals.paired_difference(points, a, b)
    return _as_dict(result, episodes, len(a))


def require_interval(d: Mapping[str, Any], what: str) -> None:
    if d["estimate"] is None or d["low"] is None or d["high"] is None:
        raise ValueError(f"{what} is undefined: no episode counts in its rate")


def wholly_above_zero(d: Mapping[str, Any]) -> bool:
    """The interval's lower end is above zero."""
    return d["low"] > 0


def wholly_below_zero(d: Mapping[str, Any]) -> bool:
    """The interval's upper end is below zero."""
    return d["high"] < 0


def includes_zero(d: Mapping[str, Any]) -> bool:
    """The interval's lower end is at or below zero and its upper end at or above it."""
    return d["low"] <= 0 <= d["high"]


def require_test(pairs: Sequence[tuple[Record, Record]]) -> None:
    for record_a, _ in pairs:
        if record_a.get("part") not in PARTS:
            raise ValueError(f"episode {record_a['episode_id']!r} carries no part of test; this reading is on test")
