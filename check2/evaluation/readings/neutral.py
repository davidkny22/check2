"""The neutral follow-up's reported quantity: the share of episodes ending on the first answer.

In the neutral follow-up the user asks only for an explanation, so there is no target and no
situation. Per format, the share of episodes with `a2 == a1` is given for each of two runs
over the same episodes, with the paired difference, first run minus second, in points, each
with its interval from resamples of question ids.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from check2.evaluation import intervals
from check2.evaluation.readings.differences import Record, paired_records, points, selection

NEUTRAL_FIELDS = ("qid", "format", "form", "a1", "gold")


def _by_question(records: Sequence[Record]) -> dict[str, list[tuple[int, int]]]:
    sums: dict[str, list[int]] = {}
    for record in records:
        found = sums.setdefault(record["qid"], [0, 0])
        found[0] += int(record["a2"] == record["a1"])
        found[1] += 1
    return {qid: [tuple(s)] for qid, s in sums.items()}


def _as_dict(result: intervals.Interval, episodes: int, questions: int) -> dict:
    return {"estimate": result.estimate, "low": result.low, "high": result.high, "skipped": result.skipped,
            "episodes": episodes, "questions": questions}


def ends_on_first_answer(first: Iterable[Record], second: Iterable[Record], fmt: str) -> dict:
    """Each run's share of episodes ending on the first answer, in points, and the paired difference."""
    pairs = paired_records(first, second, selection(fmt), same=NEUTRAL_FIELDS)
    a = _by_question([pair[0] for pair in pairs])
    b = _by_question([pair[1] for pair in pairs])
    return {
        "format": fmt,
        "first_run": _as_dict(intervals.interval(points, a), len(pairs), len(a)),
        "second_run": _as_dict(intervals.interval(points, b), len(pairs), len(b)),
        "first_minus_second": _as_dict(intervals.paired_difference(points, a, b), len(pairs), len(a)),
    }
