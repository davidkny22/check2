"""Bootstrap intervals that resample question ids.

Episodes of one question are not independent, so a resample draws question ids with
replacement and brings every record of each drawn id, as many times as the id was drawn.

A statistic maps a list of records to a number, or to None where it is undefined on
that list (a rate with a zero denominator). A resample on which the statistic is
undefined is skipped, and the count of skipped resamples is carried by the result.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

Statistic = Callable[[list[Any]], float | None]
RecordsByQid = Mapping[str, Sequence[Any]]

DEFAULT_RESAMPLES = 2000
DEFAULT_SEED = 2033
DEFAULT_LEVEL = 0.95


class Interval(tuple):
    """An (estimate, low, high) triple that also carries `skipped`, the number of resamples skipped.

    The triple unpacks as three values. The estimate is the statistic on all records;
    low and high are the percentile bounds over the resamples on which the statistic
    was defined, or None when there were none.
    """

    skipped: int

    def __new__(cls, estimate: float | None, low: float | None, high: float | None, skipped: int):
        instance = super().__new__(cls, (estimate, low, high))
        instance.skipped = skipped
        return instance

    @property
    def estimate(self) -> float | None:
        return self[0]

    @property
    def low(self) -> float | None:
        return self[1]

    @property
    def high(self) -> float | None:
        return self[2]


def resamples(question_ids: Sequence[str], n: int = DEFAULT_RESAMPLES, seed: int = DEFAULT_SEED) -> Iterator[list[str]]:
    """Yield `n` resamples of the question ids, each drawn with replacement and as long as the list.

    One generator seeded with `seed` serves all `n` draws, from the sorted ids.
    """
    rng = random.Random(seed)
    ids = sorted(question_ids)
    for _ in range(n):
        yield rng.choices(ids, k=len(ids))


def _gather(records_by_qid: RecordsByQid, drawn: Sequence[str]) -> list[Any]:
    records: list[Any] = []
    for qid in drawn:
        records.extend(records_by_qid[qid])
    return records


def _all_records(records_by_qid: RecordsByQid) -> list[Any]:
    return _gather(records_by_qid, sorted(records_by_qid))


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """The q-th quantile, 0 to 1, by linear interpolation between order statistics."""
    position = q * (len(sorted_values) - 1)
    below = math.floor(position)
    above = math.ceil(position)
    fraction = position - below
    return sorted_values[below] + (sorted_values[above] - sorted_values[below]) * fraction


def _summarise(estimate: float | None, values: list[float], skipped: int, level: float) -> Interval:
    if not values:
        return Interval(estimate, None, None, skipped)
    if not 0 < level < 1:
        raise ValueError(f"level must lie strictly between 0 and 1, got {level}")
    values = sorted(values)
    tail = (1 - level) / 2
    return Interval(estimate, _percentile(values, tail), _percentile(values, 1 - tail), skipped)


def _single_values(statistic: Statistic, records_by_qid: RecordsByQid, n: int, seed: int) -> tuple[list[float], int]:
    values: list[float] = []
    skipped = 0
    for drawn in resamples(list(records_by_qid), n, seed):
        value = statistic(_gather(records_by_qid, drawn))
        if value is None:
            skipped += 1
        else:
            values.append(value)
    return values, skipped


def _paired_values(
    statistic: Statistic,
    records_a_by_qid: RecordsByQid,
    records_b_by_qid: RecordsByQid,
    n: int,
    seed: int,
) -> tuple[list[float], int]:
    values: list[float] = []
    skipped = 0
    for drawn in resamples(list(records_a_by_qid), n, seed):
        value_a = statistic(_gather(records_a_by_qid, drawn))
        value_b = statistic(_gather(records_b_by_qid, drawn))
        if value_a is None or value_b is None:
            skipped += 1
        else:
            values.append(value_a - value_b)
    return values, skipped


def _require_same_questions(records_a_by_qid: RecordsByQid, records_b_by_qid: RecordsByQid) -> None:
    only_a = set(records_a_by_qid) - set(records_b_by_qid)
    only_b = set(records_b_by_qid) - set(records_a_by_qid)
    if only_a or only_b:
        raise ValueError(
            "a paired difference needs the same question ids in both runs; "
            f"{len(only_a)} only in the first, {len(only_b)} only in the second"
        )


def interval(
    statistic: Statistic,
    records_by_qid: RecordsByQid,
    n: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
) -> Interval:
    """The percentile interval of `statistic` over resamples of question ids.

    Returns an `Interval`: (estimate, low, high), with the count of skipped resamples
    in its `skipped` attribute. The estimate is the statistic on all records.
    """
    estimate = statistic(_all_records(records_by_qid))
    values, skipped = _single_values(statistic, records_by_qid, n, seed)
    return _summarise(estimate, values, skipped, level)


def paired_difference(
    statistic: Statistic,
    records_a_by_qid: RecordsByQid,
    records_b_by_qid: RecordsByQid,
    n: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
) -> Interval:
    """The percentile interval of `statistic(a) - statistic(b)`, the same question ids drawn for both runs.

    Both runs must hold the same question ids. A resample on which the statistic is
    undefined for either run is skipped and counted in the result's `skipped`.
    """
    _require_same_questions(records_a_by_qid, records_b_by_qid)
    estimate_a = statistic(_all_records(records_a_by_qid))
    estimate_b = statistic(_all_records(records_b_by_qid))
    estimate = None if estimate_a is None or estimate_b is None else estimate_a - estimate_b
    values, skipped = _paired_values(statistic, records_a_by_qid, records_b_by_qid, n, seed)
    return _summarise(estimate, values, skipped, level)


def standard_error(
    statistic: Statistic,
    records_by_qid: RecordsByQid,
    n: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> float:
    """The standard deviation of `statistic` over the resamples on which it is defined.

    The divisor is the number of defined resamples less one. Fewer than two defined
    resamples leave it undefined and raise a ValueError.
    """
    values, _ = _single_values(statistic, records_by_qid, n, seed)
    if len(values) < 2:
        raise ValueError(f"the standard error needs at least two defined resamples, got {len(values)}")
    mean = math.fsum(values) / len(values)
    return math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (len(values) - 1))
