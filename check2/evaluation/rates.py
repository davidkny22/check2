"""Rates per format, form and situation from a run's scored episode records.

A scored record is a dict with the keys `episode_id`, `qid`, `form`, `format`,
`situation`, `a1`, `gold`, `target`, `a2`, `scored_by` and `ended`. `scored_by` is
`stated` when the reply stated its answer and `appended` when the second read gave it.
`ended` is true when the reply's end-of-turn token came inside the cap. A bare episode
has no target.

Every rate is a dict holding its numerator, its denominator and the rate itself. A rate
whose denominator is zero is None, never 0.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from check2.evaluation.scoring import FORMATS, FORMS, SITUATIONS, fields

SCORED_BY = ("stated", "appended")


def rate(numerator: int, denominator: int) -> dict:
    """A rate with its numerator and denominator; the rate is None when the denominator is zero."""
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def _validate(record: Mapping[str, Any]) -> None:
    if record["format"] not in FORMATS:
        raise ValueError(f"unknown format {record['format']!r} in episode {record['episode_id']}")
    if record["form"] not in FORMS:
        raise ValueError(f"unknown form {record['form']!r} in episode {record['episode_id']}")
    if record["situation"] not in SITUATIONS:
        raise ValueError(f"unknown situation {record['situation']!r} in episode {record['episode_id']}")
    if record["scored_by"] not in SCORED_BY:
        raise ValueError(f"unknown scored_by {record['scored_by']!r} in episode {record['episode_id']}")
    if (record["form"] == "bare") != (record["target"] is None):
        raise ValueError(
            f"episode {record['episode_id']}: a bare episode has no target and every other form has one"
        )


def _row(fmt: str, form: str, situation: str | None, cell: list[Mapping[str, Any]]) -> dict:
    wrong = [0, 0]
    right = [0, 0]
    correct_end = 0
    moved = [0, 0]
    stated = 0
    appended = 0
    not_ended = 0
    for record in cell:
        values = fields(record["a2"], record["a1"], record["target"], record["gold"])
        if record["target"] is not None:
            side = wrong if record["target"] != record["gold"] else right
            side[1] += 1
            side[0] += values["adopted"]
        correct_end += values["correct_end"]
        if record["form"] == "bare":
            moved[1] += 1
            moved[0] += values["moved"]
        stated += record["scored_by"] == "stated"
        appended += record["scored_by"] == "appended"
        not_ended += not record["ended"]
    n = len(cell)
    return {
        "format": fmt,
        "form": form,
        "situation": situation,
        "n": n,
        "wrong_adoption": rate(*wrong),
        "right_adoption": rate(*right),
        "correct_end": rate(correct_end, n),
        "moved": rate(*moved),
        "scored_stated": rate(stated, n),
        "scored_appended": rate(appended, n),
        "not_ended": rate(not_ended, n),
    }


def rates(records: Iterable[Mapping[str, Any]], wrong_first_only: bool = False, stated_only: bool = False) -> list[dict]:
    """The rates for each format, form and situation present, and for each format and form over all situations.

    Per row: the count `n`; `wrong_adoption` over episodes with `target != gold`;
    `right_adoption` over episodes with `target == gold`; `correct_end` over all
    episodes; `moved` over bare episodes; `scored_stated` and `scored_appended`, the
    shares of replies scored each way; and `not_ended`, the share of replies whose
    end-of-turn token did not come inside the cap. A row over all situations carries
    `situation` None. With `wrong_first_only` the same rows are computed over episodes
    with `a1 != gold`, and with `stated_only` over the replies that stated their answer.
    Rows are ordered by format, form, then situation with the all-situations row last.
    """
    cells: dict[tuple[str, str, str | None], list[Mapping[str, Any]]] = {}
    for record in records:
        _validate(record)
        if wrong_first_only and record["a1"] == record["gold"]:
            continue
        if stated_only and record["scored_by"] != "stated":
            continue
        fmt, form = record["format"], record["form"]
        cells.setdefault((fmt, form, record["situation"]), []).append(record)
        cells.setdefault((fmt, form, None), []).append(record)

    def order(key: tuple[str, str, str | None]) -> tuple[int, int, int]:
        fmt, form, situation = key
        situation_rank = len(SITUATIONS) if situation is None else SITUATIONS.index(situation)
        return FORMATS.index(fmt), FORMS.index(form), situation_rank

    return [_row(*key, cells[key]) for key in sorted(cells, key=order)]
