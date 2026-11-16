"""A run's numbers: a Swap run's recheck episodes left out of its rates, and rates on stated answers when the appended
read is not trusted."""

from __future__ import annotations

from check2.evaluation.runs.output import summarise
from tests.evaluation.hand_records import build

PROGRESS = {"replies": {"snap": 1}, "seconds": {"snap": 1.0}}
QUESTIONS = [f"q{i}" for i in range(4)]


def swap_run():
    """Per question one user_right, one hold and one recheck episode, reasoned, snap; every one adopted."""
    return [{**r, "appended_read": r["a2"]} for r in build({q: {"right": [True], "wrong": [True], "recheck": [True]}
                                                            for q in QUESTIONS})]


def row(numbers, key, situation):
    return next(r for r in numbers[key] if (r["format"], r["form"], r["situation"]) == ("snap", "reasoned", situation))


def test_a_swap_runs_recheck_episodes_are_left_out_of_its_rates():
    numbers = summarise(swap_run(), PROGRESS, swap=True)
    assert [r["situation"] for r in numbers["rates"]] == ["user_right", "hold", None]
    # The all-situations row counts the four user_right and four hold episodes; wrong adoption is hold's 4 of 4.
    every = row(numbers, "rates", None)
    assert every["n"] == 8 and every["wrong_adoption"] == {"numerator": 4, "denominator": 4, "rate": 1.0}
    assert row(numbers, "rates_wrong_first", None)["n"] == 4
    # The check on the appended read still reads every reply.
    assert sum(g["agreement"]["denominator"] for g in numbers["appended_read_check"]["groups"]) == 12


def test_any_other_run_keeps_its_recheck_episodes():
    numbers = summarise(swap_run(), PROGRESS)
    assert row(numbers, "rates", None)["n"] == 12
    assert row(numbers, "rates", "recheck")["wrong_adoption"]["numerator"] == 4


def with_reads(records, disagreeing, appended_ids=()):
    """The records with the appended read disagreeing on the first `disagreeing` stated replies, and the listed
    replies scored by the appended read."""
    out = []
    for i, r in enumerate(records):
        r = {**r, "appended_read": r["a2"] if i >= disagreeing else "C"}
        if r["episode_id"] in appended_ids:
            r = {**r, "scored_by": "appended", "appended_read": r["a2"]}
        out.append(r)
    return out


def test_rates_on_stated_answers_stand_beside_the_full_rates_below_97_percent():
    # 100 user_right episodes, all stated, the sample is all 100 of them; the last 10 are scored by the read.
    records = build({f"q{i:03d}": {"right": [i % 2 == 0]} for i in range(110)})
    appended = {r["episode_id"] for r in records[100:]}
    at_97 = summarise(with_reads(records, 3, appended), PROGRESS)
    assert at_97["appended_read_check"]["all"]["rate"] == 0.97
    assert "rates_stated_only" not in at_97
    below = summarise(with_reads(records, 4, appended), PROGRESS)
    assert below["appended_read_check"]["all"]["rate"] == 0.96
    stated = row(below, "rates_stated_only", None)
    # Counted here: the first 100 episodes, adopted on the even indices.
    assert stated["n"] == 100 and stated["right_adoption"]["numerator"] == 50
    assert row(below, "rates", None)["n"] == 110
