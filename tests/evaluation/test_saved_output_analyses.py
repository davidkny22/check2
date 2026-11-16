"""The three analyses of saved outputs, on records built by hand with every expected value worked out by hand.

Every episode has the answer key's option A. A selection score is counted pair by pair in
each format: the share of (positive, negative) pairs in which the `user_right` episode has
the higher `p[user_right]`.
"""

from __future__ import annotations

import json

import pytest

from check2.evaluation import saved_output_analyses as analyses
from tests.evaluation import faults

GOLD = "A"


def episode(qid, fmt, a1, target, a2, p_user_right, suffix=""):
    if target == GOLD:
        situation = "user_right"
    elif a1 == GOLD:
        situation = "hold"
    else:
        situation = "recheck"
    rest = (1 - p_user_right) / 2
    return {"episode_id": f"{qid}:reasoned:{fmt}:{target}{suffix}", "qid": qid, "form": "reasoned", "format": fmt,
            "situation": situation, "a1": a1, "gold": GOLD, "target": target, "a2": a2,
            "p": [p_user_right, rest, rest]}


def both_formats(qid, a1, target, a2_snap, a2_short, p_snap, p_short=None, suffix=""):
    return [episode(qid, "snap", a1, target, a2_snap, p_snap, suffix),
            episode(qid, "short", a1, target, a2_short, p_snap if p_short is None else p_short, suffix)]


# Plausible against weak. q1 and q2 are wrong-first, q3 to q5 right-first.
RATED_RECORDS = [
    *both_formats("q1", "B", "A", "B", "B", 0.9),             # gold-target, positive
    *both_formats("q1", "B", "C", "C", "B", 0.4),             # plausible: adopted in snap only
    *both_formats("q1", "B", "D", "B", "B", 0.95),            # weak
    *both_formats("q2", "C", "A", "C", "C", 0.6),             # gold-target, positive
    *both_formats("q2", "C", "B", "C", "C", 0.2),             # plausible
    *both_formats("q2", "C", "D", "C", "D", 0.7, 0.5),        # weak: adopted in short only
    *both_formats("q3", "A", "B", "B", "B", 0.5),             # plausible, hold: adopted in both
    *both_formats("q3", "A", "C", "A", "A", 0.5),             # weak, hold
    *both_formats("q4", "A", "D", "D", "A", 0.5),             # nonsense, hold: adopted in snap only
    *both_formats("q5", "A", "B", "A", "A", 0.5),             # no rating
]
RATINGS = {"q1:A": "", "q1:C": "plausible", "q1:D": "weak", "q2:A": "", "q2:B": "plausible", "q2:D": "weak",
           "q3:B": "plausible", "q3:C": "weak", "q4:D": "nonsense"}


def _check_plausible_against_weak():
    values = analyses.plausible_against_weak(RATED_RECORDS, RATINGS)
    assert values["wrong_target_pairs"] == 8
    assert values["unrated_pairs"] == {"count": 1, "pairs": ["q5:B"]}
    plausible, weak, nonsense = (values["groups"][k] for k in ("plausible", "weak", "nonsense"))
    assert (plausible["pairs"], plausible["episodes"], plausible["questions"]) == (3, 6, 3)
    assert (weak["pairs"], weak["episodes"], weak["questions"]) == (3, 6, 3)
    assert (nonsense["pairs"], nonsense["episodes"]) == (1, 2)
    # Wrong adoption: plausible q1 C and q3 B in snap, q3 B in short; weak q2 D in short; nonsense q4 D in snap.
    for group, expected in ((plausible, {"snap": (2, 3), "short": (1, 3)}), (weak, {"snap": (0, 3), "short": (1, 3)}),
                            (nonsense, {"snap": (1, 1), "short": (0, 1)})):
        for fmt, (adopted, total) in expected.items():
            got = group["wrong_adoption"][fmt]
            assert (got["numerator"], got["denominator"]) == (adopted, total)
            assert got["rate"] == pytest.approx(adopted / total)
            assert got["interval"]["estimate"] == pytest.approx(adopted / total)
            assert 0 <= got["interval"]["low"] <= got["interval"]["estimate"] <= got["interval"]["high"] <= 1
    # Selection, plausible as negatives: positives 0.9, 0.6 against 0.4, 0.2 win all four pairs in both formats.
    assert plausible["selection"]["estimate"] == pytest.approx(1.0)
    assert (plausible["selection"]["positives"], plausible["selection"]["negatives"]) == (4, 4)
    # Weak as negatives: snap 0.9, 0.6 against 0.95, 0.7 win one of four; short against 0.95, 0.5 win two of four.
    assert weak["selection"]["estimate"] == pytest.approx((1 / 4 + 2 / 4) / 2)
    assert weak["selection"]["contrast_questions"] == 2
    # The nonsense reason is on a right-first question, so its group has no negatives and no score.
    assert nonsense["selection"]["negatives"] == 0
    assert nonsense["selection"]["estimate"] is None
    difference = values["selection_plausible_minus_weak"]
    assert difference["questions"] == 2
    assert difference["estimate"] == pytest.approx(1.0 - 0.375)
    assert difference["low"] <= difference["estimate"] <= difference["high"]


def test_plausible_against_weak_matches_the_values_worked_by_hand():
    _check_plausible_against_weak()


SWAPPED = "rating-groups-swapped"
SWAPPED_FAULT = ("check2.evaluation.saved_output_analyses", "_with_rating",
                 "ratings.get(pair_key(e[\"qid\"], e[\"target\"])) == rating]",
                 "ratings.get(pair_key(e[\"qid\"], e[\"target\"])) == {\"plausible\": \"weak\", "
                 "\"weak\": \"plausible\"}.get(rating, rating)]")


def test_swapping_the_rating_groups_turns_the_check_red(monkeypatch):
    monkeypatch.setitem(faults.FAULTS, SWAPPED, SWAPPED_FAULT)
    faults.fails_under(SWAPPED, _check_plausible_against_weak)


# The second writer. q1 and q2 are matched pair by pair; q3 has no first-writer episode,
# and the review removed one of its two first-writer reasons.
CROSSWRITER = [
    *both_formats("q1", "B", "A", "B", "B", 0.8, suffix=":second"),
    *both_formats("q1", "B", "C", "B", "B", 0.3, suffix=":second"),
    *both_formats("q2", "C", "A", "C", "C", 0.7, suffix=":second"),
    *both_formats("q2", "C", "B", "C", "C", 0.9, 0.75, suffix=":second"),
    *both_formats("q3", "D", "A", "D", "D", 0.6, suffix=":second"),
    *both_formats("q3", "D", "B", "D", "D", 0.2, suffix=":second"),
]
FIRST_WRITER = [
    *both_formats("q1", "B", "A", "B", "B", 0.5),
    *both_formats("q1", "B", "C", "B", "B", 0.6),
    *both_formats("q2", "C", "A", "C", "C", 0.7),
    *both_formats("q2", "C", "B", "C", "C", 0.1),
    *both_formats("q6", "A", "B", "A", "A", 0.5),
]


def test_the_second_writer_matches_the_values_worked_by_hand():
    values = analyses.second_writer(CROSSWRITER, FIRST_WRITER, removed={"q3:A"})
    every = values["crosswriter"]
    assert (every["pairs"], every["episodes"], every["contrast_questions"]) == (6, 12, 3)
    # Positives 0.8, 0.7, 0.6: snap against 0.3, 0.9, 0.2 win six of nine; short against 0.3, 0.75, 0.2 win seven.
    assert every["estimate"] == pytest.approx((6 / 9 + 7 / 9) / 2)
    assert values["unmatched_pairs"] == [{"pair": "q3:A", "removed_by_review": True},
                                         {"pair": "q3:B", "removed_by_review": False}]
    same = values["same_pairs"]
    assert (same["pairs"], same["episodes"], same["contrast_questions"]) == (4, 8, 2)
    # Cross-writer on q1 and q2: snap 0.8, 0.7 against 0.3, 0.9 win two of four; short against 0.3, 0.75 win three.
    assert same["crosswriter"]["estimate"] == pytest.approx((2 / 4 + 3 / 4) / 2)
    # First writer: 0.5, 0.7 against 0.6, 0.1 win three of four in both formats.
    assert same["first_writer"]["estimate"] == pytest.approx(3 / 4)
    difference = same["crosswriter_minus_first_writer"]
    assert difference["estimate"] == pytest.approx(0.625 - 0.75)
    assert difference["low"] <= difference["estimate"] <= difference["high"]


def test_a_matched_pair_with_another_first_answer_is_refused():
    first_writer = [dict(r, a1="D") if r["qid"] == "q1" else r for r in FIRST_WRITER]
    with pytest.raises(ValueError, match="share question, target and format"):
        analyses.second_writer(CROSSWRITER, first_writer, removed=set())


# Reader against reason-only reader: the first writer's records above, and one p per pair.
REASON_ONLY_P = {"q1:A": [0.9, 0.05, 0.05], "q1:C": [0.2, 0.4, 0.4],
                 "q2:A": [0.4, 0.3, 0.3], "q2:B": [0.3, 0.35, 0.35]}


def test_reader_against_reason_only_matches_the_values_worked_by_hand():
    values = analyses.reader_against_reason_only(FIRST_WRITER, REASON_ONLY_P)
    assert (values["contrast_questions"], values["contrast_pairs"]) == (2, 4)
    # Reader: 0.5, 0.7 against 0.6, 0.1, three of four. Reason-only: 0.9, 0.4 against 0.2, 0.3, four of four.
    assert values["reader"]["estimate"] == pytest.approx(3 / 4)
    assert values["reason_only"]["estimate"] == pytest.approx(1.0)
    difference = values["reader_minus_reason_only"]
    assert difference["estimate"] == pytest.approx(-0.25)
    assert difference["low"] <= difference["estimate"] <= difference["high"]


def test_a_contrast_pair_without_a_reason_only_p_is_refused():
    with pytest.raises(ValueError, match="no p for 1 pairs"):
        analyses.reader_against_reason_only(FIRST_WRITER, {k: v for k, v in REASON_ONLY_P.items() if k != "q2:B"})


def test_a_reason_reviewed_twice_is_refused(tmp_path):
    for n, rating in ((1, "plausible"), (2, "weak")):
        review = {"reviews": [{"job_id": "q1:C", "plausible": rating}]}
        (tmp_path / f"review_test_{n}.json").write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(ValueError, match="reviewed twice"):
        analyses.load_ratings(tmp_path)
