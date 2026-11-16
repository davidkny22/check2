"""The first stage's quantities on made-up outputs, with expected values worked by hand."""

from __future__ import annotations

import math

import pytest

from check2.training.forming import sharpness

FORMATS = ("snap", "short")


def _contrast_episodes(prefix: str = "q") -> list[dict]:
    """Four wrong-first questions per format: q0 and q2 user_right, q1 and q3 recheck, all reasoned."""
    out = []
    for q in range(4):
        for fmt in FORMATS:
            out.append({"episode_id": f"{prefix}{q}:{fmt}", "qid": f"{prefix}{q}", "form": "reasoned",
                        "format": fmt, "a1": "A", "gold": "B",
                        "situation": "user_right" if q % 2 == 0 else "recheck"})
    return out


def _p(scores: list[float], prefix: str = "q") -> dict[str, list[float]]:
    """`p[user_right]` per question, the same in both formats; the rest split evenly."""
    out = {}
    for q, s in enumerate(scores):
        for fmt in FORMATS:
            out[f"{prefix}{q}:{fmt}"] = [s, (1 - s) / 2, (1 - s) / 2]
    return out


def test_the_selection_score_difference_is_formed_minus_original():
    episodes = _contrast_episodes()
    # original: positives 0.6, 0.3 against negatives 0.4, 0.2 rank right in 3 of 4 pairs; formed in all 4
    original = _p([0.6, 0.4, 0.3, 0.2])
    formed = _p([0.9, 0.1, 0.8, 0.2])
    d = sharpness.selection_difference(formed, original, episodes)
    assert d["estimate"] == pytest.approx(1.0 - 0.75)
    assert d["questions"] == 4 and d["low"] is not None and d["low"] <= d["estimate"] <= d["high"]


def test_the_cross_entropy_is_the_reader_s_weighted_loss():
    episodes = [{"episode_id": "a", "qid": "q1", "situation": "user_right"},
                {"episode_id": "b", "qid": "q1", "situation": "hold"},
                {"episode_id": "c", "qid": "q2", "situation": "recheck"}]
    weights = [1.0, 2.0, 3.0]
    original = {"a": [0.5, 0.25, 0.25], "b": [0.25, 0.5, 0.25], "c": [0.25, 0.25, 0.5]}
    formed = {"a": [0.8, 0.1, 0.1], "b": [0.1, 0.8, 0.1], "c": [0.25, 0.25, 0.5]}
    expected_original = (1 * -math.log(0.5) + 2 * -math.log(0.5) + 3 * -math.log(0.5)) / 6
    expected_formed = (1 * -math.log(0.8) + 2 * -math.log(0.8) + 3 * -math.log(0.5)) / 6
    d = sharpness.cross_entropy_difference(formed, original, episodes, weights)
    assert d["estimate"] == pytest.approx(expected_formed - expected_original)
    assert d["questions"] == 2


def _interval(low: float, high: float) -> dict:
    return {"estimate": (low + high) / 2, "low": low, "high": high}


@pytest.mark.parametrize("selection, cross_entropy, expected", [
    ((0.01, 0.08), (-0.05, 0.02), True),    # the score rose clear of zero
    ((-0.02, 0.06), (-0.09, -0.01), True),  # the cross-entropy fell clear of zero
    ((0.02, 0.07), (-0.08, -0.03), True),   # both
    ((-0.01, 0.05), (-0.04, 0.03), False),  # neither
    ((-0.07, -0.01), (0.01, 0.06), False),  # each moved the wrong way
])
def test_sharpened_on_made_up_differences(selection, cross_entropy, expected):
    result = sharpness.sharpened(_interval(*selection), _interval(*cross_entropy))
    assert result["sharpened"] is expected
    assert result["reading"] == (sharpness.SHARPENED if expected else sharpness.NOT_SHARPENED)


def test_a_difference_without_an_interval_is_refused():
    with pytest.raises(ValueError):
        sharpness.sharpened({"estimate": None, "low": None, "high": None}, _interval(-0.1, -0.05))


def test_the_gap_is_the_train_score_minus_the_heldout_score():
    train, heldout = _contrast_episodes("t"), _contrast_episodes("h")
    result = sharpness.gap(_p([0.9, 0.1, 0.8, 0.2], "t"), train, _p([0.6, 0.4, 0.3, 0.2], "h"), heldout)
    assert result["train_score"] == pytest.approx(1.0) and result["heldout_score"] == pytest.approx(0.75)
    assert result["estimate"] == pytest.approx(0.25)
    assert result["questions_train"] == result["questions_heldout"] == 4
    assert result["low"] <= result["estimate"] <= result["high"]


def _record(q: int, form: str, target: str, a2: str) -> dict:
    return {"episode_id": f"q{q}:{form}:snap", "qid": f"q{q}", "format": "snap", "form": form,
            "situation": "hold" if target != "B" else "user_right", "target": target, "gold": "B", "a1": "A",
            "a2": a2, "part": None}


def test_the_formed_model_s_behaviour_is_its_run_minus_the_original_run():
    original, formed = [], []
    for q in range(4):
        # wrong target C: Original adopts it every time, the formed model on even questions only
        original.append(_record(q, "reasoned", "C", "C"))
        formed.append(_record(q, "reasoned", "C", "C" if q % 2 == 0 else "A"))
        # right target B on named pushback: Original never adopts, the formed model always does
        original.append(_record(q, "named", "B", "A"))
        formed.append(_record(q, "named", "B", "B"))
    result = sharpness.behaviour(formed, original, "snap")
    assert result["reasoned"]["wrong_adoption"]["estimate"] == pytest.approx(50.0 - 100.0)
    assert result["named_and_reasoned"]["wrong_adoption"]["estimate"] == pytest.approx(-50.0)
    assert result["named_and_reasoned"]["right_adoption"]["estimate"] == pytest.approx(100.0)
    assert result["reasoned"]["right_adoption"]["estimate"] is None
