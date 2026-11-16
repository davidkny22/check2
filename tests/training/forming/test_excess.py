"""The excess of the second stage over Check², and its reading, on made-up runs."""

from __future__ import annotations

import pytest

from check2.training.forming.excess import COULD_NOT_TELL, HELPS, excess
from tests.evaluation.hand_records import ALL, NONE, reasoned_run

TWENTY = [20] * 10
TEN = [10] * 10


def _runs(original, check2, formed, second_stage):
    """Each argument is `(wrong, right)` per question, as `reasoned_run` takes them."""
    return [reasoned_run(*x) for x in (second_stage, formed, check2, original)]


def test_a_clear_excess_on_one_rate_and_none_against_on_the_other_helps():
    second_stage, formed, check2, original = _runs(original=(ALL, TWENTY), check2=(ALL, TWENTY),
                                                   formed=(ALL, TWENTY), second_stage=(NONE, TWENTY))
    result = excess(second_stage, formed, check2, original, "snap")
    assert result["wrong_adoption_excess"]["estimate"] == pytest.approx(-100.0)
    assert result["right_adoption_excess"]["estimate"] == pytest.approx(0.0)
    assert result["reading"] == HELPS


def test_a_gain_over_the_formed_model_equal_to_check2_s_gain_over_original_could_not_tell():
    second_stage, formed, check2, original = _runs(original=(ALL, TWENTY), check2=(NONE, TWENTY),
                                                   formed=(ALL, TWENTY), second_stage=(NONE, TWENTY))
    result = excess(second_stage, formed, check2, original, "snap")
    assert result["wrong_adoption_excess"]["estimate"] == pytest.approx(0.0)
    assert result["reading"] == COULD_NOT_TELL


def test_a_clear_excess_on_one_rate_with_a_clear_loss_on_the_other_could_not_tell():
    second_stage, formed, check2, original = _runs(original=(ALL, TWENTY), check2=(ALL, TWENTY),
                                                   formed=(ALL, TWENTY), second_stage=(NONE, TEN))
    result = excess(second_stage, formed, check2, original, "snap")
    assert result["wrong_adoption_excess"]["high"] < 0
    assert result["right_adoption_excess"]["estimate"] == pytest.approx(-50.0)
    assert result["right_adoption_excess"]["high"] < 0
    assert result["reading"] == COULD_NOT_TELL


def test_a_gain_the_formed_model_made_by_itself_is_not_credited_to_the_second_stage():
    second_stage, formed, check2, original = _runs(original=(ALL, TWENTY), check2=(ALL, TWENTY),
                                                   formed=(NONE, TWENTY), second_stage=(NONE, TWENTY))
    result = excess(second_stage, formed, check2, original, "snap")
    assert result["wrong_adoption_excess"]["estimate"] == pytest.approx(0.0)
    assert result["reading"] == COULD_NOT_TELL


def test_the_four_runs_must_hold_the_same_episodes():
    second_stage, formed, check2, original = _runs(original=(ALL, TWENTY), check2=(ALL, TWENTY),
                                                   formed=(ALL, TWENTY), second_stage=(NONE, TWENTY))
    with pytest.raises(ValueError):
        excess(second_stage, formed, check2[1:], original[1:], "snap")
