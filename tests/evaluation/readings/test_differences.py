"""Pairing two runs' records: the same episodes, agreeing on what describes an episode."""

import pytest

from check2.evaluation import readings
from tests.evaluation.hand_records import ALL, reasoned_run


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


def test_runs_that_disagree_on_an_episode_are_refused():
    run = reasoned_run(ALL, [20] * 10)
    other = [dict(r) for r in run]
    other[0]["gold"] = "D"
    with pytest.raises(ValueError):
        readings.paired_records(run, other, readings.selection("snap"))


def test_two_records_with_one_episode_id_are_refused():
    run = reasoned_run(ALL, [20] * 10)
    with pytest.raises(ValueError):
        readings.paired_records(run + run[:1], run + run[:1], readings.selection("snap"))
