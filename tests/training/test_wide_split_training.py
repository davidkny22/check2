"""Split training equals plain with wide writers, on the tiny model: green in both forms, red under its fault."""

from __future__ import annotations

import pytest

from check2.gates import split_training, training_setup

WIDE = "probabilities_and_state"


@pytest.fixture(scope="module")
def episodes(targets_file):
    args = training_setup.parser("split training").parse_args(["--model", "tiny", "--targets", str(targets_file)])
    return training_setup.draw(args, split_training.EPISODES_PER_FORMAT)


def test_both_forms_match_the_plain_reference_with_wide_writers(bench, episodes, targets_file):
    values = split_training.check(bench, episodes, targets_file, writer_input=WIDE)
    assert values["reference_zero_gradients"] == []
    for name, form in values["forms"].items():
        assert form["error"] is None and form["green"], (name, form)
    assert values["green"]


def test_the_zeroed_recurrent_state_turns_the_shared_prefix_form_red_with_wide_writers(bench, episodes, targets_file):
    values = split_training.check(bench, episodes, targets_file, plant=True, writer_input=WIDE)
    assert not values["forms"]["shared_prefix"]["green"] and values["forms"]["whole_sequence"]["green"]
