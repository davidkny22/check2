"""Split training equals plain on the tiny model: green in both forms, and red under its planted fault."""

from __future__ import annotations

import pytest

from check2.gates import split_training, training_setup


@pytest.fixture(scope="module")
def episodes(targets_file):
    args = training_setup.parser("split training").parse_args(["--model", "tiny", "--targets", str(targets_file)])
    return training_setup.draw(args, split_training.EPISODES_PER_FORMAT)


def test_both_forms_match_the_plain_reference(bench, episodes, targets_file):
    values = split_training.check(bench, episodes, targets_file)
    assert values["episodes"] == {"snap": 8, "short": 8}
    assert values["additions_scale"] == 0.01
    assert values["reference_zero_gradients"] == []
    for name, form in values["forms"].items():
        assert form["error"] is None, name
        assert form["batch_rows"] == [8, 8], name
        assert form["green"], (name, form)
    assert values["green"] and values["training_form"] == "shared_prefix"


def test_the_zeroed_recurrent_state_turns_the_shared_prefix_form_red(bench, episodes, targets_file):
    values = split_training.check(bench, episodes, targets_file, plant=True)
    shared, whole = values["forms"]["shared_prefix"], values["forms"]["whole_sequence"]
    assert not shared["green"] and shared["max_gradient_relative_difference"] > 10 * split_training.TOLERANCE
    assert whole["green"]
    assert not values["green"] and values["training_form"] == "whole_sequence"


def test_both_forms_match_the_plain_reference_on_the_answer_weighted_loss(bench, episodes, targets_file):
    plain = split_training.check(bench, episodes, targets_file)
    values = split_training.check(bench, episodes, targets_file, answer_weight=0.5)
    assert values["answer_token_weight"] == 0.5 and "answer_token_weight" not in plain
    positions = values["answer_positions"]
    assert [p is not None for p in positions] == [False] * 8 + [True] * 8
    # The weight moves the loss by more than ten times the tolerance, so a form that ignored it would be red.
    assert abs(values["reference_loss"] - plain["reference_loss"]) > 10 * split_training.TOLERANCE * abs(
        plain["reference_loss"])
    for name, form in values["forms"].items():
        assert form["error"] is None, name
        assert form["batch_rows"] == [8, 8], name
        assert form["green"], (name, form)
    assert values["green"] and values["training_form"] == "shared_prefix"


def test_the_zeroed_recurrent_state_turns_the_shared_prefix_form_red_on_the_answer_weighted_loss(bench, episodes,
                                                                                                targets_file):
    values = split_training.check(bench, episodes, targets_file, plant=True, answer_weight=0.5)
    shared, whole = values["forms"]["shared_prefix"], values["forms"]["whole_sequence"]
    assert not shared["green"] and shared["max_gradient_relative_difference"] > 10 * split_training.TOLERANCE
    assert whole["green"]
    assert not values["green"] and values["training_form"] == "whole_sequence"


def test_a_form_that_leaves_the_weight_out_is_red_against_the_weighted_reference(bench, episodes, targets_file,
                                                                               monkeypatch):
    real = split_training.consumer_step
    monkeypatch.setattr(split_training, "consumer_step",
                        lambda form, batch, p, beta, step_size, answer=None: real(form, batch, p, beta, step_size))
    values = split_training.check(bench, episodes, targets_file, answer_weight=0.5)
    assert not values["forms"]["shared_prefix"]["green"] and not values["forms"]["whole_sequence"]["green"]
