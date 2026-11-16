"""The answer token's weight as a setting: its values, where it is refused, the name it gives a training,
and where it is written."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from check2.config.schema import Config, ConfigError, load_config, parse_config
from check2.evaluation.roles import run_role
from check2.evaluation.runs import settings_of
from check2.training import checkpoints, loop
from tests.config.test_writer_input import EXISTING_NAMES

CONFIGS = Path(__file__).resolve().parents[2] / "experiments" / "configs"

# The training each config with the wide writer input names, by the naming rule of those trainings.
WIDE_NAMES = {
    "heldout/wide-check2.yaml": "consumer-reader-wide-rank32-beta1-seed0",
    "heldout/wide-connector.yaml": "connector-reader-wide-seed0",
    "heldout/wide-off.yaml": "consumer-reader-wide-rank32-beta1-seed0",
    "test/wide-check2.yaml": "consumer-reader-wide-rank32-beta1-seed0",
    "test/wide-off.yaml": "consumer-reader-wide-rank32-beta1-seed0",
    "test/wide-swap.yaml": "consumer-reader-wide-rank32-beta1-seed0",
}
# Every config that takes no answer weight, with the training it names.
UNWEIGHTED_NAMES = {**EXISTING_NAMES, **WIDE_NAMES}


def _consumer(weight: float, **more) -> dict:
    return {"check2": {"adapter": {"kind": "consumer"}, "training": {"answer_token_weight": weight}, **more}}


def test_the_default_weight_is_zero():
    assert Config().training.answer_token_weight == 0.0
    assert parse_config({"check2": {}}).training.answer_token_weight == 0.0


def test_a_half_is_accepted_for_consumer_training():
    config = parse_config(_consumer(0.5))
    assert config.training.answer_token_weight == 0.5
    assert parse_config(_consumer(0.5, signal="perfect")).training.answer_token_weight == 0.5


@pytest.mark.parametrize("weight", [0.25, 1.0, 0.49, -0.5, "0.5", True, None])
def test_any_other_weight_is_refused(weight):
    with pytest.raises(ConfigError):
        parse_config(_consumer(weight))


@pytest.mark.parametrize("raw", [
    {"training": {"answer_token_weight": 0.5}},
    {"adapter": {"kind": "none"}, "training": {"answer_token_weight": 0.5}},
    {"adapter": {"kind": "adapter_only"}, "writer": {"alpha": 0.0}, "training": {"answer_token_weight": 0.5}},
])
def test_a_half_is_refused_unless_the_adapter_is_the_consumer(raw):
    with pytest.raises(ConfigError, match="answer_token_weight"):
        parse_config({"check2": raw})
    unweighted = {**raw, "training": {"answer_token_weight": 0.0}}
    assert parse_config({"check2": unweighted}).training.answer_token_weight == 0.0


def test_the_weighted_training_has_the_existing_name_followed_by_its_weight():
    control = parse_config(_consumer(0.5, signal="perfect"))
    assert checkpoints.training_id(control) == "consumer-perfect-rank32-beta1-seed0-answer0.5"
    assert checkpoints.training_id(parse_config(_consumer(0.5))) == "consumer-reader-rank32-beta1-seed0-answer0.5"
    for config in (control, parse_config(_consumer(0.5))):
        unweighted = replace(config, training=replace(config.training, answer_token_weight=0.0))
        assert checkpoints.training_id(config) == checkpoints.training_id(unweighted) + "-answer0.5"
        # It continues from the connector training its unweighted counterpart continues from.
        assert checkpoints.connector_id(config) == checkpoints.connector_id(unweighted)


def test_every_config_without_the_weight_names_the_training_it_named_and_gives_its_file_as_its_settings():
    found = {p.relative_to(CONFIGS).as_posix(): p for p in sorted(CONFIGS.rglob("*.yaml"))}
    unweighted = {k: p for k, p in found.items()
                  if "answer_token_weight" not in yaml.safe_load(p.read_text(encoding="utf-8"))["check2"]["training"]}
    assert sorted(unweighted) == sorted(UNWEIGHTED_NAMES)
    for key, path in unweighted.items():
        config = load_config(path, for_run=False)
        name = checkpoints.training_id(config) if checkpoints.has_trained_part(config) else None
        assert name == UNWEIGHTED_NAMES[key], key
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert settings_of(config) == raw, key
        assert "answer_token_weight" not in settings_of(config)["check2"]["training"], key


def test_the_settings_and_an_epoch_record_carry_the_weight_only_when_it_is_not_zero():
    unweighted, weighted = parse_config(_consumer(0.0)), parse_config(_consumer(0.5))
    assert "answer_token_weight" not in settings_of(unweighted)["check2"]["training"]
    assert settings_of(weighted)["check2"]["training"]["answer_token_weight"] == 0.5
    assert parse_config(settings_of(weighted)) == weighted
    assert "answer_token_weight" not in loop._recorded(unweighted)["training"]
    assert loop._recorded(weighted)["training"]["answer_token_weight"] == 0.5
    # Nothing else of the record's config moves.
    with_key = {**loop._recorded(unweighted)["training"], "answer_token_weight": 0.5}
    assert loop._recorded(weighted) == {**loop._recorded(unweighted), "training": with_key}


def test_a_run_with_the_weight_has_a_role_of_its_own():
    plain = settings_of(parse_config(_consumer(0.0, signal="perfect")))
    weighted = settings_of(parse_config(_consumer(0.5, signal="perfect")))
    assert run_role(plain) == ("perfect signal", "")
    assert run_role(weighted) == ("perfect signal answer_weight=0.5", "")


def test_the_answer_weighted_control_differs_from_the_control_in_the_weight_alone_and_has_a_name_of_its_own():
    control = yaml.safe_load((CONFIGS / "heldout" / "perfect-signal.yaml").read_text(encoding="utf-8"))
    weighted_file = CONFIGS / "heldout" / "perfect-signal-answer-weighted.yaml"
    weighted = yaml.safe_load(weighted_file.read_text(encoding="utf-8"))
    assert weighted["check2"]["training"].pop("answer_token_weight") == 0.5
    assert weighted == control
    config = load_config(weighted_file, for_run=False)
    name = checkpoints.training_id(config)
    assert name == "consumer-perfect-rank32-beta1-seed0-answer0.5"
    assert name not in {n for n in UNWEIGHTED_NAMES.values() if n is not None}
    assert checkpoints.connector_id(config) == "connector-perfect-seed0"
    found = [p for p in sorted(CONFIGS.rglob("*.yaml"))
             if "answer_token_weight" in yaml.safe_load(p.read_text(encoding="utf-8"))["check2"]["training"]]
    assert found == sorted([weighted_file, CONFIGS / "heldout" / "check2-answer-weighted.yaml",
                            CONFIGS / "test" / "check2-answer-weighted.yaml"])


def test_check2_with_the_answer_weight_differs_from_check2_in_the_weight_alone_and_has_a_name_of_its_own():
    for split in ("heldout", "test"):
        plain = yaml.safe_load((CONFIGS / split / "check2.yaml").read_text(encoding="utf-8"))
        weighted_file = CONFIGS / split / "check2-answer-weighted.yaml"
        weighted = yaml.safe_load(weighted_file.read_text(encoding="utf-8"))
        assert weighted["check2"]["training"].pop("answer_token_weight") == 0.5
        assert weighted == plain
        config = load_config(weighted_file, for_run=False)
        assert checkpoints.training_id(config) == "consumer-reader-rank32-beta1-seed0-answer0.5"
        assert checkpoints.connector_id(config) == "connector-reader-seed0"
