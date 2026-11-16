"""The writer's input setting and the names of the trainings that use the wide input."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from check2.config.schema import Config, ConfigError, load_config, parse_config
from check2.training import checkpoints

CONFIGS = Path(__file__).resolve().parents[2] / "experiments" / "configs"

# The training each config file of the evaluation plan names, as the naming rule gave it before the
# wide input existed. A config with no trained part names none.
EXISTING_NAMES = {
    "heldout/adapter-only.yaml": "adapter-only-rank32-seed0",
    "heldout/check2-beta4.yaml": "consumer-reader-rank32-beta4-seed0",
    "heldout/check2-seed1.yaml": "consumer-reader-rank32-beta1-seed1",
    "heldout/check2.yaml": "consumer-reader-rank32-beta1-seed0",
    "heldout/connector-seed1.yaml": "connector-reader-seed1",
    "heldout/connector.yaml": "connector-reader-seed0",
    "heldout/formed-check2.yaml": "consumer-reader-rank32-beta1-seed0-formed",
    "heldout/formed-connector.yaml": "connector-reader-seed0-formed",
    "heldout/formed-model.yaml": None,
    "heldout/formed-off.yaml": "consumer-reader-rank32-beta1-seed0-formed",
    "heldout/off-beta4.yaml": "consumer-reader-rank32-beta4-seed0",
    "heldout/off.yaml": "consumer-reader-rank32-beta1-seed0",
    "heldout/original-repeat.yaml": None,
    "heldout/original.yaml": None,
    "heldout/perfect-signal-connector.yaml": "connector-perfect-seed0",
    "heldout/perfect-signal-rank64.yaml": "consumer-perfect-rank64-beta1-seed0",
    "heldout/perfect-signal.yaml": "consumer-perfect-rank32-beta1-seed0",
    "heldout/steering-g1.yaml": None,
    "heldout/steering-g2.yaml": None,
    "heldout/steering-g4.yaml": None,
    "heldout/steering-g8.yaml": None,
    "test/adapter-only.yaml": "adapter-only-rank32-seed0",
    "test/check2-alpha0.5.yaml": "consumer-reader-rank32-beta1-seed0",
    "test/check2-alpha2.yaml": "consumer-reader-rank32-beta1-seed0",
    "test/check2-bands-11-15-off.yaml": "consumer-reader-rank32-beta1-seed0",
    "test/check2-bands-18-21-23-off.yaml": "consumer-reader-rank32-beta1-seed0",
    "test/check2-bands-3-6-9-off.yaml": "consumer-reader-rank32-beta1-seed0",
    "test/check2-crosswriter.yaml": "consumer-reader-rank32-beta1-seed0",
    "test/check2-neutral.yaml": "consumer-reader-rank32-beta1-seed0",
    "test/check2-seed1.yaml": "consumer-reader-rank32-beta1-seed1",
    "test/check2.yaml": "consumer-reader-rank32-beta1-seed0",
    "test/connector.yaml": "connector-reader-seed0",
    "test/formed-check2.yaml": "consumer-reader-rank32-beta1-seed0-formed",
    "test/formed-model.yaml": None,
    "test/off.yaml": "consumer-reader-rank32-beta1-seed0",
    "test/original-believe.yaml": None,
    "test/original-crosswriter.yaml": None,
    "test/original-disagree.yaml": None,
    "test/original-expanded.yaml": None,
    "test/original-hmm.yaml": None,
    "test/original-neutral.yaml": None,
    "test/original-repeat.yaml": None,
    "test/original.yaml": None,
    "test/perfect-signal-rank64.yaml": "consumer-perfect-rank64-beta1-seed0",
    "test/perfect-signal.yaml": "consumer-perfect-rank32-beta1-seed0",
    "test/steering-g1.yaml": None,
    "test/steering-g2.yaml": None,
    "test/steering-g4.yaml": None,
    "test/steering-g8.yaml": None,
    "test/swap.yaml": "consumer-reader-rank32-beta1-seed0",
    "test/system-prompt.yaml": None,
    "train/original-train.yaml": None,
}

WIDE = "probabilities_and_state"


def _name(config: Config) -> str | None:
    if not checkpoints.has_trained_part(config):
        return None
    return checkpoints.training_id(config)


def _wide(config: Config) -> Config:
    return replace(config, writer=replace(config.writer, input=WIDE))


def test_the_default_input_is_the_three_probabilities():
    assert Config().writer.input == "probabilities"
    assert parse_config({"check2": {}}).writer.input == "probabilities"


def test_the_wide_input_is_accepted_with_the_readers_signal_live_or_swapped():
    for signal in ("reader", "swapped"):
        config = parse_config({"check2": {"signal": signal, "writer": {"input": WIDE}}})
        assert config.writer.input == WIDE


@pytest.mark.parametrize("raw", [
    {"signal": "perfect", "writer": {"input": WIDE}},
    {"adapter": {"kind": "adapter_only"}, "writer": {"alpha": 0.0, "input": WIDE}},
    {"writer": {"input": "state"}},
])
def test_the_wide_input_is_refused_where_it_has_no_meaning(raw):
    with pytest.raises(ConfigError):
        parse_config({"check2": raw})


def test_the_wide_trainings_have_names_of_their_own():
    connector = parse_config({"check2": {"adapter": {"kind": "none"}}})
    consumer = parse_config({"check2": {"adapter": {"kind": "consumer"}}})
    assert checkpoints.training_id(_wide(connector)) == "connector-reader-wide-seed0"
    assert checkpoints.training_id(_wide(consumer)) == "consumer-reader-wide-rank32-beta1-seed0"
    assert checkpoints.connector_id(_wide(consumer)) == "connector-reader-wide-seed0"
    swapped = replace(_wide(consumer), signal="swapped")
    assert checkpoints.training_id(swapped) == "consumer-reader-wide-rank32-beta1-seed0"


def test_every_existing_config_names_the_training_it_named_before():
    found = {p.relative_to(CONFIGS).as_posix(): p for p in sorted(CONFIGS.rglob("*.yaml"))}
    existing = {k: v for k, v in found.items()
                if not Path(k).name.startswith("wide-") and not Path(k).stem.endswith("-answer-weighted")}
    assert sorted(existing) == sorted(EXISTING_NAMES)
    for key, path in existing.items():
        assert _name(load_config(path, for_run=False)) == EXISTING_NAMES[key], key


def test_each_wide_config_names_a_training_no_existing_config_names():
    taken = {name for name in EXISTING_NAMES.values() if name is not None}
    for path in sorted(CONFIGS.rglob("wide-*.yaml")):
        config = load_config(path, for_run=False)
        assert config.writer.input == WIDE, path
        name = _name(config)
        assert name is not None and name not in taken, path
