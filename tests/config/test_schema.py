"""The settings schema: every planned run is one config, and nothing outside the schema is accepted."""

from __future__ import annotations

import json

import pytest
import yaml

from check2.config.schema import ConfigError, SYSTEM_PROMPT_SUFFIX, load_config, parse_config, resolve

# Each run of the evaluation plan, written as the settings that differ from the reference's.
# Original has no writer and no adapter; with the write at strength 0 and no adapter, nothing is added.
RUNS = {
    "original": {"writer": {"alpha": 0.0}, "adapter": {"kind": "none"}},
    "connector": {"adapter": {"kind": "none"}},
    "check2": {"adapter": {"kind": "consumer"}},
    "off": {"adapter": {"kind": "consumer"}, "writer": {"alpha": 0}},
    "swap": {"adapter": {"kind": "consumer"}, "signal": "swapped"},
    "adapter_only": {"adapter": {"kind": "adapter_only"}, "writer": {"alpha": 0.0}},
    "perfect_signal": {"adapter": {"kind": "consumer"}, "signal": "perfect"},
    "perfect_signal_rank_64": {"adapter": {"kind": "consumer", "rank": 64, "alpha_lora": 128}, "signal": "perfect"},
    "system_prompt": {"writer": {"alpha": 0.0}, "baseline": {"system_prompt_suffix": SYSTEM_PROMPT_SUFFIX}},
    **{f"steering_{g}": {"writer": {"alpha": 0.0}, "baseline": {"steering_strength": g}} for g in (1, 2, 4, 8)},
    "dial_half": {"adapter": {"kind": "consumer"}, "writer": {"alpha": 0.5}},
    "dial_double": {"adapter": {"kind": "consumer"}, "writer": {"alpha": 2.0}},
    **{f"band_{i}": {"adapter": {"kind": "consumer"}, "writer": {"disabled_layers": band}}
       for i, band in enumerate(([3, 6, 9], [11, 15], [18, 21, 23]))},
    "formed_model": {"writer": {"alpha": 0.0}, "signal_forming": "formed_model_only"},
    "formed_with_channel": {"adapter": {"kind": "consumer"}, "signal_forming": "formed_with_channel"},
    "second_seed": {"adapter": {"kind": "consumer"}, "seed": 1},
    "off_repeat_beta_4": {"adapter": {"kind": "consumer", "beta": 4.0}, "writer": {"alpha": 0.0}},
    "neutral_original": {"writer": {"alpha": 0.0}, "data": {"forms": ["neutral"]}},
    "neutral_check2": {"adapter": {"kind": "consumer"}, "data": {"forms": ["neutral"]}},
    "crosswriter_original": {"writer": {"alpha": 0.0}, "data": {"forms": ["reasoned"], "reason_writer": "sonnet"}},
    "crosswriter_check2": {"adapter": {"kind": "consumer"}, "data": {"forms": ["reasoned"], "reason_writer": "sonnet"}},
    "heldout_original": {"writer": {"alpha": 0.0}, "data": {"eval_split": "heldout"}},
    "steering_direction_pass": {"writer": {"alpha": 0.0}, "data": {"eval_split": "train", "forms": ["named", "reasoned"]}},
    "training_whole_sequences": {"adapter": {"kind": "consumer"}, "training": {"shared_prefix": False}},
    "training_builtin_kernel": {"adapter": {"kind": "consumer"}, "training": {"delta_kernel": "builtin_bf16"}},
    "decode_settings": {"generation_cache": "preallocated", "blas": "cublaslt", "attention_backend": "cudnn"},
    "target_samples_8": {"data": {"target_samples": 8}},
    "reader_linear_probe": {"reader": {"layers": [18], "pooled": False}},
    "reader_reason_only": {"reader": {"input": "reason_only"}},
}


def _write(tmp_path, body: dict, suffix: str = ".yaml"):
    path = tmp_path / f"config{suffix}"
    text = json.dumps({"check2": body}) if suffix == ".json" else yaml.safe_dump({"check2": body})
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("name", sorted(RUNS))
def test_every_run_is_one_config(tmp_path, name):
    config = load_config(_write(tmp_path, RUNS[name]), for_run=False)
    assert config.model == "Qwen/Qwen3.5-4B"


def test_the_runs_that_must_differ_are_different_settings():
    settings = {name: parse_config({"check2": body}) for name, body in RUNS.items()}
    assert len(set(settings.values())) == len(settings)


def test_defaults_are_the_reference_values():
    config = parse_config({"check2": {}})
    assert config.writer.layers == (3, 6, 9, 11, 15, 18, 21, 23)
    assert config.writer.hidden == 32 and config.writer.alpha == 1.0 and config.writer.disabled_layers == ()
    assert config.reader.layers == (14, 15, 18, 19, 22, 23) and config.reader.pooled is True
    assert (config.adapter.rank, config.adapter.alpha_lora, config.adapter.layers) == (32, 64, (3, 31))
    assert config.data.formats == ("snap", "short") and config.seed == 0
    assert config.libraries.flash_linear_attention == "0.6.0"


@pytest.mark.parametrize("body", [
    {"writer": {"strength": 1.0}},
    {"adapterr": {}},
    {"data": {"formats": ["snap"], "cap": 16}},
    {"libraries": {"flash_linear_attention": "0.6.0"}},
])
def test_an_unknown_key_is_refused(body):
    with pytest.raises(ConfigError, match="unknown key"):
        parse_config({"check2": body})


@pytest.mark.parametrize("body", [
    {"writer": {"alpha": 1.5}},
    {"writer": {"disabled_layers": [3, 6]}},
    {"writer": {"layers": [3, 6, 9]}},
    {"adapter": {"kind": "lora"}},
    {"adapter": {"rank": 16}},
    {"adapter": {"rank": 64}},
    {"adapter": {"kind": "adapter_only"}},
    {"adapter": {"kind": "adapter_only"}, "writer": {"alpha": 2.0}},
    {"signal": "random"},
    {"data": {"eval_split": "dev"}},
    {"data": {"formats": ["snap", "snap"]}},
    {"data": {"forms": []}},
    {"reader": {"layers": [18, 19]}},
    {"reader": {"layers": [18]}},
    {"reader": {"pooled": 1}},
    {"baseline": {"steering_strength": 3}},
    {"seed": 2},
    {"model": "Qwen/Qwen3-4B"},
    {"model_revision": "main"},
])
def test_a_value_outside_its_set_is_refused(body):
    with pytest.raises(ConfigError):
        parse_config({"check2": body})


def test_a_config_needs_the_one_top_level_key():
    with pytest.raises(ConfigError):
        parse_config({"writer": {}})


def test_placeholders_resolve_from_their_files(tmp_path):
    pins, selection = tmp_path / "pins.json", tmp_path / "selection.json"
    pins.write_text(json.dumps({"model_revision": "abc123"}), encoding="utf-8")
    selection.write_text(json.dumps({"read_position": 2}), encoding="utf-8")
    config = resolve(parse_config({"check2": {}}), pins_file=pins, selection_file=selection)
    assert config.model_revision == "abc123" and config.reader.read_position == 2


def test_an_absent_pinned_file_is_an_error_for_a_run(tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"read_position": 0}), encoding="utf-8")
    with pytest.raises(ConfigError, match="does not exist"):
        resolve(parse_config({"check2": {}}), pins_file=tmp_path / "missing.json", selection_file=selection)
    pins = tmp_path / "pins.json"
    pins.write_text(json.dumps({"model_revision": "abc"}), encoding="utf-8")
    with pytest.raises(ConfigError, match="does not exist"):
        resolve(parse_config({"check2": {}}), pins_file=pins, selection_file=tmp_path / "missing.json")


def test_json_and_yaml_read_alike(tmp_path):
    body = RUNS["band_1"]
    assert load_config(_write(tmp_path, body, ".json"), for_run=False) == load_config(_write(tmp_path, body), for_run=False)
