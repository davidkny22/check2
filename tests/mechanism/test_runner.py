"""The runner's two commands load the config for a run and call the function each names."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from check2.config import schema

RUNNER = Path(__file__).resolve().parents[2] / "experiments" / "run.py"


@pytest.fixture()
def runner():
    spec = importlib.util.spec_from_file_location("runner_under_test", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def config_file(tmp_path, monkeypatch):
    pins, selection = tmp_path / "pins.json", tmp_path / "selection.json"
    pins.write_text(json.dumps({"model_revision": "abc"}), encoding="utf-8")
    selection.write_text(json.dumps({"read_position": 1}), encoding="utf-8")
    monkeypatch.setattr(schema, "PINS_FILE", pins)
    monkeypatch.setattr(schema, "SELECTION_FILE", selection)
    path = tmp_path / "run.yaml"
    path.write_text("check2:\n  adapter:\n    kind: consumer\n", encoding="utf-8")
    return path


def test_a_command_whose_module_is_missing_says_so(runner, config_file, monkeypatch, capsys):
    monkeypatch.setitem(runner.COMMANDS, "train", ("check2.no_such_package.loop", "train"))
    assert runner.main(["train", str(config_file)]) != 0
    assert "does not exist yet" in capsys.readouterr().err


def test_train_calls_its_function_with_the_loaded_config(runner, config_file, monkeypatch):
    received = []
    fake = types.ModuleType("fake_run_module")
    fake.call = lambda config: received.append(config)
    monkeypatch.setitem(sys.modules, "fake_run_module", fake)
    monkeypatch.setitem(runner.COMMANDS, "train", ("fake_run_module", "call"))
    assert runner.main(["train", str(config_file)]) == 0
    (config,) = received
    assert config.adapter.kind == "consumer"
    assert config.model_revision == "abc" and config.reader.read_position == 1


def test_a_train_command_whose_function_returns_its_checkpoint_folder_exits_0(runner, config_file, monkeypatch,
                                                                              tmp_path):
    fake = types.ModuleType("fake_run_module")
    fake.call = lambda config: tmp_path / "checkpoints" / "consumer-reader-rank32-beta1-seed0"
    monkeypatch.setitem(sys.modules, "fake_run_module", fake)
    monkeypatch.setitem(runner.COMMANDS, "train", ("fake_run_module", "call"))
    assert runner.main(["train", str(config_file)]) == 0


@pytest.mark.parametrize("extra, epoch", [([], None), (["--epoch", "2"], 2)])
def test_evaluate_passes_the_config_file_and_the_epoch(runner, config_file, monkeypatch, extra, epoch):
    received = []
    fake = types.ModuleType("fake_run_module")
    fake.call = lambda config, checkpoint_epoch=None: received.append((config, checkpoint_epoch))
    monkeypatch.setitem(sys.modules, "fake_run_module", fake)
    monkeypatch.setitem(runner.COMMANDS, "evaluate", ("fake_run_module", "call"))
    assert runner.main(["evaluate", str(config_file), *extra]) == 0
    assert received == [(config_file, epoch)]


def test_an_epoch_is_refused_for_training(runner, config_file):
    with pytest.raises(SystemExit):
        runner.main(["train", str(config_file), "--epoch", "1"])


def test_the_commands_name_the_functions_other_parts_provide(runner):
    assert runner.COMMANDS == {"train": ("check2.training.loop", "train"),
                               "evaluate": ("check2.evaluation.runs", "execute")}


@pytest.mark.parametrize("settings, forming", [
    ("  signal_forming: formed_model_only\n  writer:\n    alpha: 0.0\n", True),
    ("  adapter:\n    kind: consumer\n", False),
    ("  adapter:\n    kind: consumer\n  signal_forming: formed_with_channel\n", False),
    ("  adapter:\n    kind: adapter_only\n  writer:\n    alpha: 0.0\n", False),
    ("  writer:\n    alpha: 1.0\n", False),
])
def test_a_formed_model_only_config_trains_the_forming_update_and_every_other_config_the_loop(
        runner, config_file, monkeypatch, settings, forming):
    from check2.training import loop
    from check2.training.forming import training

    called = []
    monkeypatch.setattr(training, "train", lambda config: called.append(("forming", config.signal_forming)))
    monkeypatch.setattr(loop, "train", lambda config: called.append(("loop", config.signal_forming)))
    config_file.write_text("check2:\n" + settings, encoding="utf-8")
    assert runner.main(["train", str(config_file)]) == 0
    assert len(called) == 1 and called[0][0] == ("forming" if forming else "loop")
    assert runner.FORMING_TRAINING == ("check2.training.forming.training", "train")
