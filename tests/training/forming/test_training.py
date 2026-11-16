"""The forming training on the tiny model: what it trains, what it records, and how it resumes."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest
import torch

from check2 import record
from check2.config.schema import Config
from check2.reader.crossfit import save_reader
from check2.training import checkpoints
from check2.training.forming import training
from tests.training.fake_drive import FolderDrive
from tests.training.forming.helpers import tiny_reader

EPISODES = 32  # two steps an epoch
EPOCHS = 2


def _config() -> Config:
    base = Config()
    return replace(base, signal_forming="formed_model_only", reader=replace(base.reader, read_position=0),
                   writer=replace(base.writer, alpha=0.0), adapter=replace(base.adapter, implementation="hooks"))


@pytest.fixture(autouse=True)
def results(tmp_path, monkeypatch):
    monkeypatch.setattr(record, "RESULTS", str(tmp_path / "results"))
    return tmp_path / "results"


@pytest.fixture
def reader_path(bench, tmp_path):
    path = tmp_path / "reader" / "reader.pt"
    save_reader(tiny_reader(bench), path)
    return path


def _options(storage, reader_path) -> training.Options:
    return training.Options(storage=storage, reader_path=reader_path, episode_limit=EPISODES, epochs=EPOCHS,
                            token_budget=float("inf"))


def _files(folder) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(folder.iterdir())}


def test_the_training_moves_the_update_and_the_reader_copy_and_nothing_of_the_model(bench, reader_path, tmp_path,
                                                                                 results):
    before = copy.deepcopy(bench.model.state_dict())
    storage = checkpoints.Storage(tmp_path / "session", "checkpoints", FolderDrive(tmp_path / "drive"))
    config = _config()
    training.train(config, _options(storage, reader_path), bench=bench)
    name = training.training_id(config)
    assert name == "forming-rank32-seed0"
    assert checkpoints.complete_epochs(name, storage) == [1, 2]
    state = checkpoints.load(name, EPOCHS, storage)
    assert state["writers"] is None and len(state["adapter"]) == 31
    assert any(float(b.abs().max()) > 0 for _, b in state["adapter"].values())
    saved = torch.load(reader_path, weights_only=True)["state_dict"]
    assert any(not torch.equal(saved[k], v) for k, v in state["reader"].items())
    after = bench.model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    records = sorted((results / "measurements" / training.RECORD_NAME).glob("*.json"))
    first = json.loads(records[0].read_text(encoding="utf-8"))["values"]
    assert first["epoch"] == 1 and first["steps"] == 2
    assert first["first_step"]["kl"] == 0.0 and first["first_step"]["cross_entropy"] > 0
    assert (storage.local_root / name / "records" / records[0].name).exists()


def test_a_forming_training_stopped_after_its_first_epoch_and_resumed_gives_the_same_checkpoint(
        bench, reader_path, tmp_path, monkeypatch):
    config = _config()
    name = training.training_id(config)
    through = checkpoints.Storage(tmp_path / "through", "checkpoints", FolderDrive(tmp_path / "drive-through"))
    training.train(config, _options(through, reader_path), bench=bench)

    drive = FolderDrive(tmp_path / "drive-stopped")
    first_session = checkpoints.Storage(tmp_path / "first-session", "checkpoints", drive)
    real_save = checkpoints.save

    class Stopped(Exception):
        pass

    def save_then_stop(training_id, epoch, state, storage=None):
        out = real_save(training_id, epoch, state, storage)
        if epoch == 1:
            raise Stopped
        return out

    monkeypatch.setattr(checkpoints, "save", save_then_stop)
    with pytest.raises(Stopped):
        training.train(config, _options(first_session, reader_path), bench=bench)
    monkeypatch.setattr(checkpoints, "save", real_save)
    second_session = checkpoints.Storage(tmp_path / "second-session", "checkpoints", drive)
    training.train(config, _options(second_session, reader_path), bench=bench)
    assert _files(second_session.folder(name, EPOCHS)) == _files(through.folder(name, EPOCHS))


def test_a_config_that_names_no_forming_training_is_refused():
    with pytest.raises(training.TrainingError):
        training.training_id(Config())
