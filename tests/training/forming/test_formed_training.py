"""Connector training on the formed model, on the tiny model: the update is loaded, acts, and stays frozen."""

from __future__ import annotations

import shutil
from dataclasses import replace

import pytest
import torch

from check2 import record
from check2.config.schema import Config
from check2.reader import files as reader_files
from check2.training import checkpoints, loop
from check2.training.forming import attach, guard, training
from check2.training.forming.update import FormingUpdate, layers_for
from tests.training.fake_drive import FolderDrive

EPISODES = 32  # two steps an epoch


def _config() -> Config:
    base = Config()
    return replace(base, signal_forming="formed_with_channel", reader=replace(base.reader, read_position=0),
                   writer=replace(base.writer, alpha=1.0), adapter=replace(base.adapter, implementation="hooks"))


@pytest.fixture(autouse=True)
def results(tmp_path, monkeypatch):
    monkeypatch.setattr(record, "RESULTS", str(tmp_path / "results"))


@pytest.fixture
def formed(bench, tmp_path):
    """A forming checkpoint with random factors on the storage, and the formed model's p beside its record."""
    storage = checkpoints.Storage(tmp_path / "session", "checkpoints", FolderDrive(tmp_path / "drive"))
    torch.manual_seed(13)
    update = FormingUpdate(bench.model, "hooks", *layers_for(bench.layout))
    with torch.no_grad():
        for _, b in update.factors().values():
            b.normal_(0.0, 0.05)
    factors = {name: (a.detach().clone(), b.detach().clone()) for name, (a, b) in update.factors().items()}
    name = training.training_id(training.forming_config(_config()))
    checkpoints.save(name, training.EPOCHS, {"writers": None, "adapter": factors, "optimizer": {}, "schedule": {},
                                             "generators": {}, "step": 0}, storage)
    p_file = tmp_path / "formed" / "p_train.jsonl"
    p_file.parent.mkdir()
    shutil.copyfile(reader_files.P_TRAIN, p_file)
    guard.write_source(p_file.parent, name, factors)
    return storage, factors, p_file


def test_a_training_on_the_formed_model_keeps_the_update_s_factors_bit_for_bit(bench, targets_file, formed,
                                                                              monkeypatch):
    storage, factors, p_file = formed
    updates, masks = [], []
    real_load = attach.load_frozen

    def spy(*args, **kwargs):
        update = real_load(*args, **kwargs)
        original_set = update.control.set
        update.control.set = lambda mask, *a, **k: masks.append(mask.clone()) or original_set(mask, *a, **k)
        updates.append(update)
        return update
    monkeypatch.setattr(attach, "load_frozen", spy)
    hooks = sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in bench.model.modules())
    options = loop.Options(storage=storage, targets_file=targets_file, p_file=p_file, episode_limit=EPISODES,
                           epochs=1, token_budget=float("inf"))
    loop.train(_config(), options, bench=bench)
    (update,) = updates
    for name, (a, b) in update.factors().items():
        assert torch.equal(a, factors[name][0]) and torch.equal(b, factors[name][1]), name
        assert not a.requires_grad and not b.requires_grad
    # The update acted in the shared prefix's pass over tokens 0..r, and nowhere after it.
    assert any(bool(m.any()) for m in masks) and any(not bool(m.any()) for m in masks)
    assert sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in bench.model.modules()) == hooks
    assert checkpoints.complete_epochs(checkpoints.training_id(_config()), storage) == [1]


def test_the_original_model_s_saved_p_is_refused_for_a_training_on_the_formed_model(bench, targets_file, formed):
    storage, _, _ = formed
    options = loop.Options(storage=storage, targets_file=targets_file, p_file=reader_files.P_TRAIN,
                           episode_limit=EPISODES, epochs=1, token_budget=float("inf"))
    with pytest.raises(guard.StaleSignal, match="original model"):
        loop.train(_config(), options, bench=bench)
