"""Checkpoint names, completeness, the Drive copy, and what a run loads."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from check2.config.schema import Config, load_config
from check2.training import checkpoints
from check2.training.drive import size_mismatches
from tests.training.fake_drive import FolderDrive

CONFIGS = Path(__file__).resolve().parents[2] / "experiments" / "configs"


def _config(split: str, name: str) -> Config:
    return load_config(CONFIGS / split / f"{name}.yaml", for_run=False)


def _with(config: Config, **adapter) -> Config:
    return replace(config, adapter=replace(config.adapter, **adapter))


def test_a_training_and_every_config_that_loads_it_share_one_name():
    consumer = checkpoints.training_id(_config("heldout", "check2"))
    for split, name in [("heldout", "off"), ("test", "check2"), ("test", "off"), ("test", "swap"),
                        ("test", "check2-alpha0.5"), ("test", "check2-alpha2"), ("test", "check2-bands-3-6-9-off"),
                        ("test", "check2-crosswriter"), ("test", "check2-neutral")]:
        assert checkpoints.training_id(_config(split, name)) == consumer, name
    assert checkpoints.training_id(_config("test", "connector")) == checkpoints.training_id(
        _config("heldout", "connector"))
    assert checkpoints.training_id(_config("test", "perfect-signal")) == checkpoints.training_id(
        _config("heldout", "perfect-signal"))
    assert checkpoints.training_id(_config("test", "adapter-only")) == checkpoints.training_id(
        _config("heldout", "adapter-only"))


def test_the_name_differs_when_any_distinguishing_setting_differs():
    base = _config("heldout", "check2")
    names = {
        "main": checkpoints.training_id(base),
        "perfect": checkpoints.training_id(replace(base, signal="perfect")),
        "rank64": checkpoints.training_id(_with(base, rank=64, alpha_lora=128)),
        "beta4": checkpoints.training_id(_with(base, beta=4.0)),
        "seed1": checkpoints.training_id(replace(base, seed=1)),
        "formed": checkpoints.training_id(replace(base, signal_forming="formed_with_channel")),
        "connector": checkpoints.training_id(_with(base, kind="none")),
        "adapter_only": checkpoints.training_id(replace(_with(base, kind="adapter_only"),
                                                        writer=replace(base.writer, alpha=0.0))),
    }
    assert len(set(names.values())) == len(names), names
    assert names["main"] != checkpoints.training_id(_config("heldout", "check2-beta4"))
    assert names["main"] != checkpoints.training_id(_config("heldout", "check2-seed1"))
    assert checkpoints.training_id(_config("heldout", "perfect-signal")) != checkpoints.training_id(
        _config("heldout", "perfect-signal-rank64"))


def test_a_connector_name_ignores_the_adapter_s_rank_and_beta():
    connector = _config("heldout", "connector")
    assert checkpoints.training_id(connector) == checkpoints.training_id(_with(connector, rank=64, alpha_lora=128))
    assert checkpoints.training_id(connector) == checkpoints.training_id(_with(connector, beta=4.0))
    rank64 = _config("heldout", "perfect-signal-rank64")
    assert checkpoints.connector_id(rank64) == checkpoints.connector_id(_config("heldout", "perfect-signal"))
    assert checkpoints.connector_id(rank64) == checkpoints.training_id(_config("heldout", "perfect-signal-connector"))


def test_a_config_with_no_trained_part_has_no_name_and_loads_nothing():
    original = _config("heldout", "original")
    with pytest.raises(checkpoints.CheckpointError):
        checkpoints.training_id(original)
    assert checkpoints.load_for_run(original) == (None, None)


def _state(seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {"writers": {"w": torch.randn(4, 3, generator=g)}, "adapter": {"m": (torch.randn(2, 3, generator=g),
                                                                                torch.randn(3, 2, generator=g))},
            "optimizer": {"state": {}, "param_groups": []}, "schedule": {"last_epoch": 7},
            "generators": {"torch": torch.get_rng_state()}, "step": 7}


def test_a_saved_checkpoint_loads_back_equal_and_is_not_overwritten(tmp_path):
    storage = checkpoints.Storage(tmp_path / "local")
    folder = checkpoints.save("t", 1, _state(0), storage)
    assert (folder / checkpoints.MARKER).is_file()
    loaded = checkpoints.load("t", 1, storage)
    assert torch.equal(loaded["writers"]["w"], _state(0)["writers"]["w"])
    assert torch.equal(loaded["adapter"]["m"][1], _state(0)["adapter"]["m"][1])
    assert loaded["step"] == 7 and loaded["schedule"] == {"last_epoch": 7}
    with pytest.raises(checkpoints.CheckpointError):
        checkpoints.save("t", 1, _state(1), storage)


def test_an_incomplete_checkpoint_is_never_loaded(tmp_path):
    storage = checkpoints.Storage(tmp_path / "local")
    folder = checkpoints.save("t", 1, _state(0), storage)
    checkpoints.save("t", 2, _state(1), storage)
    assert checkpoints.complete_epochs("t", storage) == [1, 2]

    (storage.folder("t", 2) / checkpoints.MARKER).unlink()
    assert checkpoints.complete_epochs("t", storage) == [1]
    with pytest.raises(checkpoints.CheckpointError):
        checkpoints.load("t", 2, storage)

    # A file that does not match the marker: a write cut short.
    data = (folder / "writers.pt").read_bytes()
    (folder / "writers.pt").write_bytes(data[: len(data) // 2])
    assert checkpoints.complete_epochs("t", storage) == []
    with pytest.raises(checkpoints.CheckpointError):
        checkpoints.load("t", 1, storage)

    # A marker of another epoch or training.
    checkpoints.save("u", 1, _state(0), storage)
    marker = storage.folder("u", 1) / checkpoints.MARKER
    marker.write_text(json.dumps({**json.loads(marker.read_text()), "epoch": 3}))
    with pytest.raises(checkpoints.CheckpointError):
        checkpoints.load("u", 1, storage)


def test_each_checkpoint_goes_to_drive_marker_last_and_comes_back_when_missing_here(tmp_path):
    drive = FolderDrive(tmp_path / "drive")
    storage = checkpoints.Storage(tmp_path / "session-1", "checkpoints", drive)
    checkpoints.save("t", 1, _state(0), storage)
    assert drive.calls == [("put", "checkpoints/t/epoch1", (checkpoints.MARKER,)), ("put", "checkpoints/t/epoch1", ())]
    assert drive.listing("checkpoints/t/epoch1") == {
        p: s for p, s in checkpoints.local_listing(storage.folder("t", 1)).items()}

    lost = checkpoints.Storage(tmp_path / "session-2", "checkpoints", drive)
    assert checkpoints.complete_epochs("t", lost) == [1]
    loaded = checkpoints.load("t", 1, lost)
    assert torch.equal(loaded["writers"]["w"], _state(0)["writers"]["w"])
    assert (lost.folder("t", 1) / checkpoints.MARKER).is_file()


def test_a_drive_copy_without_its_marker_is_not_fetched(tmp_path):
    drive = FolderDrive(tmp_path / "drive")
    storage = checkpoints.Storage(tmp_path / "session-1", "checkpoints", drive)
    checkpoints.save("t", 1, _state(0), storage)
    (tmp_path / "drive" / "checkpoints" / "t" / "epoch1" / checkpoints.MARKER).unlink()
    lost = checkpoints.Storage(tmp_path / "session-2", "checkpoints", drive)
    assert checkpoints.complete_epochs("t", lost) == []
    with pytest.raises(checkpoints.CheckpointError):
        checkpoints.load("t", 1, lost)


def test_the_size_check_names_a_missing_or_shorter_file():
    assert size_mismatches({"a": 3, "b": 4}, {"a": 3, "b": 4, "c": 1}) == []
    assert size_mismatches({"a": 3, "b": 4}, {"a": 3, "b": 2}) == ["b"]
    assert size_mismatches({"a": 3, "b": 4}, {"a": 3}) == ["b"]


def _write_chosen(root: Path, folder: str, chosen: dict) -> None:
    (root / folder).mkdir(parents=True, exist_ok=True)
    (root / folder / checkpoints.CHOSEN_FILE).write_text(json.dumps(chosen))


def test_the_chosen_epoch_comes_from_its_file_and_an_absent_file_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoints, "CHOSEN_ROOT", tmp_path)
    with pytest.raises(checkpoints.CheckpointError):
        checkpoints.chosen_epoch("t")
    _write_chosen(tmp_path, "t", {"training_id": "t", "epoch": 2, "record": "r.json"})
    assert checkpoints.chosen_epoch("t") == 2


def test_a_chosen_epoch_file_of_another_training_or_of_none_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoints, "CHOSEN_ROOT", tmp_path)
    _write_chosen(tmp_path, "t", {"training_id": "u", "epoch": 2, "record": "r.json"})
    with pytest.raises(checkpoints.CheckpointError, match="'u'.*'t'"):
        checkpoints.chosen_epoch("t")
    _write_chosen(tmp_path, "t", {"epoch": 2, "record": "r.json"})
    with pytest.raises(checkpoints.CheckpointError, match="names no training.*'t'"):
        checkpoints.chosen_epoch("t")


def test_consumer_training_does_not_continue_from_an_epoch_chosen_for_another_training(tmp_path, monkeypatch):
    # The reviewers' input: the file the rule wrote for the consumer training's held-out runs sits in the folder of
    # the connector training it continues from, once as the rule used to write it and once naming the consumer.
    monkeypatch.setattr(checkpoints, "CHOSEN_ROOT", tmp_path)
    consumer = _with(Config(), kind="consumer", implementation="hooks")
    connector_name = checkpoints.connector_id(consumer)
    consumer_name = checkpoints.training_id(consumer)
    for written in ({"epoch": 2, "record": "a-record-of-the-rule-over-check2-heldout-epoch1-and-epoch2.json"},
                    {"training_id": consumer_name, "epoch": 2, "record": "r.json"}):
        _write_chosen(tmp_path, connector_name, written)
        with pytest.raises(checkpoints.CheckpointError, match=connector_name):
            checkpoints.chosen_epoch(connector_name)


def test_a_run_loads_the_chosen_epoch_or_the_one_it_names(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoints, "CHOSEN_ROOT", tmp_path / "chosen")
    storage = checkpoints.Storage(tmp_path / "local")
    config = _config("heldout", "check2")
    name = checkpoints.training_id(config)
    checkpoints.save(name, 1, _state(1), storage)
    checkpoints.save(name, 2, _state(2), storage)
    _write_chosen(tmp_path / "chosen", name, {"training_id": name, "epoch": 1, "record": "r"})
    writers, factors = checkpoints.load_for_run(config, storage=storage)
    assert torch.equal(writers["w"], _state(1)["writers"]["w"])
    writers, factors = checkpoints.load_for_run(config, 2, storage=storage)
    assert torch.equal(factors["m"][0], _state(2)["adapter"]["m"][0])
