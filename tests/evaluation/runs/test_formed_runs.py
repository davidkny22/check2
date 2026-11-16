"""Runs of a formed model on the tiny model: the update carried, `p` computed live on its states."""

from __future__ import annotations

import pytest
import torch

from check2.evaluation.runs import RunError, output, parts, run
from check2.evaluation.runs.parts import check_executable
from check2.training.forming import guard, refit
from check2.training.forming.update import FormingUpdate, layers_for

from .conftest import config

FORMED_MODEL = {"signal_forming": "formed_model_only"}
SECOND_STAGE = {"signal_forming": "formed_with_channel", "adapter.kind": "consumer", "writer.alpha": 1.0}


@pytest.fixture(scope="module")
def forming(bench):
    """The forming update's factors on the tiny model, with every `B` non-zero."""
    torch.manual_seed(11)
    update = FormingUpdate(bench.model, "hooks", *layers_for(bench.layout))
    with torch.no_grad():
        for _, b in update.factors().values():
            b.normal_(0.0, 0.05)
    return {name: (a.detach().clone(), b.detach().clone()) for name, (a, b) in update.factors().items()}


@pytest.fixture()
def loaded(monkeypatch, forming):
    """The run loads these factors as the forming training's; records which configs asked."""
    asked = []
    monkeypatch.setattr(parts, "forming_factors",
                        lambda cfg: asked.append(cfg.signal_forming) or (forming if cfg.signal_forming != "none"
                                                                         else None))
    return asked


def _hooks(bench) -> int:
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in bench.model.modules())


def _p(folder) -> dict[str, list[float]]:
    return {r["episode_id"]: r["p"] for r in output.read_records(folder)}


def test_a_formed_model_run_and_a_second_stage_run_record_a_live_p_that_differs_from_the_original_reader_s(
        bench, reader, results, episode_folder, checkpoints, loaded):
    hooks = _hooks(bench)
    original = run(config(), "original", bench.model, bench.tokenizer, bench.layout, reader,
                   episode_folder=episode_folder)
    formed = run(config(**FORMED_MODEL), "formed-model", bench.model, bench.tokenizer, bench.layout, reader,
                 episode_folder=episode_folder)
    assert _hooks(bench) == hooks
    second = run(config(**SECOND_STAGE), "formed-check2", bench.model, bench.tokenizer, bench.layout, reader,
                 episode_folder=episode_folder)
    assert _hooks(bench) == hooks
    assert loaded == ["none", "formed_model_only", "formed_with_channel"]
    assert len(checkpoints.calls) == 1 and checkpoints.calls[0][0].signal_forming == "formed_with_channel"
    p_original, p_formed, p_second = _p(original), _p(formed), _p(second)
    assert sorted(p_formed) == sorted(p_original)
    assert all(p_formed[e] != p_original[e] for e in p_original)
    # The same formed model's states give the same live p, whatever the channel adds after the read.
    assert p_second == p_formed
    assert all(r["p_used"] is None for r in output.read_records(formed))
    assert all(r["p_used"] == r["p"] for r in output.read_records(second))


def test_a_formed_run_reads_the_formed_reader_through_the_guard(bench, reader, tmp_path, monkeypatch, forming):
    from check2.reader import files as reader_files
    from check2.reader.crossfit import save_reader

    monkeypatch.setattr(refit, "READER_WEIGHTS", tmp_path / "formed" / "reader.pt")
    monkeypatch.setattr(reader_files, "READER_WEIGHTS", tmp_path / "original" / "reader.pt")
    save_reader(reader, tmp_path / "formed" / "reader.pt")
    save_reader(reader, tmp_path / "original" / "reader.pt")
    with pytest.raises(guard.StaleSignal):
        parts.run_reader(config(**FORMED_MODEL), forming)
    guard.write_source(tmp_path / "formed", "forming-rank32-seed0", forming)
    assert parts.run_reader(config(**FORMED_MODEL), forming).layers == reader.layers
    assert parts.run_reader(config(), None).layers == reader.layers


def test_a_formed_setting_with_parts_it_cannot_have_is_refused():
    check_executable(config(**FORMED_MODEL), None)
    check_executable(config(**SECOND_STAGE), None)
    with pytest.raises(RunError):
        check_executable(config(**{**FORMED_MODEL, "writer.alpha": 1.0}), None)
    with pytest.raises(RunError):
        check_executable(config(signal_forming="formed_with_channel"), None)
    with pytest.raises(RunError):
        check_executable(config(**{**SECOND_STAGE, "signal": "perfect"}), None)
