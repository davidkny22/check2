"""The stale-signal rule in its four cases, and the gate that shows it can fail."""

from __future__ import annotations

import pytest
import torch

from check2.gates import stale_signal
from check2.reader import files
from check2.reader.crossfit import save_reader
from check2.training.forming import guard
from tests.training.forming.helpers import random_update, tiny_reader

ROWS = [{"episode_id": "q/1:reasoned:snap:A", "p": [0.2, 0.3, 0.5]}]


def _factors(seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {"layers.1.mlp.up_proj": (torch.randn(4, 3, generator=g), torch.randn(5, 4, generator=g))}


@pytest.fixture
def folders(tmp_path):
    original, formed = tmp_path / "original", tmp_path / "formed"
    files.write_jsonl(original / "p.jsonl", ROWS)
    files.write_jsonl(formed / "p.jsonl", ROWS)
    guard.write_source(formed, "forming-made-up", _factors(0))
    return original / "p.jsonl", formed / "p.jsonl"


def test_with_an_update_the_original_model_s_outputs_raise(folders):
    original, _ = folders
    with pytest.raises(guard.StaleSignal, match="original model"):
        guard.load_p(original, _factors(0))


def test_with_an_update_outputs_recorded_from_it_load(folders):
    _, formed = folders
    assert guard.load_p(formed, _factors(0)) == {r["episode_id"]: r["p"] for r in ROWS}


def test_with_an_update_outputs_recorded_from_another_update_raise(folders):
    _, formed = folders
    with pytest.raises(guard.StaleSignal, match="not of the loaded update"):
        guard.load_p(formed, _factors(1))


def test_without_an_update_outputs_that_carry_a_record_raise(folders):
    _, formed = folders
    with pytest.raises(guard.StaleSignal, match="no forming update is loaded"):
        guard.load_p(formed, None)


def test_without_an_update_the_original_model_s_outputs_load(folders):
    original, _ = folders
    assert guard.load_p(original, None) == {r["episode_id"]: r["p"] for r in ROWS}


def test_a_fitted_reader_goes_through_the_same_rule(bench, tmp_path):
    reader = tiny_reader(bench)
    save_reader(reader, tmp_path / "original" / "reader.pt")
    save_reader(reader, tmp_path / "formed" / "reader.pt")
    guard.write_source(tmp_path / "formed", "forming-made-up", _factors(0))
    with pytest.raises(guard.StaleSignal):
        guard.load_fitted_reader(tmp_path / "original" / "reader.pt", _factors(0))
    with pytest.raises(guard.StaleSignal):
        guard.load_fitted_reader(tmp_path / "formed" / "reader.pt", None)
    loaded = guard.load_fitted_reader(tmp_path / "formed" / "reader.pt", _factors(0))
    assert all(torch.equal(a, b) for a, b in zip(loaded.state_dict().values(), reader.state_dict().values()))


def test_the_gate_is_green_on_the_tiny_model_and_red_with_the_check_removed(bench):
    update = random_update(bench)
    update.install()
    try:
        green = stale_signal.check(update, files.P_HELDOUT)
        red = stale_signal.check(update, files.P_HELDOUT, plant=True)
    finally:
        update.remove()
    assert green["green"] and all(c["as_required"] for c in green["cases"].values())
    assert green["cases"]["update_with_original_outputs"]["outcome"] == "raised"
    assert not red["green"]
    assert red["cases"]["update_with_original_outputs"]["outcome"] == "loaded"
    assert guard.check_source.__name__ == "check_source"
