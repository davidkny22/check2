"""Runs of the wide-signal variant on the tiny model: saved states, strength zero, Swap and the saved-state check."""

from __future__ import annotations

import pytest
import torch

from check2.data.episodes import TAIL_LENGTH
from check2.evaluation import swap
from check2.evaluation.runs import execution, output, run
from check2.gates import mechanism_setup as setup
from check2.gates import saved_state
from check2.mechanism import wide_signal
from check2.mechanism.control import ControlState
from check2.mechanism.write import Writers
from check2.reader import cache, wide_states
from check2.reader.crossfit import load_reader, save_reader
from check2.reader.model import Reader

from .conftest import config

WIDE = "probabilities_and_state"
WIDE_READ_LAYERS = [2, wide_signal.TINY_STATE_LAYER]


class _CacheTokenizer:
    """The tiny model's folded tokenizer, taking the keyword the cache's window code passes."""

    def __init__(self, base) -> None:
        self.base = base
        self.pad_token_id = base.pad_token_id

    def __call__(self, texts, add_special_tokens: bool = True):
        if isinstance(texts, str):
            return {"input_ids": self.base.encode(texts, add_special_tokens=add_special_tokens)}
        return self.base(texts)

    def convert_tokens_to_ids(self, token):
        return self.base.convert_tokens_to_ids(token)


@pytest.fixture(scope="module")
def wide_reader(bench, tmp_path_factory):
    torch.manual_seed(0)
    fitted = Reader(WIDE_READ_LAYERS, bench.model.config.hidden_size)
    with torch.no_grad():
        fitted.u.normal_()
    path = tmp_path_factory.mktemp("wide-reader") / "reader.pt"
    save_reader(fitted, path)
    return load_reader(path)


@pytest.fixture(scope="module")
def wide_trained(bench, small_episodes):
    """Random wide writers, and adapter factors whose second factor is zero, so the adapter adds exactly nothing."""
    writers, adapter = setup.random_additions(bench, ControlState(), small_episodes, writer_input=WIDE)
    factors = {name: (a.detach().clone(), torch.zeros_like(b)) for name, (a, b) in adapter.factors().items()}
    return writers.state_dict(), factors


@pytest.fixture()
def wide(checkpoints, wide_trained, tmp_path, monkeypatch):
    checkpoints.answer = wide_trained
    monkeypatch.setattr(execution, "WIDE_STATES_ROOT", tmp_path / "wide")
    return checkpoints


def _go(bench, reader, cfg, run_id, folder):
    return run(cfg, run_id, bench.model, bench.tokenizer, bench.layout, reader, episode_folder=folder)


def _state_alone(bench, episode: dict) -> torch.Tensor:
    ids = bench.tokenizer([episode["prompt"]])["input_ids"][0]
    read_index = episode["n_prompt_tokens"] - TAIL_LENGTH
    with torch.no_grad():
        out = bench.model(input_ids=torch.tensor([ids[:read_index + 1]]), output_hidden_states=True)
    return out.hidden_states[wide_signal.TINY_STATE_LAYER + 1][0, -1]


def _n(h: torch.Tensor) -> torch.Tensor:
    h = h.to(torch.float32)
    return (h - h.mean()) / torch.sqrt(((h - h.mean()) ** 2).mean() + 1e-5)


WIDE_CHECK2 = {"adapter.kind": "consumer", "writer.alpha": 1.0, "writer.input": WIDE}


def test_a_wide_run_saves_each_episodes_own_live_state_and_records_three_probabilities(
        bench, wide_reader, results, episode_folder, small_episodes, wide):
    folder = _go(bench, wide_reader, config(**WIDE_CHECK2), "wide-check2-heldout", episode_folder)
    saved = execution.load_wide_states("wide-check2-heldout")
    assert sorted(saved) == sorted(e["episode_id"] for e in small_episodes)
    for e in small_episodes:
        assert torch.allclose(saved[e["episode_id"]], _state_alone(bench, e), rtol=1e-4, atol=1e-5), e["episode_id"]
    assert all(len(r["p"]) == 3 and len(r["p_used"]) == 3 for r in output.read_records(folder))


def test_a_wide_run_at_strength_zero_gives_the_original_replies(bench, wide_reader, results, episode_folder, wide):
    original = _go(bench, wide_reader, config(), "original", episode_folder)
    off = _go(bench, wide_reader, config(**{**WIDE_CHECK2, "writer.alpha": 0.0}), "wide-off", episode_folder)
    on = _go(bench, wide_reader, config(**WIDE_CHECK2), "wide-on", episode_folder)

    def replies(folder):
        return [(r["episode_id"], r["reply"], r["reply_tokens"], r["a2"]) for r in output.read_records(folder)]
    assert replies(off) == replies(original)
    assert replies(on) != replies(original)


def test_a_wide_swap_gives_each_episode_its_donors_p_and_its_donors_state(
        bench, wide_reader, results, episode_folder, wide, monkeypatch):
    _go(bench, wide_reader, config(**WIDE_CHECK2), "wide-check2-heldout", episode_folder)
    donor_states = execution.load_wide_states("wide-check2-heldout")
    seen: list[torch.Tensor] = []
    original = Writers.forward

    def spy(self, signal):
        seen.append(signal.detach().clone())
        return original(self, signal)
    monkeypatch.setattr(Writers, "forward", spy)
    folder = _go(bench, wide_reader, config(**{**WIDE_CHECK2, "signal": "swapped"}), "wide-swap-heldout",
                 episode_folder)
    monkeypatch.undo()
    records = output.read_records(folder)
    # Two writer calls a batch, generation and the appended read, both with the batch's S.
    sizes = [s.shape[0] for s in seen[::2]]
    assert sum(sizes) == len(records) and len(seen) == 2 * len(sizes)
    start = 0
    for k, size in enumerate(sizes):
        for b in range(size):
            r = records[start + b]
            for signal in (seen[2 * k], seen[2 * k + 1]):
                if r["donor"] is None:
                    assert r["situation"] == "recheck"
                    continue
                assert torch.allclose(signal[b, :3], torch.tensor(r["p_used"]), atol=1e-6)
                assert torch.allclose(signal[b, 3:], _n(donor_states[r["donor"]]), rtol=1e-5, atol=1e-5)
        start += size
    assert any(r["donor"] is not None for r in records)


def test_the_plain_swap_still_finds_the_plain_check2_run_beside_a_wide_one(bench, reader, wide_reader, results,
                                                                          episode_folder, checkpoints, trained,
                                                                          wide_trained, tmp_path, monkeypatch):
    monkeypatch.setattr(execution, "WIDE_STATES_ROOT", tmp_path / "wide")
    checkpoints.answer = wide_trained
    _go(bench, wide_reader, config(**WIDE_CHECK2), "wide-check2-heldout", episode_folder)
    checkpoints.answer = trained
    plain = _go(bench, reader, config(**{"adapter.kind": "consumer", "writer.alpha": 1.0}), "check2-heldout",
                episode_folder)
    assert swap.check2_run("heldout") == plain
    assert swap.check2_run("heldout", writer_input=WIDE).name == "wide-check2-heldout"


def test_saved_state_equals_live_state_on_a_run_and_its_export_and_is_red_with_rows_rotated(
        bench, wide_reader, results, episode_folder, small_episodes, wide, tmp_path):
    folder = _go(bench, wide_reader, config(**WIDE_CHECK2), "wide-check2-heldout", episode_folder)
    cache.cache_states(bench.model, _CacheTokenizer(bench.tokenizer), small_episodes, tmp_path / "cache",
                       WIDE_READ_LAYERS, batch_size=4)
    exported = wide_states.write_states("heldout", [e["episode_id"] for e in small_episodes], 0, tmp_path / "cache",
                                        tmp_path / "export", layer=wide_signal.TINY_STATE_LAYER)
    values = saved_state.check(folder, exported)
    assert values["green"] and values["compared"] == len(small_episodes), values
    assert values["lowest_cosine"] > 0.9999
    planted = saved_state.check(folder, exported, plant=True)
    assert not planted["green"] and planted["lowest_cosine"] < saved_state.LOWEST_COSINE
