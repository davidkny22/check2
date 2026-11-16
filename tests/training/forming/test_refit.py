"""The formed model's cache by the reader's own code, and the reader fitted on it from scratch."""

from __future__ import annotations

import pytest
import torch

from check2.reader import model as reader_model
from check2.reader.cache import StateBank, cache_states, episode_rows
from check2.reader.inputs import Inputs
from check2.reader.model import OUTPUTS
from check2.reader.window import window_length
from check2.training.forming import guard
from check2.training.forming.refit import cache_formed_states, refit, save_outputs
from check2.training.forming.update import FormingUpdate
from tests.reader.helpers import FakeTokenizer, made_up_episodes
from tests.tiny_model import tiny_model

LAYERS = (1, 2, 5)
QUESTIONS = 15
FOLDS = [sorted(f"q/{q}" for q in range(start, QUESTIONS, 5)) for start in range(5)]


def _episodes(tokenizer):
    train = made_up_episodes(tokenizer, QUESTIONS)
    heldout = made_up_episodes(tokenizer, 6, split="heldout", seed=1)
    for e in heldout:
        e["episode_id"], e["qid"] = "h" + e["episode_id"], "h" + e["qid"]
    for rows in (train, heldout):
        for i, e in enumerate(rows):
            e["situation"] = OUTPUTS[i % 3]
    return train, heldout


def _update(model, nonzero: bool) -> FormingUpdate:
    torch.manual_seed(5)
    update = FormingUpdate(model, "hooks", 1, 5)
    if nonzero:
        with torch.no_grad():
            for _, b in update.factors().values():
                b.normal_(0.0, 0.05)
    update.install()
    return update


@pytest.fixture(scope="module")
def made_up(tmp_path_factory):
    tokenizer = FakeTokenizer()
    train, heldout = _episodes(tokenizer)
    model = tiny_model()
    folder = tmp_path_factory.mktemp("caches")
    cache_states(model, tokenizer, train + heldout, folder / "original", LAYERS, batch_size=4)
    zero = _update(model, nonzero=False)
    try:
        cache_formed_states(model, tokenizer, zero, train + heldout, folder / "at_start", 0, LAYERS, batch_size=4)
    finally:
        zero.remove()
    update = _update(model, nonzero=True)
    try:
        cache_formed_states(model, tokenizer, update, train + heldout, folder / "formed", 0, LAYERS, batch_size=4)
    finally:
        update.remove()
    return train, heldout, folder, update


def test_at_initialisation_the_formed_cache_equals_the_original(made_up):
    train, heldout, folder, _ = made_up
    for e in train + heldout:
        assert torch.equal(episode_rows(folder / "original", e["episode_id"]),
                           episode_rows(folder / "at_start", e["episode_id"]))


def test_the_formed_cache_holds_the_formed_model_s_states_at_every_window_token(made_up):
    train, heldout, folder, _ = made_up
    for e in train + heldout:
        original = episode_rows(folder / "original", e["episode_id"])
        formed = episode_rows(folder / "formed", e["episode_id"])
        window = window_length(original.shape[0], 0)
        assert bool(((formed[:window] - original[:window]).flatten(1).norm(dim=-1) > 0).all())


def test_no_fold_s_fit_on_formed_states_holds_a_question_of_the_fold_it_scores(made_up, monkeypatch, tmp_path):
    train, heldout, folder, update = made_up
    monkeypatch.setitem(reader_model.VARIANTS, "tiny_variant",
                        {"layers": list(LAYERS), "pooled": True, "linear_probe": False})
    ids = sorted(e["episode_id"] for e in train + heldout)
    inputs = Inputs(train=train, heldout=heldout, bank=StateBank.from_cache(folder / "formed", ids),
                    bank_layers=list(LAYERS))
    result = refit(inputs, FOLDS, "tiny_variant", 0, seed=0)
    assert sorted(result.p_train) == inputs.train_ids and sorted(result.p_heldout) == inputs.heldout_ids
    for k, fold_fit in enumerate(result.fold_fits):
        assert fold_fit["scored_questions"] == FOLDS[k]
        assert set(fold_fit["trained_questions"]).isdisjoint(FOLDS[k])
        assert fold_fit["epochs"] == result.fit.best_epoch
    factors = update.factors()
    save_outputs(result, "forming-made-up", factors, outputs=tmp_path / "outputs",
                 reader_path=tmp_path / "session" / "reader.pt")
    assert guard.load_p(tmp_path / "outputs" / "p_heldout.jsonl", factors) == result.p_heldout
    with pytest.raises(guard.StaleSignal):
        guard.load_p(tmp_path / "outputs" / "p_train.jsonl", None)
    with pytest.raises(guard.StaleSignal):
        guard.load_fitted_reader(tmp_path / "session" / "reader.pt", None)
    assert guard.load_fitted_reader(tmp_path / "session" / "reader.pt", factors).layers == list(LAYERS)
