"""The cache saves the read layers' outputs over the window and the tail, and gives them back."""
import json

import pytest
import torch

from check2.data.episodes import TAIL_LENGTH
from check2.reader import cache
from check2.reader.cache import StateBank, cache_states, load_states, read_meta
from check2.reader.window import spans
from tests.reader.helpers import FakeTokenizer, made_up_episodes
from tests.tiny_model import tiny_model

LAYERS = (1, 2, 5)


@pytest.fixture
def setup(tmp_path):
    tokenizer = FakeTokenizer()
    episodes = made_up_episodes(tokenizer, 6) + made_up_episodes(tokenizer, 3, split="heldout", seed=1)
    for e in episodes[6:]:
        e["episode_id"] = "h" + e["episode_id"]
        e["qid"] = "h" + e["qid"]
    model = tiny_model()
    index = cache_states(model, tokenizer, episodes, tmp_path, LAYERS, batch_size=4)
    return tokenizer, episodes, model, index, tmp_path


def _alone(model, tokenizer, episode):
    ids = tokenizer(episode["prompt"])["input_ids"]
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), output_hidden_states=True)
    return ids, torch.stack([out.hidden_states[l + 1][0] for l in LAYERS], dim=1)    # T, L, d


def test_saved_rows_run_from_the_window_start_through_the_last_token(setup):
    tokenizer, episodes, model, index, out = setup
    for e in episodes:
        ids, states = _alone(model, tokenizer, e)
        window_start, tail_start = spans(ids, tokenizer)
        shard, offset, length = index[e["episode_id"]]
        assert length == len(ids) - window_start
        assert length - TAIL_LENGTH == tail_start - window_start
        saved = cache.episode_rows(out, e["episode_id"])
        # a row in a left-padded batch matches the row alone up to float32 rounding
        torch.testing.assert_close(saved, states[window_start:], rtol=1e-4, atol=1e-5)


def test_load_states_right_aligns_the_window_at_each_read_position(setup):
    tokenizer, episodes, model, index, out = setup
    ids_list = [e["episode_id"] for e in episodes]
    for k in range(TAIL_LENGTH - 1):
        h, mask = load_states(out, ids_list, k)
        assert h.shape[:2] == (len(episodes), len(LAYERS)) and h.shape[2] == mask.shape[1]
        for b, e in enumerate(episodes):
            saved = cache.episode_rows(out, e["episode_id"])
            content = saved.shape[0] - TAIL_LENGTH
            w = content + k + 1
            assert int(mask[b].sum()) == w and bool(mask[b, -w:].all())
            assert torch.equal(h[b, :, -w:], saved[:w].transpose(0, 1))
            assert torch.equal(h[b, :, :-w], torch.zeros_like(h[b, :, :-w]))
            # the read token is tail[k]
            assert torch.equal(h[b, :, -1], saved[content + k])


def test_bank_gives_what_load_states_gives(setup):
    _, episodes, _, _, out = setup
    ids_list = [e["episode_id"] for e in episodes]
    bank = StateBank.from_cache(out, ids_list)
    for k in (0, 3, TAIL_LENGTH - 2):
        h, mask = load_states(out, ids_list[::-1], k)
        bh, bmask = bank.window(ids_list[::-1], k)
        assert torch.equal(h, bh) and torch.equal(mask, bmask)
        sh, _ = bank.window(ids_list, k, layer_index=[2, 0])
        assert torch.equal(sh, bank.window(ids_list, k)[0][:, [2, 0]])


def test_the_last_tail_token_is_never_a_read_position(setup):
    _, episodes, _, _, out = setup
    with pytest.raises(ValueError):
        load_states(out, [episodes[0]["episode_id"]], TAIL_LENGTH - 1)


def test_saved_states_equal_a_fresh_pass_of_the_same_batch(setup):
    tokenizer, episodes, model, index, out = setup
    meta = read_meta(out)
    by_id = {e["episode_id"]: e for e in episodes}
    for batch in meta["batches"]:
        items = cache.batch_order(cache.tokenize(tokenizer, [by_id[i] for i in batch]))
        fresh = cache.saved_states(model, items, meta["layers"], tokenizer.pad_token_id)
        for it, rows in zip(items, fresh):
            assert torch.equal(rows, cache.episode_rows(out, it["episode_id"]))


def test_shards_and_offsets(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "SHARD_SIZE", 4)
    tokenizer = FakeTokenizer()
    episodes = made_up_episodes(tokenizer, 10)
    index = cache_states(tiny_model(), tokenizer, episodes, tmp_path, LAYERS, batch_size=2)
    assert sorted({v[0] for v in index.values()}) == [0, 1, 2]
    assert read_meta(tmp_path)["shards"] == 3
    for shard in range(3):
        rows = sorted((v[1], v[2]) for v in index.values() if v[0] == shard)
        positions = [0]
        for offset, length in rows:
            assert offset == positions[-1]
            positions.append(offset + length)
        assert torch.load(tmp_path / cache.shard_name(shard)).shape[0] == positions[-1]
    assert json.loads((tmp_path / cache.INDEX_FILE).read_text()) == index


def test_no_test_state_is_written(tmp_path):
    tokenizer = FakeTokenizer()
    episodes = made_up_episodes(tokenizer, 2) + made_up_episodes(tokenizer, 1, split="test", seed=3)
    episodes[-1]["episode_id"] = "t/0"
    with pytest.raises(ValueError, match="test"):
        cache_states(tiny_model(), tokenizer, episodes, tmp_path / "c", LAYERS, batch_size=2)
    assert not (tmp_path / "c").exists()
