"""Sampling: what is sampled in what order, the reply's end, left padding, seeding and resumption."""

import pytest
import torch

from check2.data import sampling
from check2.data.sampling import PlanMismatch, generate_samples, left_padded, parse_budgets, \
    plan_batches, read_jsonl, split_reply, training_episodes, write_jsonl
from tests.tiny_model import tiny_model


def ep(eid: str, fmt: str, n: int, form: str = "named", situation: str = "hold", split: str = "train",
       reason: str | None = None) -> dict:
    return {"episode_id": eid, "format": fmt, "form": form, "situation": situation, "split": split,
            "reason": reason, "n_prompt_tokens": n,
            "prompt": eid}


def test_split_reply_stops_at_the_first_end_of_turn():
    assert split_reply([5, 6, 9, 7, 9], 9) == ([5, 6], True)
    assert split_reply([9, 1], 9) == ([], True)
    assert split_reply([5, 6, 7], 9) == ([5, 6, 7], False)


def test_left_padding_puts_the_padding_before_each_prompt():
    ids, mask = left_padded([[5, 6, 7], [8]], pad_id=0)
    assert ids.tolist() == [[5, 6, 7], [0, 0, 8]]
    assert mask.tolist() == [[1, 1, 1], [0, 0, 1]]


def test_batches_take_each_format_in_turn_sorted_by_length_then_id():
    episodes = [ep("e", "short", 5), ep("a", "snap", 9), ep("b", "snap", 3), ep("c", "snap", 3),
                ep("d", "short", 1)]
    batches = plan_batches(episodes, batch_size=2)
    assert [[e["episode_id"] for e in b] for b in batches] == [["b", "c"], ["a"], ["d", "e"]]
    limited = plan_batches(episodes, batch_size=2, limit=1)
    assert [[e["episode_id"] for e in b] for b in limited] == [["b"], ["d"]]


def test_a_token_budget_closes_a_batch_before_episodes_times_longest_prompt_exceed_it():
    episodes = [ep(f"s{n}", "snap", n) for n in (10, 10, 10, 20, 20, 30)]
    batches = plan_batches(episodes, batch_size=3, batch_tokens={"snap": 40})
    # 3 x 10 fits; 2 x 20 fits; 30 alone.
    assert [[e["episode_id"] for e in b] for b in batches] == [["s10", "s10", "s10"], ["s20", "s20"], ["s30"]]
    assert all(len(b) * max(e["n_prompt_tokens"] for e in b) <= 40 for b in batches)
    assert plan_batches(episodes, 3) != batches
    assert plan_batches(episodes, 3, batch_tokens={"short": 40}) == plan_batches(episodes, 3)
    assert parse_budgets("snap=8832,short=5000") == {"snap": 8832, "short": 5000}


def test_only_named_and_reasoned_train_episodes_are_sampled(tmp_path):
    path = tmp_path / "train.jsonl"
    rows = [ep("a", "snap", 3, form="bare"), ep("b", "snap", 3), ep("c", "short", 3, form="reasoned", reason="r"),
            ep("d", "short", 3, situation="recheck")]
    write_jsonl(path, rows)
    assert [e["episode_id"] for e in training_episodes(path)] == ["b", "c", "d"]
    assert [e["episode_id"] for e in training_episodes(path, cells={("recheck", "short")})] == ["d"]


def test_a_reasoned_episode_without_a_reason_or_an_episode_outside_train_is_refused(tmp_path):
    path = tmp_path / "train.jsonl"
    write_jsonl(path, [ep("a", "snap", 3, form="reasoned", reason="")])
    with pytest.raises(ValueError, match="no reason"):
        training_episodes(path)
    write_jsonl(path, [ep("a", "snap", 3, split="heldout")])
    with pytest.raises(ValueError, match="only train"):
        training_episodes(path)


def test_samples_on_the_tiny_model_end_inside_the_cap_and_repeat_under_their_seed():
    model = tiny_model()
    prompts = [[5, 6, 7, 8, 9], [10, 11]]
    end_id, cap = 3, 12
    first = generate_samples(model, prompts, cap, n_samples=4, seed=7, end_id=end_id, pad_id=0)
    assert len(first) == 2 and all(len(s) == 4 for s in first)
    for samples in first:
        for tokens, ended in samples:
            assert end_id not in tokens
            assert (len(tokens) < cap) if ended else (len(tokens) == cap)
    assert generate_samples(model, prompts, cap, 4, 7, end_id, 0) == first
    assert generate_samples(model, prompts, cap, 4, 8, end_id, 0) != first


def test_samples_follow_the_specified_sampler_settings(monkeypatch):
    seen = {}

    class Recorder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def generate(self, **kwargs):
            seen.update(kwargs)
            ids = kwargs["input_ids"].repeat_interleave(kwargs["num_return_sequences"], 0)
            return torch.cat([ids, torch.full((ids.shape[0], 2), 3)], dim=1)

    generate_samples(Recorder(), [[5, 6], [7]], cap=16, n_samples=4, seed=0, end_id=3, pad_id=0)
    assert (seen["temperature"], seen["top_p"], seen["top_k"]) == (0.7, 0.8, 20)
    assert seen["do_sample"] is True and seen["max_new_tokens"] == 16
    assert seen["input_ids"].tolist() == [[5, 6], [0, 7]]


def fake_batch(calls):
    def sample_batch(model, tokenizer, batch, n_samples, seed, batch_index, end_id):
        calls.append(batch_index)
        return [{"episode_id": e["episode_id"], "format": e["format"], "form": e["form"],
                 "situation": e["situation"], "batch": batch_index, "seed": seed,
                 "samples": [{"text": "B", "ended": True, "n_tokens": 2}] * n_samples} for e in batch]
    return sample_batch


class FakeTokenizer:
    pad_token_id = 0

    def convert_tokens_to_ids(self, token):
        return 9


def test_a_run_started_again_keeps_its_complete_shards_and_joins_them_in_order(tmp_path, monkeypatch):
    batches = plan_batches([ep("a", "snap", 1), ep("b", "snap", 2), ep("c", "short", 1)], batch_size=1)
    out = tmp_path / "samples.jsonl.gz"
    calls = []
    monkeypatch.setattr(sampling, "sample_batch", fake_batch(calls))
    sampling.parts_dir(out).mkdir()
    done = fake_batch([])(None, None, batches[0], 4, 10, 0, 9)
    write_jsonl(sampling.shard_path(out, 0), done)
    sampling.seconds_path(out, 0).write_text("2.5\n", encoding="utf-8")
    timing = sampling.sample_all(None, FakeTokenizer(), batches, 4, 10, out)
    assert calls == [1, 2]
    lines = read_jsonl(out)
    assert [(x["episode_id"], x["seed"]) for x in lines] == [("a", 10), ("b", 11), ("c", 12)]
    assert timing["snap"]["episodes"] == 2 and timing["snap"]["replies"] == 8
    assert timing["snap"]["seconds"] >= 2.5
    assert not sampling.parts_dir(out).exists()


def test_a_shard_from_another_plan_is_refused(tmp_path, monkeypatch):
    batches = plan_batches([ep("a", "snap", 1), ep("b", "snap", 2)], batch_size=1)
    out = tmp_path / "samples.jsonl.gz"
    monkeypatch.setattr(sampling, "sample_batch", fake_batch([]))
    sampling.parts_dir(out).mkdir()
    write_jsonl(sampling.shard_path(out, 0), fake_batch([])(None, None, batches[1], 4, 0, 0, 9))
    sampling.seconds_path(out, 0).write_text("1\n", encoding="utf-8")
    with pytest.raises(PlanMismatch):
        sampling.sample_all(None, FakeTokenizer(), batches, 4, 0, out)


def test_the_samples_file_is_the_same_bytes_for_the_same_lines(tmp_path):
    rows = [{"episode_id": "a", "samples": [{"text": "é", "ended": True}]}]
    write_jsonl(tmp_path / "x.jsonl.gz", rows)
    write_jsonl(tmp_path / "y.jsonl.gz", rows)
    assert (tmp_path / "x.jsonl.gz").read_bytes() == (tmp_path / "y.jsonl.gz").read_bytes()
    assert read_jsonl(tmp_path / "x.jsonl.gz") == rows
