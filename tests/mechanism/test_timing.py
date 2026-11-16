"""The timing pass on the tiny model: the batch, the same work in every arm, the profile and the settings."""

from __future__ import annotations

import json

import pytest
from transformers import StaticCache

from check2.data import paths
from check2.mechanism.generate import REPLY_CAPS
from check2.mechanism.timing import __main__ as timing
from check2.mechanism.timing import batch as timing_batch
from check2.mechanism.timing.batch import BATCH_SIZE, FORMAT, first_batch, load, make_pipeline
from check2.mechanism.timing.settings import setting_on
from tests.tiny_model import TINY_FULL_ATTENTION_LAYERS, TINY_NUM_LAYERS

# The tiny model rarely ends a reply, so its tests generate to the shorter cap of the snap format.
CAP = REPLY_CAPS["snap"]


@pytest.fixture(autouse=True)
def short_cap(monkeypatch):
    monkeypatch.setattr(timing_batch, "CAP", CAP)
    monkeypatch.setattr(timing, "CAP", CAP)


def test_the_batch_is_the_first_96_short_heldout_episodes_by_prompt_length_then_id():
    batch = first_batch()
    keys = [(e["n_prompt_tokens"], e["episode_id"]) for e in batch]
    assert len(batch) == BATCH_SIZE and all(e["format"] == FORMAT for e in batch)
    assert keys == sorted(keys) and len(set(keys)) == len(keys)
    with (paths.EPISODES / "heldout.jsonl").open(encoding="utf-8") as f:
        others = [json.loads(line) for line in f if line.strip()]
    chosen = {e["episode_id"] for e in batch}
    assert all((e["n_prompt_tokens"], e["episode_id"]) > keys[-1]
               for e in others if e["format"] == FORMAT and e["episode_id"] not in chosen)
    assert first_batch(3) == batch[:3]


def test_every_arm_generates_the_same_tokens_and_records_three_timed_repeats():
    digests = set()
    for implementation in ("none", "hooks", "wrappers"):
        values = timing.run_generation("tiny", implementation, 2)
        assert len(values["seconds"]) == 3 and values["repeats_identical"]
        assert values["spread"] == (max(values["seconds"]) - min(values["seconds"])) / min(values["seconds"])
        digests.add(values["replies_sha256"])
    assert len(digests) == 1


def test_the_profile_marks_every_mixer_once_per_step():
    values = timing.run_profile("tiny", 2)
    parts = values["parts"]
    assert parts["full_attention_layers"]["calls"] == len(TINY_FULL_ATTENTION_LAYERS)
    assert parts["deltanet_layers"]["calls"] == TINY_NUM_LAYERS - len(TINY_FULL_ATTENTION_LAYERS)
    assert parts["matrix_multiplies"]["calls"] > 0 and parts["cache_handling"]["calls"] > 0
    assert parts["hooks"]["calls"] == 0 and values["wall_seconds"] > 0


@pytest.mark.parametrize("setting", ["preallocated_cache", "cublaslt"])
def test_a_setting_is_timed_on_and_off_with_its_count_of_changed_answers(setting):
    values = timing.run_setting("tiny", setting, 2)
    assert "error" not in values, values.get("error")
    assert len(values["off"]["seconds"]) == len(values["on"]["seconds"]) == 3
    assert 0 <= values["final_answers_differing"] == len(values["differing_episodes"]) <= values["episodes"]


def test_the_preallocated_cache_is_the_one_the_pipeline_continues_from():
    model, tokenizer, layout = load("tiny")
    batch = first_batch(2)
    pipeline = make_pipeline(model, tokenizer, layout, "none")
    try:
        with setting_on("preallocated_cache", pipeline, batch, CAP):
            prefilled = pipeline.prefill(batch)
            assert isinstance(prefilled.cache, StaticCache)
            replies = pipeline.generate(prefilled, None, CAP)
        assert not isinstance(pipeline.prefill(batch).cache, StaticCache)
    finally:
        pipeline.close()
    assert all(len(r.token_ids) == CAP or r.ended for r in replies)
