"""A run executes with the decode settings its config names on, for generation and for the appended read."""

from __future__ import annotations

import contextlib

import pytest
import torch
from transformers import StaticCache

from check2.evaluation import steering
from check2.evaluation.runs import output, run
from check2.evaluation.runs.parts import check_executable
from check2.mechanism.generate import Pipeline
from check2.mechanism.timing import settings as decode_settings

from .conftest import SMALL_BATCH, config

REAL_SETTING_ON = decode_settings.setting_on
TRAINED = {"adapter.kind": "consumer", "writer.alpha": 1.0, "adapter.implementation": "hooks"}
PREALLOCATED = {"generation_cache": "preallocated"}
CUDNN = {"attention_backend": "cudnn"}
BATCHES = 10
TINY_STEERING_LAYER = 4
# The record fields computed from the call over 0..r alone, and how far float32 rounding moves them here.
ROUNDED = ("p", "p_used")
P_ROUNDING = 1e-6


@pytest.fixture()
def calls(monkeypatch) -> list[dict]:
    """Every model call any pipeline makes: the settings on for that pipeline, the cache it got, and its kind.

    `setting_on` is wrapped so that it records the pipeline and the name it was entered with.
    The cuDNN attention backend cannot serve a call on this machine's CPU, so for it the
    wrapper records the switch and runs the call as it is; the preallocated cache runs for real.
    """
    active: list[tuple[str, int]] = []
    seen: list[dict] = []
    real_run = Pipeline._run

    def run_call(self, ids, attention, read_column, cache, *args, **kwargs):
        use_cache = kwargs.get("use_cache", True)
        seen.append({"settings": sorted(s for s, p in active if p == id(self)), "cache": cache,
                     "use_cache": use_cache, "over_prefix": use_cache and ids.shape[1] == attention.shape[1]})
        return real_run(self, ids, attention, read_column, cache, *args, **kwargs)

    @contextlib.contextmanager
    def setting_on(setting, pipeline, batch, cap):
        active.append((setting, id(pipeline)))
        try:
            if setting == "cudnn_attention":
                yield decode_settings.Room(0)
            else:
                with REAL_SETTING_ON(setting, pipeline, batch, cap) as room:
                    yield room
        finally:
            active.remove((setting, id(pipeline)))

    monkeypatch.setattr(Pipeline, "_run", run_call)
    monkeypatch.setattr(decode_settings, "setting_on", setting_on)
    return seen


def _run(bench, reader, cfg, run_id, folder):
    return run(cfg, run_id, bench.model, bench.tokenizer, bench.layout, reader, None, episode_folder=folder)


def test_the_runner_no_longer_refuses_the_two_settings():
    for changes in (PREALLOCATED, CUDNN, {**PREALLOCATED, **CUDNN}):
        check_executable(config(**TRAINED, **changes), None)


def test_a_preallocated_cache_run_runs_every_call_on_it_and_gives_the_growing_run_s_records(
        bench, reader, results, episode_folder, checkpoints, calls):
    growing = _run(bench, reader, config(**TRAINED), "growing", episode_folder)
    assert calls and all(c["settings"] == [] for c in calls)
    assert not any(isinstance(c["cache"], StaticCache) for c in calls)
    assert sum(c["over_prefix"] for c in calls) == BATCHES
    calls.clear()
    preallocated = _run(bench, reader, config(**TRAINED, **PREALLOCATED), "preallocated", episode_folder)
    assert calls and all(c["settings"] == ["preallocated_cache"] for c in calls)
    assert all(isinstance(c["cache"], StaticCache) for c in calls if c["use_cache"])
    # Each batch: the call over 0..r for generation, and the appended read's own.
    assert sum(c["over_prefix"] for c in calls) == 2 * BATCHES
    # The static cache changes the sums of the call over 0..r in the last bit, so the reader's
    # p may move by float32 rounding; every reply, answer and appended read is the same.
    on, off = output.read_records(preallocated), output.read_records(growing)
    assert len(on) == len(off) == 2 * BATCHES
    for a, b in zip(on, off):
        assert {k: v for k, v in a.items() if k not in ROUNDED} == {k: v for k, v in b.items() if k not in ROUNDED}
        for key in ROUNDED:
            assert max(abs(x - y) for x, y in zip(a[key], b[key])) <= P_ROUNDING


def test_the_cudnn_backend_alone_and_with_the_preallocated_cache_is_on_for_every_call(
        bench, reader, results, episode_folder, calls):
    _run(bench, reader, config(**CUDNN), "cudnn", episode_folder)
    assert calls and all(c["settings"] == ["cudnn_attention"] for c in calls)
    assert not any(isinstance(c["cache"], StaticCache) for c in calls)
    calls.clear()
    _run(bench, reader, config(**CUDNN, **PREALLOCATED), "both", episode_folder)
    assert calls and all(c["settings"] == ["cudnn_attention", "preallocated_cache"] for c in calls)
    assert all(isinstance(c["cache"], StaticCache) for c in calls if c["use_cache"])


def test_the_train_pass_under_the_preallocated_cache_saves_the_generation_call_s_read_states(
        bench, reader, results, episode_folder, tmp_path, monkeypatch, calls):
    monkeypatch.setattr(steering, "STEERING_LAYER", TINY_STEERING_LAYER)
    monkeypatch.setattr(steering, "STATES_ROOT", tmp_path / "states")
    (episode_folder / "train.jsonl").write_text((episode_folder / "heldout.jsonl").read_text(encoding="utf-8"),
                                                encoding="utf-8")
    train = {"data.eval_split": "train", "data.forms": ("named", "reasoned")}
    _run(bench, reader, config(**train), "pass-growing", episode_folder)
    calls.clear()
    _run(bench, reader, config(**train, **PREALLOCATED), "pass-preallocated", episode_folder)
    assert calls and all(isinstance(c["cache"], StaticCache) for c in calls if c["use_cache"])
    batches = len(steering.load_states("pass-preallocated")) // SMALL_BATCH
    assert sum(c["over_prefix"] for c in calls) == 2 * batches
    growing, preallocated = steering.load_states("pass-growing"), steering.load_states("pass-preallocated")
    assert growing and set(preallocated) == set(growing)
    for k in growing:
        torch.testing.assert_close(preallocated[k], growing[k], rtol=0, atol=P_ROUNDING)
