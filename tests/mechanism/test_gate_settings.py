"""The decode-setting switch on the four gates a setting must pass: what every pipeline call ran with."""

from __future__ import annotations

import contextlib
import json

import pytest
from transformers import StaticCache

from check2 import record
from check2.gates import cached_equals_uncached, gating, identity, padding_invariance
from check2.mechanism.generate import Pipeline
from check2.mechanism.timing import settings as decode_settings

GATES = {"identity": identity, "gating": gating, "padding_invariance": padding_invariance,
         "cached_equals_uncached": cached_equals_uncached}
REAL_SETTING_ON = decode_settings.setting_on


@pytest.fixture()
def calls(monkeypatch) -> list[dict]:
    """Every model call any pipeline makes: the settings on for that pipeline at the time, and the cache it got.

    `setting_on` is wrapped so that it records which pipeline it was entered for and under
    which name. The cuDNN attention backend cannot serve a call on this machine's CPU, so for
    it the wrapper records the switch and runs the call as it is; the other two run for real.
    """
    active: list[tuple[str, int]] = []
    seen: list[dict] = []
    real_run = Pipeline._run

    def run(self, ids, attention, read_column, cache, *args, **kwargs):
        seen.append({"settings": [s for s, p in active if p == id(self)], "cache": cache,
                     "use_cache": kwargs.get("use_cache", True)})
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

    monkeypatch.setattr(Pipeline, "_run", run)
    monkeypatch.setattr(decode_settings, "setting_on", setting_on)
    return seen


def _assert_every_call_ran_under(calls: list[dict], setting: str | None, gate: str) -> None:
    assert calls, "the gate made no model call"
    assert all(c["settings"] == ([setting] if setting else []) for c in calls)
    statics = [isinstance(c["cache"], StaticCache) for c in calls if c["use_cache"]]
    if setting == "preallocated_cache":
        assert all(statics)
    else:
        assert not any(statics)
    uncached = [c for c in calls if not c["use_cache"]]
    if gate == "cached_equals_uncached":
        assert uncached and all(c["cache"] is None for c in uncached)


@pytest.mark.parametrize("setting", [None, *decode_settings.SETTINGS])
@pytest.mark.parametrize("gate", sorted(GATES))
def test_the_command_line_setting_is_on_for_every_call_of_every_pipeline(gate, setting, calls, monkeypatch,
                                                                         tmp_path):
    monkeypatch.setattr(record, "RESULTS", str(tmp_path))
    argv = ["--model", "tiny", "--limit", "1"] + (["--setting", setting] if setting else [])
    code = GATES[gate].main(argv)
    _assert_every_call_ran_under(calls, setting, gate)
    values = json.loads(next(tmp_path.rglob("*.json")).read_text(encoding="utf-8"))["values"]
    if setting is None:
        assert "setting" not in values
    else:
        assert values["setting"] == setting
    assert code == 0, values


@pytest.mark.parametrize("gate", sorted(GATES))
def test_each_gate_is_green_under_the_preallocated_cache_and_red_with_its_planted_fault(gate, bench, episodes,
                                                                                        calls):
    module = GATES[gate]
    episodes = {fmt: chosen[:2] for fmt, chosen in episodes.items()}
    green = module.check(bench, episodes, setting="preallocated_cache")
    _assert_every_call_ran_under(calls, "preallocated_cache", gate)
    calls.clear()
    planted = module.check(bench, episodes, plant=True, setting="preallocated_cache")
    _assert_every_call_ran_under(calls, "preallocated_cache", gate)
    assert green["setting"] == planted["setting"] == "preallocated_cache"
    assert green["green"] is True, green
    assert planted["green"] is False
