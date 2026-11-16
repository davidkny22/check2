"""What a run hands the model, on the tiny model: the appended read's arguments, and the System prompt run's prompts."""

from __future__ import annotations

import inspect
import json

import torch

from check2.config.schema import SYSTEM_PROMPT_SUFFIX
from check2.data.episodes import SYSTEM
from check2.evaluation.runs import output, run
from check2.evaluation.runs import episodes as run_episodes
from check2.evaluation.runs.parts import disabled_layers
from check2.mechanism.generate import Pipeline
from tests.evaluation.faults import fails_under

from .conftest import config


def _recorder(monkeypatch):
    """Record, per call of generation and of the appended read, the `p`, `alpha` and switched-off layers handed in."""
    calls: dict[str, list[dict]] = {"generate": [], "appended_read": []}
    for name in calls:
        real = getattr(Pipeline, name)
        signature = inspect.signature(real)

        def wrapper(*args, real=real, name=name, signature=signature, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            p = bound.arguments["p"]
            calls[name].append({"p": None if p is None else p.detach().cpu().clone(), "alpha": bound.arguments["alpha"],
                                "disabled": tuple(bound.arguments["disabled_layers"]),
                                "ids": [e["episode_id"] for e in bound.arguments["prefilled"].episodes]})
            return real(*args, **kwargs)
        monkeypatch.setattr(Pipeline, name, wrapper)
    return calls


def _go(bench, reader, folder):
    def go(cfg, name):
        return run(cfg, name, bench.model, bench.tokenizer, bench.layout, reader, episode_folder=folder)
    return go


def check_appended_read_arguments(go, layout, calls, changes, name, donors=None):
    """Every batch hands the appended read the same `p`, `alpha` and switched-off layers as generation, the run's own."""
    for kept in calls.values():
        kept.clear()
    cfg = config(**changes)
    folder = go(cfg, name)
    generated, read = calls["generate"], calls["appended_read"]
    assert [i for g in generated for i in g["ids"]] == [r["episode_id"] for r in output.read_records(folder)]
    assert len(read) == len(generated)
    for g, a in zip(generated, read):
        assert g["ids"] == a["ids"]
        assert g["alpha"] == a["alpha"] == cfg.writer.alpha
        assert g["disabled"] == a["disabled"] == disabled_layers(cfg, layout)
        assert torch.equal(g["p"], a["p"])
    if donors is not None:
        by_id = {r["episode_id"]: r for r in output.read_records(folder)}
        for a in read:
            for b, eid in enumerate(a["ids"]):
                record = by_id[eid]
                expected = donors[record["donor"]] if record["donor"] is not None else record["p"]
                assert torch.equal(a["p"][b], torch.tensor(expected, dtype=a["p"].dtype))


CONSUMER = {"adapter.kind": "consumer", "writer.alpha": 1.0}


def test_the_appended_read_gets_the_runs_own_p_alpha_and_band(bench, reader, results, episode_folder, checkpoints,
                                                             monkeypatch):
    calls = _recorder(monkeypatch)
    go = _go(bench, reader, episode_folder)
    check2 = output.read_records(go(config(**CONSUMER), "check2-heldout"))
    for changes, name in (({**CONSUMER, "writer.alpha": 0.0}, "off-heldout"),
                          ({**CONSUMER, "writer.alpha": 0.5}, "check2-alpha0.5-heldout"),
                          ({**CONSUMER, "writer.disabled_layers": (3, 6, 9)}, "check2-bands-3-6-9-off-heldout")):
        check_appended_read_arguments(go, bench.layout, calls, changes, name)
    check_appended_read_arguments(go, bench.layout, calls, {**CONSUMER, "signal": "swapped"}, "swap-heldout",
                                  donors={r["episode_id"]: r["p"] for r in check2})


def test_an_appended_read_at_full_strength_turns_the_check_red(bench, reader, results, episode_folder, checkpoints,
                                                               monkeypatch):
    calls = _recorder(monkeypatch)
    go = _go(bench, reader, episode_folder)
    fails_under("appended-read-at-full-strength", lambda: check_appended_read_arguments(
        go, bench.layout, calls, {**CONSUMER, "writer.alpha": 0.0}, "off-heldout"))


SYSTEM_TURN = f"<|im_start|>system\n{SYSTEM}{SYSTEM_PROMPT_SUFFIX}<|im_end|>\n"


def check_the_system_prompt_run(go, bench, monkeypatch, episode_folder, prefix):
    """Every prompt that reaches the prefill holds the suffix in its system turn, reads at the same tail token as the
    unchanged episode, and the run keeps the Original run's order."""
    seen: list[dict] = []
    real = Pipeline.prefill

    def prefill(self, episodes):
        seen.extend(episodes)
        return real(self, episodes)
    monkeypatch.setattr(Pipeline, "prefill", prefill)
    original = go(config(), f"{prefix}original-heldout")
    seen.clear()
    suffixed = go(config(**{"baseline.system_prompt_suffix": SYSTEM_PROMPT_SUFFIX}), f"{prefix}system-prompt-heldout")
    unchanged = {e["episode_id"]: e for e in run_episodes.select(config(), episode_folder)}
    assert sorted(e["episode_id"] for e in seen) == sorted(unchanged)
    for e in seen:
        before = unchanged[e["episode_id"]]
        assert e["prompt"].startswith(SYSTEM_TURN)
        assert e["prompt"] == before["prompt"].replace(f"{SYSTEM}<|im_end|>", f"{SYSTEM}{SYSTEM_PROMPT_SUFFIX}<|im_end|>", 1)
        ids, ids_before = bench.tokenizer(e["prompt"])["input_ids"], bench.tokenizer(before["prompt"])["input_ids"]
        assert ids[e["read_index"]] == ids_before[before["read_index"]]
        assert len(ids) - e["read_index"] == len(ids_before) - before["read_index"]
    meta = {name: json.loads((folder / "run.json").read_text(encoding="utf-8")) for name, folder in
            (("original", original), ("suffixed", suffixed))}
    assert meta["suffixed"]["order_sha256"] == meta["original"]["order_sha256"]


def test_the_system_prompt_run_carries_the_suffix_in_every_prompt(bench, reader, results, episode_folder,
                                                                  monkeypatch):
    check_the_system_prompt_run(_go(bench, reader, episode_folder), bench, monkeypatch, episode_folder, "")


def test_dropping_the_suffix_turns_the_check_red(bench, reader, results, episode_folder, monkeypatch):
    fails_under("system-prompt-suffix-dropped", lambda: check_the_system_prompt_run(
        _go(bench, reader, episode_folder), bench, monkeypatch, episode_folder, "planted-"))
