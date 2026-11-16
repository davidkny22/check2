"""The adapter by wrapper modules and by the library's position-gated adapter, against the adapter by hooks."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from check2.gates import adapter_formula, gating
from check2.gates import mechanism_setup as setup
from check2.mechanism.adapter.implementations import IMPLEMENTATIONS, copy_in
from check2.mechanism.adapter.wrappers import WrapperAdapter
from check2.mechanism.control import ControlState
from check2.mechanism.generate import Pipeline
from check2.mechanism.model_parts import text_model

CANDIDATES = ["wrappers", pytest.param("library", marks=pytest.mark.skipif(
    importlib.util.find_spec("peft") is None, reason="peft is not installed here"))]


def _random_hook_adapter(bench, episodes) -> "tuple[ControlState, object]":
    control = ControlState()
    _, adapter = setup.random_additions(bench, control, episodes["short"], writers_on=False)
    return control, adapter


def _outputs_of_listed_modules(bench, adapter, episodes, names=None) -> dict[str, list[torch.Tensor]]:
    """Every listed module's output, per call, over a prompt pass and a few decoding steps of each format.

    With `adapter` None, the same modules' outputs with no adapter installed.
    """
    body = text_model(bench.model)
    seen: dict[str, list[torch.Tensor]] = {name: [] for name in (names or [n for n, _ in adapter.modules])}
    control = ControlState() if adapter is None else adapter.control
    pipeline = Pipeline(bench.model, bench.tokenizer, control, adapter=adapter)
    handles = []
    for name in seen:
        module = body.get_submodule(name)
        handles.append(module.register_forward_hook(lambda m, i, o, name=name: seen[name].append(o.detach().clone())))
    try:
        for batch in episodes.values():
            pipeline.generate(pipeline.prefill(batch), None, 4)
    finally:
        for handle in handles:
            handle.remove()
        pipeline.close()
    return seen


def _model_state(bench) -> tuple:
    return (list(bench.model.named_modules()), sorted(vars(bench.model)),
            [(n, p.data_ptr()) for n, p in bench.model.named_parameters()])


@pytest.mark.parametrize("implementation", CANDIDATES)
def test_the_candidate_output_equals_the_hook_output_on_every_listed_module(bench, episodes, implementation):
    _, hooks = _random_hook_adapter(bench, episodes)
    candidate = copy_in(hooks, implementation, bench.model)
    by_hooks = _outputs_of_listed_modules(bench, hooks, episodes)
    by_candidate = _outputs_of_listed_modules(bench, candidate, episodes)
    assert by_hooks.keys() == by_candidate.keys()
    for name in by_hooks:
        assert len(by_hooks[name]) == len(by_candidate[name]) > 0
        for a, b in zip(by_hooks[name], by_candidate[name]):
            assert torch.allclose(a, b, rtol=1e-6, atol=1e-6), name
    # The update is not zero on any listed module, so the comparison could fail.
    plain = _outputs_of_listed_modules(bench, None, episodes, names=list(by_hooks))
    assert all(any(not torch.equal(a, b) for a, b in zip(plain[name], by_hooks[name])) for name in by_hooks)


def test_the_wrapper_output_is_exactly_the_hook_output(bench, episodes):
    _, hooks = _random_hook_adapter(bench, episodes)
    by_hooks = _outputs_of_listed_modules(bench, hooks, episodes)
    by_wrappers = _outputs_of_listed_modules(bench, copy_in(hooks, "wrappers", bench.model), episodes)
    assert all(torch.equal(a, b) for name in by_hooks for a, b in zip(by_hooks[name], by_wrappers[name]))


@pytest.mark.parametrize("implementation", CANDIDATES)
def test_remove_restores_the_model_exactly(bench, episodes, implementation):
    _, hooks = _random_hook_adapter(bench, episodes)
    before = _model_state(bench)
    plain = Pipeline(bench.model, bench.tokenizer, ControlState())
    logits_before = plain.prompt_pass(plain.prefill(episodes["short"]), None)
    candidate = copy_in(hooks, implementation, bench.model)
    candidate.install()
    assert _model_state(bench)[0] != before[0]
    candidate.remove()
    assert _model_state(bench) == before
    assert torch.equal(plain.prompt_pass(plain.prefill(episodes["short"]), None), logits_before)


@pytest.mark.parametrize("implementation", CANDIDATES)
def test_the_adapted_modules_are_the_same_list_as_the_hook_adapter(bench, implementation):
    hooks = IMPLEMENTATIONS["hooks"](bench.model, ControlState(), 4, 8, bench.layout.adapter_first,
                                     bench.layout.adapter_last)
    candidate = IMPLEMENTATIONS[implementation](bench.model, ControlState(), 4, 8, bench.layout.adapter_first,
                                                bench.layout.adapter_last)
    assert [name for name, _ in candidate.modules] == [name for name, _ in hooks.modules]
    assert all(a is b for (_, a), (_, b) in zip(candidate.modules, hooks.modules))


@pytest.mark.parametrize("implementation", ["hooks", *CANDIDATES])
def test_gradients_reach_every_factor(bench, episodes, implementation):
    control, hooks = _random_hook_adapter(bench, episodes)
    candidate = copy_in(hooks, implementation, bench.model)
    pipeline = Pipeline(bench.model, bench.tokenizer, control, adapter=candidate)
    try:
        batch = episodes["snap"]
        with torch.inference_mode():
            prefilled = pipeline.prefill(batch)
            replies = [r.token_ids for r in pipeline.generate(prefilled, None, 4)]
        adapter_formula.reply_loss_backward(bench.model, control, prefilled, replies, pipeline.pad_id)
    finally:
        pipeline.close()
    for name, (a, b) in candidate.factors().items():
        assert a.grad is not None and bool(a.grad.abs().sum() > 0), name
        assert b.grad is not None and bool(b.grad.abs().sum() > 0), name


@pytest.mark.parametrize("implementation", ["hooks", *CANDIDATES])
def test_adapter_matches_its_formula_and_is_red_with_one_module_left_out(bench, episodes, implementation):
    values = adapter_formula.check(bench, episodes, implementation)
    assert values["green"], values
    planted = adapter_formula.check(bench, episodes, implementation, plant=True)
    assert not planted["green"]
    for result in planted["formats"].values():
        left_out = f"layers.{bench.layout.adapter_last}.mlp.down_proj"
        assert result["missing_from_observed"] == [left_out]
        assert result["max_relative_output_difference"] > adapter_formula.TOLERANCE
        assert result["max_relative_gradient_difference"] > adapter_formula.TOLERANCE


@pytest.mark.parametrize("implementation", ["hooks", "wrappers"])
def test_adapter_matches_its_formula_is_red_with_every_adapter_s_scale_halved(bench, episodes, implementation):
    """The fault sits in the code every implementation shares, so the adapter the gate draws its factors
    through carries it too; the gate sees it because it takes the scale from the settings.

    The library computes its own scale from the settings and refuses to install under this fault.
    """
    values = adapter_formula.check(bench, episodes, implementation, plant_scale=True)
    assert not values["green"] and values["planted_faults"] == ["scale_halved"]
    for result in values["formats"].values():
        assert result["declared_set_matches"] and result["observed_set_matches"]
        assert result["max_relative_output_difference"] > adapter_formula.TOLERANCE
        assert result["max_relative_gradient_difference"] > adapter_formula.TOLERANCE
    restored = IMPLEMENTATIONS[implementation](bench.model, ControlState(), 4, 8, bench.layout.adapter_first,
                                               bench.layout.adapter_last)
    assert restored.scale == 8 / 4


def test_the_formula_gate_sees_a_module_adapted_that_is_not_listed(bench, episodes, monkeypatch):
    """A candidate that adapts one projection beyond the list is red, and the extra module is named."""
    original = WrapperAdapter.__init__
    extra = f"layers.{bench.layout.adapter_first - 1}.mlp.up_proj"

    def adapting_one_more(self, model, *args, **kwargs):
        original(self, model, *args, **kwargs)
        module = text_model(model).get_submodule(extra)
        self.modules = [(extra, module)] + self.modules
        self.a[extra] = torch.nn.Parameter(torch.randn(self.rank, module.in_features))
        self.b[extra] = torch.nn.Parameter(torch.randn(module.out_features, self.rank))
    monkeypatch.setattr(WrapperAdapter, "__init__", adapting_one_more)
    def load_listed(self, factors):
        with torch.no_grad():
            for name, (a, b) in factors.items():
                self.a[name].copy_(a)
                self.b[name].copy_(b)
    monkeypatch.setattr(WrapperAdapter, "load_factors", load_listed)
    values = adapter_formula.check(bench, episodes, "wrappers")
    assert not values["green"]
    assert all(r["extra_in_observed"] == [extra] for r in values["formats"].values())


@pytest.mark.parametrize("implementation", CANDIDATES)
def test_the_gate_switch_reaches_the_candidate(bench, episodes, implementation, monkeypatch):
    values = gating.check(bench, episodes, implementation=implementation)
    assert values["green"] and values["implementation"] == implementation
    monkeypatch.setattr(IMPLEMENTATIONS[implementation], "install", lambda self: None)
    assert not gating.check(bench, episodes, implementation=implementation)["second_half_green"]


@pytest.mark.parametrize("implementation", CANDIDATES)
def test_a_right_padded_whole_sequence_call_is_checked_at_every_position(bench, episodes, implementation):
    """Replies of different lengths leave padding after the shorter rows' last tokens.

    The wrappers match the formula everywhere. The library matches it at every real token
    and differs at that padding, where it adds the update and the mask does not.
    """
    control, hooks = _random_hook_adapter(bench, episodes)
    candidate = copy_in(hooks, implementation, bench.model)
    factors = {name: (a.detach().clone(), b.detach().clone()) for name, (a, b) in hooks.factors().items()}
    probe = adapter_formula.OutputProbe(bench.model, control, factors, hooks.scale)
    pipeline = Pipeline(bench.model, bench.tokenizer, control, adapter=candidate)
    try:
        with torch.inference_mode():
            prefilled = pipeline.prefill(episodes["snap"])
        replies = [[5 + i] * (2 + 3 * i) for i in range(len(episodes["snap"]))]
        with probe.watching():
            adapter_formula.whole_sequence_outputs(bench.model, control, prefilled, replies, pipeline.pad_id, probe)
    finally:
        pipeline.close()
    assert probe.worst["real"][0] <= adapter_formula.TOLERANCE
    if implementation == "library":
        assert probe.worst["all"][0] > adapter_formula.TOLERANCE
    else:
        assert probe.worst["all"][0] <= adapter_formula.TOLERANCE
