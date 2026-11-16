"""The adapter by hooks: the listed modules, the update's formula, and its gate at the read token."""

from __future__ import annotations

import math

import pytest
import torch
from transformers import AutoConfig, Qwen3_5ForCausalLM

from check2.data.tokenizer import tokenizer_folder
from check2.gates import gating, identity
from check2.gates import mechanism_setup as setup
from check2.mechanism import generate as generate_module
from check2.mechanism.adapter.base import adapted_modules
from check2.mechanism.adapter.hooks import HookAdapter
from check2.mechanism.control import ControlState


def test_the_full_model_has_182_adapted_projections_in_29_layers():
    config = AutoConfig.from_pretrained(str(tokenizer_folder())).text_config
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)
    names = [name for name, _ in adapted_modules(model, 3, 31)]
    layers = sorted({int(name.split(".")[1]) for name in names})
    assert layers == list(range(3, 32))
    full = [i for i, kind in enumerate(config.layer_types) if kind == "full_attention" and i >= 3]
    for index in layers:
        count = sum(name.startswith(f"layers.{index}.") for name in names)
        assert count == (7 if index in full else 6), index
    assert len(names) == 8 * 7 + 21 * 6 == 182
    assert len(set(names)) == len(names)


def test_factors_start_kaiming_uniform_and_zero(bench):
    adapter = HookAdapter(bench.model, ControlState(), 32, 64, 1, 7)
    for name, module in adapter.modules:
        a, b = adapter.a[name], adapter.b[name]
        assert a.dtype == b.dtype == torch.float32
        assert a.shape == (32, module.in_features) and b.shape == (module.out_features, 32)
        assert bool((b == 0).all())
        assert float(a.abs().max()) <= 1 / math.sqrt(module.in_features) + 1e-7
    assert len(adapter.parameters()) == 2 * len(adapter.modules)


def test_the_hook_adds_the_masked_update_and_nothing_else(bench):
    control = ControlState()
    adapter = HookAdapter(bench.model, control, 4, 8, 3, 3)
    name, module = adapter.modules[0]
    torch.manual_seed(3)
    with torch.no_grad():
        adapter.b[name].normal_()
    x = torch.randn(2, 5, module.in_features)
    mask = torch.tensor([[0, 0, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=torch.float32)
    expected = x @ module.weight.t() + mask.unsqueeze(-1) * (8 / 4) * (x @ adapter.a[name].t() @ adapter.b[name].t())
    adapter.install()
    try:
        control.set(mask, adapter_active=True)
        got = module(x)
        control.set(mask, adapter_active=False)
        off = module(x)
    finally:
        control.clear()
        adapter.remove()
    plain = module(x)
    assert torch.allclose(got, expected, rtol=1e-5, atol=1e-6)
    assert torch.equal(off, plain)


def test_saved_factors_load_back(bench):
    first = HookAdapter(bench.model, ControlState(), 4, 8, 1, 2)
    with torch.no_grad():
        for p in first.parameters():
            p.normal_()
    second = HookAdapter(bench.model, ControlState(), 4, 8, 1, 2)
    second.load_factors({k: (a.clone(), b.clone()) for k, (a, b) in first.factors().items()})
    assert all(torch.equal(x, y) for x, y in zip(first.parameters(), second.parameters()))
    with pytest.raises(KeyError):
        second.load_factors({})


def test_identity_is_green_and_red_under_its_fault(bench, episodes):
    assert identity.check(bench, episodes)["green"]
    assert not identity.check(bench, episodes, plant=True)["green"]


def test_gating_is_green_and_red_under_its_fault(bench, episodes):
    values = gating.check(bench, episodes)
    assert values["green"], values
    planted = gating.check(bench, episodes, plant=True)
    assert not planted["first_half_green"]


def test_gating_second_half_is_red_without_the_write_hooks(bench, episodes, monkeypatch):
    monkeypatch.setattr(generate_module, "install_write_hooks", lambda model, control, layers: [])
    values = gating.check(bench, episodes)
    assert not values["second_half_green"]
    assert all(v["write_alone_changes_layer_3_at_r_plus_1"] == 0 for v in values["formats"].values())


def test_gating_second_half_is_red_without_the_adapter_hooks(bench, episodes, monkeypatch):
    monkeypatch.setattr(HookAdapter, "install", lambda self: None)
    values = gating.check(bench, episodes)
    assert not values["second_half_green"]
    assert all(v["adapter_alone_changes_layer_3_at_r_plus_1"] == 0 for v in values["formats"].values())


def test_the_random_adapter_update_is_half_the_module_output(bench, episodes):
    control = ControlState()
    _, adapter = setup.random_additions(bench, control, episodes["snap"], writers_on=False)
    name, module = adapter.modules[-1]
    outs, updates = [], []

    def probe(m, inputs, output):
        outs.append(output.norm(dim=-1).flatten())
        updates.append(adapter.update(name, inputs[0]).norm(dim=-1).flatten())
    handle = module.register_forward_hook(probe)
    try:
        pipeline = setup.plain_pipeline(bench)
        for e in episodes["snap"][:setup.MEASURE_EPISODES]:
            pipeline.prompt_pass(pipeline.prefill([e]), None)
    finally:
        handle.remove()
    ratio = float(torch.cat(updates).mean() / torch.cat(outs).mean())
    assert abs(ratio - setup.ADDITION_FRACTION) < 1e-4
