"""The writer and its hooks: zero at the start, the formula at every token, exactly off when switched off."""

from __future__ import annotations

import math

import torch

from check2.gates import mechanism_setup as setup
from check2.gates import write_formula
from check2.mechanism import generate as generate_module
from check2.mechanism import write as write_module
from check2.mechanism.control import ControlState
from check2.mechanism.generate import Pipeline, one_hot_situations
from check2.mechanism.write import Writers


def test_an_untrained_writer_writes_exactly_zero():
    writers = Writers((3, 6), hidden_size=40)
    vectors = writers(torch.softmax(torch.randn(5, 3), -1))
    assert set(vectors) == {3, 6}
    for v in vectors.values():
        assert v.shape == (5, 40) and v.dtype == torch.float32 and bool((v == 0).all())
    assert all(p.dtype == torch.float32 for p in writers.parameters())


def test_the_writer_is_its_formula():
    torch.manual_seed(1)
    writers = Writers((3,), hidden_size=8, width=4)
    with torch.no_grad():
        for parameter in writers.parameters():
            parameter.normal_()
    p = torch.tensor([[0.2, 0.5, 0.3]])
    a, b = writers.first["3"], writers.second["3"]
    u = p @ a.weight.t() + a.bias
    expected = (0.5 * u * (1 + torch.erf(u / math.sqrt(2)))) @ b.weight.t() + b.bias
    assert torch.allclose(writers(p)[3], expected, rtol=1e-6, atol=1e-7)


def _changes(bench, writers, batch, alpha, band):
    pipeline = Pipeline(bench.model, bench.tokenizer, writers.control, writers=writers.module)
    try:
        p = one_hot_situations(batch)
        with write_formula.write_spies(bench.model, bench.layout.write_layers) as seen:
            pipeline.generate(pipeline.prefill(batch), p, 2, alpha=alpha, disabled_layers=band)
    finally:
        pipeline.close()
    return {j: [(o - i) for i, o in zip(r["received"], r["passed"])] for j, r in seen.items()}


class _Random:
    def __init__(self, bench, episodes):
        self.control = ControlState()
        self.module, _ = setup.random_additions(bench, self.control, episodes["snap"], adapter_on=False)


def test_alpha_zero_and_each_band_give_exactly_zero_write(bench, episodes):
    writers = _Random(bench, episodes)
    batch = episodes["short"]
    for j, changes in _changes(bench, writers, batch, 0.0, ()).items():
        assert all(bool((c == 0).all()) for c in changes), j
    for band in bench.layout.bands:
        changes = _changes(bench, writers, batch, 1.0, band)
        for j, per_call in changes.items():
            later = per_call[1:]
            if j in band:
                assert all(bool((c == 0).all()) for c in per_call), (band, j)
            else:
                assert all(bool((c != 0).any()) for c in later), (band, j)


def test_the_formula_check_is_green_on_the_tiny_model(bench, episodes):
    values = write_formula.check(bench, episodes)
    assert values["green"], values


def test_the_formula_check_fails_at_the_wrong_strength(bench, episodes, monkeypatch):
    original = write_module._write_hook

    def doubled(control, layer):
        hook = original(control, layer)

        def wrong(module, args, kwargs):
            out = hook(module, args, kwargs)
            if out is None:
                return None
            (changed, *rest), kw = out
            return (changed + (changed - args[0]), *rest), kw
        setattr(wrong, write_module.WRITE_HOOK_TAG, layer)
        return wrong

    monkeypatch.setattr(write_module, "_write_hook", doubled)
    assert not write_formula.check(bench, episodes)["green"]


def test_the_formula_check_fails_with_a_hook_on_the_wrong_layer(bench, episodes, monkeypatch):
    original = generate_module.install_write_hooks
    monkeypatch.setattr(generate_module, "install_write_hooks",
                        lambda model, control, layers: original(model, control, tuple(layers) + (0,)))
    values = write_formula.check(bench, episodes)
    assert not values["hooks_on_exactly_the_write_layers"] and not values["green"]
