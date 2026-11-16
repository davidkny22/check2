"""The optimiser and the learning-rate schedule: fused AdamW, linear warm-up over 50 steps, then cosine decay."""

from __future__ import annotations

import math

import pytest
import torch

from check2.training import schedule

TOTAL = 490  # two epochs of 245 steps


def formula(step: int, total: int, base: float) -> float:
    """Written from the sentence: linear warm-up over the first 50 steps, then cosine decay to zero at the end."""
    if step < 50:
        return base * step / 50
    return base * 0.5 * (1 + math.cos(math.pi * (step - 50) / (total - 50)))


def _rates_at(steps: list[int], total: int, groups: list[float]) -> dict[int, list[float]]:
    parameters = [torch.nn.Parameter(torch.zeros(2)) for _ in groups]
    optimizer = schedule.build_optimizer([([p], rate) for p, rate in zip(parameters, groups)])
    lr_schedule = schedule.build_schedule(optimizer, total)
    seen = {}
    for step in range(total):
        if step in steps:
            seen[step] = [g["lr"] for g in optimizer.param_groups]
        for p in parameters:
            p.grad = torch.ones_like(p)
        optimizer.step()
        lr_schedule.step()
    return seen


@pytest.mark.parametrize("groups", [[1e-3], [1e-4, 1e-4]], ids=["connector", "consumer"])
def test_the_learning_rate_at_steps_0_25_50_the_midpoint_and_the_last_equals_the_formula(groups):
    steps = [0, 25, 50, TOTAL // 2, TOTAL - 1]
    seen = _rates_at(steps, TOTAL, groups)
    for step in steps:
        for rate, base in zip(seen[step], groups):
            assert rate == pytest.approx(formula(step, TOTAL, base), rel=1e-12, abs=1e-18), step
    assert seen[0] == [0.0] * len(groups)
    assert seen[50] == pytest.approx(groups)


def test_the_optimiser_is_fused_adamw_over_float32_parameters_only():
    trained = torch.nn.Parameter(torch.zeros(3))
    optimizer = schedule.build_optimizer([([trained], 1e-3)])
    assert isinstance(optimizer, torch.optim.AdamW) and optimizer.defaults["fused"] is True
    with pytest.raises(ValueError):
        schedule.build_optimizer([([torch.nn.Parameter(torch.zeros(3, dtype=torch.bfloat16))], 1e-3)])
    frozen = torch.nn.Parameter(torch.zeros(3), requires_grad=False)
    with pytest.raises(ValueError):
        schedule.build_optimizer([([frozen], 1e-3)])
    with pytest.raises(ValueError):
        schedule.build_optimizer([([trained], 1e-3), ([trained], 1e-4)])


def test_clipping_is_at_norm_one_over_every_trained_parameter_together():
    a, b = torch.nn.Parameter(torch.zeros(1)), torch.nn.Parameter(torch.zeros(1))
    optimizer = schedule.build_optimizer([([a], 1e-3), ([b], 1e-4)])
    a.grad, b.grad = torch.tensor([3.0]), torch.tensor([4.0])
    norm = schedule.clip(optimizer)
    assert float(norm) == pytest.approx(5.0)
    assert float(a.grad) == pytest.approx(0.6, rel=1e-5) and float(b.grad) == pytest.approx(0.8, rel=1e-5)
