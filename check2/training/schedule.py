"""The optimiser and the learning-rate schedule every training uses.

The optimiser is PyTorch's AdamW with `fused=True` over the trained parameters only, one
group per trained part, at its defaults otherwise. The learning rate rises linearly over
the first 50 steps and then follows a cosine decay to zero at the end of the training:

    factor(s) = s / 50                                               for s < 50
    factor(s) = 0.5 * (1 + cos(pi * (s - 50) / (total_steps - 50)))  otherwise

where `s` counts the optimiser steps already taken, so the step numbered `s` runs at
`factor(s)` times the group's learning rate. Gradients are clipped to norm 1.0 over all
trained parameters together before each optimiser step.
"""

from __future__ import annotations

import math

import torch

WARMUP_STEPS = 50
CLIP_NORM = 1.0


def learning_rate_factor(step: int, total_steps: int, warmup_steps: int = WARMUP_STEPS) -> float:
    """The fraction of a group's learning rate at the optimiser step numbered `step`."""
    if step < warmup_steps:
        return step / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / decay_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_optimizer(groups: list[tuple[list[torch.nn.Parameter], float]]) -> torch.optim.AdamW:
    """Fused AdamW with one parameter group per `(parameters, learning rate)`."""
    seen: set[int] = set()
    for parameters, _ in groups:
        for parameter in parameters:
            if id(parameter) in seen:
                raise ValueError("a parameter belongs to two groups")
            if parameter.dtype != torch.float32 or not parameter.requires_grad:
                raise ValueError("every trained parameter is float32 and requires a gradient")
            seen.add(id(parameter))
    return torch.optim.AdamW([{"params": list(p), "lr": rate} for p, rate in groups], fused=True)


def build_schedule(optimizer: torch.optim.Optimizer, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    """The warm-up and cosine decay over a training of `total_steps` optimiser steps."""
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: learning_rate_factor(step, total_steps))


def trained_parameters(optimizer: torch.optim.Optimizer) -> list[torch.nn.Parameter]:
    return [p for group in optimizer.param_groups for p in group["params"]]


def clip(optimizer: torch.optim.Optimizer) -> torch.Tensor:
    """Clip the gradients of every trained parameter together to norm 1.0; returns the norm before clipping."""
    return torch.nn.utils.clip_grad_norm_(trained_parameters(optimizer), CLIP_NORM)
