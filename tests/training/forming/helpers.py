"""The forming update with random factors, a reader for the tiny model, and what each module adds."""

from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F

from check2.gates import mechanism_setup as setup
from check2.reader.model import Reader
from check2.training.forming.update import FormingUpdate, layers_for

# Layers of the tiny model a reader reads in these tests; the last is the forming update's last layer.
TINY_READ_LAYERS = [1, 2, 5]


def random_update(bench: setup.Bench, seed: int = 7, scale: float = 0.05) -> FormingUpdate:
    """The forming update on the tiny model with every `B` drawn non-zero; not installed."""
    torch.manual_seed(seed)
    update = FormingUpdate(bench.model, "hooks", *layers_for(bench.layout))
    with torch.no_grad():
        for _, b in update.factors().values():
            b.normal_(0.0, scale)
    return update


def tiny_reader(bench: setup.Bench, seed: int = 0) -> Reader:
    torch.manual_seed(seed)
    return Reader(layers=TINY_READ_LAYERS, hidden_size=bench.model.config.hidden_size)


@contextlib.contextmanager
def additions_seen(update: FormingUpdate):
    """Per adapted module, one tensor per call: its output minus `W x + b` recomputed from its input.

    The hooks are registered after the update's own, so each sees the output the update left.
    """
    seen: dict[str, list[torch.Tensor]] = {name: [] for name, _ in update.modules}
    handles = []
    for name, module in update.modules:
        def keep(module, inputs, output, name=name):
            seen[name].append((output - F.linear(inputs[0], module.weight, module.bias)).detach().clone())
        handles.append(module.register_forward_hook(keep))
    try:
        yield seen
    finally:
        for handle in handles:
            handle.remove()
