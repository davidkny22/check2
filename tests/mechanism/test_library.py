"""How the library candidate turns the control state's mask into the library's per-row offsets."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from check2.mechanism.adapter.library import LibraryAdapter, row_offsets
from check2.mechanism.control import ControlState

needs_library = pytest.mark.skipif(importlib.util.find_spec("peft") is None, reason="peft is not installed here")


def test_each_row_offset_counts_from_its_first_token_after_the_read_to_the_end_of_the_call():
    mask = torch.tensor([[0, 0, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 1]], dtype=torch.float32)
    assert row_offsets(mask) == [2, 0, 4, 1]


@needs_library
def test_the_library_adds_the_update_from_the_first_token_after_the_read_to_the_end_of_the_call(bench):
    """Where the mask is a stretch ending at the call's last token, the library's update is the formula's.

    Where a row has padding after its last real token, as a right-padded call over whole
    sequences does, the library also adds the update at that padding, which the mask leaves
    at zero. Only those padding columns differ; the formula gate records the difference.
    """
    control = ControlState()
    adapter = LibraryAdapter(bench.model, control, 4, 8, 3, 3)
    name, module = adapter.modules[0]
    torch.manual_seed(5)
    with torch.no_grad():
        adapter.b[name].normal_()
    x = torch.randn(2, 5, module.in_features)
    update = (8 / 4) * (x @ adapter.a[name].t() @ adapter.b[name].t())
    suffix = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]], dtype=torch.float32)
    padded = torch.tensor([[0, 0, 1, 1, 0], [0, 1, 1, 1, 1]], dtype=torch.float32)
    adapter.install()
    try:
        layer = bench.model.model.layers[3].get_submodule(name.split(".", 2)[2])
        control.set(suffix, adapter_active=True)
        got_suffix = layer(x)
        control.set(padded, adapter_active=True)
        got_padded = layer(x)
    finally:
        control.clear()
        adapter.remove()
    base = module(x)
    assert torch.allclose(got_suffix, base + suffix.unsqueeze(-1) * update, rtol=1e-5, atol=1e-6)
    differs = ~torch.isclose(got_padded, base + padded.unsqueeze(-1) * update, rtol=1e-5, atol=1e-6).all(-1)
    assert differs.tolist() == [[False, False, False, False, True], [False] * 5]
