"""The adapter's implementations by name, and a copy of an adapter in another implementation.

Every implementation computes the same update on the same projections with factors of the
same shapes, so an adapter's factors load unchanged into any of them.
"""

from __future__ import annotations

import torch

from check2.mechanism.adapter.base import GatedAdapter
from check2.mechanism.adapter.hooks import HookAdapter
from check2.mechanism.adapter.library import LibraryAdapter
from check2.mechanism.adapter.wrappers import WrapperAdapter
from check2.mechanism.control import ControlState

IMPLEMENTATIONS: dict[str, type[GatedAdapter]] = {
    "hooks": HookAdapter,
    "wrappers": WrapperAdapter,
    "library": LibraryAdapter,
}


def build(implementation: str, model: torch.nn.Module, control: ControlState, rank: int, alpha_lora: float,
          first_layer: int, last_layer: int) -> GatedAdapter:
    """A new adapter in the named implementation, its factors at initialisation, not yet installed."""
    if implementation not in IMPLEMENTATIONS:
        raise ValueError(f"the implementation must be one of {sorted(IMPLEMENTATIONS)}, not {implementation!r}")
    return IMPLEMENTATIONS[implementation](model, control, rank, alpha_lora, first_layer, last_layer)


def copy_in(adapter: GatedAdapter, implementation: str, model: torch.nn.Module) -> GatedAdapter:
    """The same adapter, with a copy of its factors, in the named implementation; not installed."""
    out = build(implementation, model, adapter.control, adapter.rank, adapter.alpha_lora, adapter.first_layer,
                adapter.last_layer)
    out.load_factors({name: (a.detach(), b.detach()) for name, (a, b) in adapter.factors().items()})
    return out
