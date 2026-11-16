"""The forming update: a second gated low-rank update that acts only at and before the read token.

It is the consumer adapter's update on the same projection names, in layers 3 to 23, with
rank 32 and `alpha_lora` 64, built from the adapter implementation a config names. It has a
control state of its own, whose mask is 1 at tokens at or before the read token that are
not padding and 0 elsewhere: among real tokens it is the complement of the consumer
adapter's mask. In a call over tokens after the read token, as in cached decoding, the
mask is all 0 and the update adds exactly nothing. The *formed model* is the original
model with this update and no other addition.

Whoever runs a forward call sets the update's control state for that call through
`set_call` and clears it afterwards; the pipeline does so in its one forward call.
"""

from __future__ import annotations

import hashlib

import torch

from check2.mechanism.adapter.base import GatedAdapter
from check2.mechanism.adapter.implementations import build
from check2.mechanism.control import ControlState
from check2.mechanism.loading import FULL_LAYOUT, TINY_LAYOUT, Layout

RANK = 32
ALPHA_LORA = 64
# Layers 3 to 23 on the full model: the consumer adapter's first layer to the last read layer.
FULL_LAYERS = (3, 23)
# The tiny model keeps that shape: its adapter's first layer to the last layer its reader tests read.
TINY_LAYERS = (1, 5)


class FormingError(RuntimeError):
    """A forming update used in a way that would let it act where it must not, or not act where it must."""


def layers_for(layout: Layout) -> tuple[int, int]:
    """The first and last layer of the forming update on a model with this layout."""
    if layout == FULL_LAYOUT:
        return FULL_LAYERS
    if layout == TINY_LAYOUT:
        return TINY_LAYERS
    raise FormingError(f"no forming layers are defined for the layout {layout}")


def before_read_mask(attention_mask: torch.Tensor, read_column: torch.Tensor, length: int) -> torch.Tensor:
    """The float mask `[B, length]` for a call over the last `length` columns of `attention_mask`.

    1 where a column lies at or before the row's read column and holds a real token, 0
    elsewhere. `attention_mask` is `[B, C]` with 1 at real tokens; `read_column` is `[B]`.
    """
    total = attention_mask.shape[1]
    columns = torch.arange(total - length, total, device=attention_mask.device)
    at_or_before = columns.unsqueeze(0) <= read_column.to(attention_mask.device).unsqueeze(1)
    real = attention_mask[:, total - length:].bool()
    return (at_or_before & real).to(torch.float32)


class FormingUpdate:
    """The forming update's factors, its hooks and its own control state."""

    def __init__(self, model: torch.nn.Module, implementation: str, first_layer: int, last_layer: int,
                 rank: int = RANK, alpha_lora: float = ALPHA_LORA) -> None:
        self.control = ControlState()
        self.adapter: GatedAdapter = build(implementation, model, self.control, rank, alpha_lora, first_layer,
                                           last_layer)
        self.device = next(model.parameters()).device
        self.installed = False

    @property
    def modules(self) -> list[tuple[str, torch.nn.Linear]]:
        return self.adapter.modules

    def install(self) -> None:
        self.adapter.install()
        self.installed = True

    def remove(self) -> None:
        self.adapter.remove()
        self.installed = False

    def set_call(self, attention_mask: torch.Tensor, read_column: torch.Tensor, length: int) -> None:
        """Set the control state for a call over the last `length` columns of `attention_mask`."""
        if not self.installed:
            raise FormingError("the forming update's control state is set while its hooks are not installed")
        mask = before_read_mask(attention_mask, read_column, length)
        self.control.set(mask.to(self.device), adapter_active=True)

    def clear(self) -> None:
        self.control.clear()

    def parameters(self) -> list[torch.nn.Parameter]:
        return self.adapter.parameters()

    def factors(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        return self.adapter.factors()

    def load_factors(self, factors: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> None:
        self.adapter.load_factors(factors)

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def factors_sha256(self) -> str:
        return factors_sha256(self.factors())


def factors_sha256(factors: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> str:
    """The SHA-256 of a set of factors: for each module in name order, its name, then `A` and `B`
    as float32 bytes with their shapes."""
    digest = hashlib.sha256()
    for name in sorted(factors):
        digest.update(name.encode("utf-8"))
        for tensor in factors[name]:
            value = tensor.detach().to("cpu", torch.float32).contiguous()
            digest.update(repr(tuple(value.shape)).encode("utf-8"))
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()
