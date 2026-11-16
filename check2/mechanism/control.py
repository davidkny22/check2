"""What the hooks read in a forward call: which tokens are after the read, what to add, how strongly.

The caller sets the state for one forward call, or one pass of a training step, and it
holds until the caller clears it. Hooks read it when they run and keep nothing from an
earlier call. Nothing else the hooks read is global.
"""

from __future__ import annotations

import torch


class ControlStateError(RuntimeError):
    """A hook found a control state that does not fit the forward call it is in."""


class ControlState:
    """Per batch row: the mask over the current call's tokens, the write vectors and the switches."""

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        """Return to the state in which no hook adds anything."""
        self.mask: torch.Tensor | None = None
        self.vectors: dict[int, torch.Tensor] = {}
        self.alpha: float = 1.0
        self.disabled_layers: frozenset[int] = frozenset()
        self.adapter_active: bool = False
        self.any_after: bool = False

    def set(self, mask: torch.Tensor, vectors: dict[int, torch.Tensor] | None = None, alpha: float = 1.0,
            disabled_layers=(), adapter_active: bool = False) -> None:
        """Set the state for the next forward call.

        `mask` is `[B, T]` over the call's tokens. `vectors` maps a write layer to `[B, d]`.
        `disabled_layers` are write layers whose write is switched off. `adapter_active`
        says whether the consumer adapter acts in this call.
        """
        if mask.ndim != 2:
            raise ControlStateError(f"the mask must be [batch, tokens], not {tuple(mask.shape)}")
        for layer, vector in (vectors or {}).items():
            if vector.ndim != 2 or vector.shape[0] != mask.shape[0]:
                raise ControlStateError(f"the vector for layer {layer} has shape {tuple(vector.shape)}; "
                                        f"expected [{mask.shape[0]}, d]")
        self.mask = mask
        self.vectors = dict(vectors or {})
        self.alpha = float(alpha)
        self.disabled_layers = frozenset(int(j) for j in disabled_layers)
        self.adapter_active = bool(adapter_active)
        # Read once here, so that hooks need not synchronise with the device to learn it.
        self.any_after = bool(mask.any())

    def mask_for(self, hidden: torch.Tensor) -> torch.Tensor:
        """The mask, checked against the batch and token sizes of the tensor a hook sees."""
        if self.mask is None:
            raise ControlStateError("a hook ran while the control state was clear")
        if tuple(self.mask.shape) != tuple(hidden.shape[:2]):
            raise ControlStateError(f"the mask has shape {tuple(self.mask.shape)} but the call has "
                                    f"{tuple(hidden.shape[:2])}; the state was set for another call")
        return self.mask.to(hidden.device)

    def write_vector(self, layer: int) -> torch.Tensor | None:
        """The vector to add at a write layer in this call, or None when nothing is written there."""
        if self.mask is None or not self.any_after or layer in self.disabled_layers:
            return None
        return self.vectors.get(layer)

    def adapter_applies(self) -> bool:
        return self.mask is not None and self.adapter_active and self.any_after
