"""The consumer adapter computed in modules that replace each adapted projection.

Each listed projection is replaced in its parent by a module that holds the original
projection, calls it on its input `x`, and returns that output plus
`mask * (alpha_lora / rank) * B (A x)` in the output's dtype, reading the control state when
it runs. The factors are the adapter's own, so the update is the hook adapter's update
computed by the same operations in the same order. `remove` puts the original projections
back.
"""

from __future__ import annotations

import torch

from check2.mechanism.adapter.base import GatedAdapter
from check2.mechanism.model_parts import decoder_layers


class GatedProjection(torch.nn.Module):
    """A projection with the adapter's gated update added to its output."""

    def __init__(self, projection: torch.nn.Linear, adapter: "WrapperAdapter", name: str) -> None:
        super().__init__()
        self.projection = projection
        self.name = name
        self.adapter = adapter

    # The wrapped projection's attributes stay reachable where code reads them from the parent.
    @property
    def weight(self) -> torch.nn.Parameter:
        return self.projection.weight

    @property
    def bias(self) -> torch.nn.Parameter | None:
        return self.projection.bias

    @property
    def in_features(self) -> int:
        return self.projection.in_features

    @property
    def out_features(self) -> int:
        return self.projection.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.projection(x)
        control = self.adapter.control
        if not control.adapter_applies():
            return output
        mask = control.mask_for(output)
        change = mask.unsqueeze(-1) * self.adapter.update(self.name, x)
        return output + change.to(output.dtype)


class WrapperAdapter(GatedAdapter):
    """The gated low-rank update, added by modules that replace the named projections."""

    def __init__(self, model: torch.nn.Module, *args, **kwargs) -> None:
        super().__init__(model, *args, **kwargs)
        self.model = model
        self.replaced: list[tuple[torch.nn.Module, str, torch.nn.Linear]] = []

    def _parent(self, name: str) -> tuple[torch.nn.Module, str]:
        """The module that holds the projection `layers.<i>.<path>`, and the attribute it is held under."""
        _, index, *path = name.split(".")
        parent = decoder_layers(self.model)[int(index)]
        for part in path[:-1]:
            parent = getattr(parent, part)
        return parent, path[-1]

    def install(self) -> None:
        if self.replaced:
            raise RuntimeError("the adapter's wrappers are already installed")
        for name, projection in self.modules:
            parent, attribute = self._parent(name)
            if getattr(parent, attribute) is not projection:
                raise RuntimeError(f"{name} is no longer the projection this adapter was built on")
            setattr(parent, attribute, GatedProjection(projection, self, name))
            self.replaced.append((parent, attribute, projection))

    def remove(self) -> None:
        for parent, attribute, projection in self.replaced:
            setattr(parent, attribute, projection)
        self.replaced = []
