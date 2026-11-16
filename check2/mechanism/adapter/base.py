"""The consumer adapter's modules and the interface every implementation of it provides.

The adapter is a low-rank update on named linear projections in a range of layers:
`q_proj`, `k_proj`, `v_proj` and `o_proj` of each full-attention layer, `in_proj_qkv`,
`in_proj_z` and `out_proj` of each Gated DeltaNet layer, and `gate_proj`, `up_proj` and
`down_proj` of every layer's MLP. For a projection with weight `W` and input `x`,

    y = W x + mask * (alpha_lora / rank) * B (A x)

with `A` Kaiming-uniform and `B` zero at initialisation, both in float32, and the mask the
control state's, so that only tokens after the read are changed.
"""

from __future__ import annotations

import math

import torch

from check2.mechanism.control import ControlState
from check2.mechanism.model_parts import decoder_layers, is_full_attention

FULL_ATTENTION_PROJECTIONS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
DELTANET_PROJECTIONS = ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj")
MLP_PROJECTIONS = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
# The bound torch.nn.Linear and the usual low-rank adapter use for their Kaiming-uniform start.
KAIMING_A = math.sqrt(5)


def _submodule(root: torch.nn.Module, dotted: str) -> torch.nn.Module:
    for part in dotted.split("."):
        root = getattr(root, part)
    return root


def adapted_modules(model: torch.nn.Module, first_layer: int, last_layer: int) -> list[tuple[str, torch.nn.Linear]]:
    """The adapted projections of layers `first_layer..last_layer`, by layer, in a fixed order."""
    blocks = decoder_layers(model)
    out = []
    for index in range(first_layer, last_layer + 1):
        layer = blocks[index]
        names = (FULL_ATTENTION_PROJECTIONS if is_full_attention(layer) else DELTANET_PROJECTIONS) + MLP_PROJECTIONS
        for name in names:
            module = _submodule(layer, name)
            if not isinstance(module, torch.nn.Linear):
                raise TypeError(f"layers.{index}.{name} is a {type(module).__name__}, not a Linear")
            out.append((f"layers.{index}.{name}", module))
    return out


class GatedAdapter:
    """One factor pair per adapted projection, applied only where the control state's mask is 1.

    A candidate implementation subclasses this and provides `install` and `remove`.
    """

    def __init__(self, model: torch.nn.Module, control: ControlState, rank: int, alpha_lora: float,
                 first_layer: int, last_layer: int) -> None:
        self.control = control
        self.rank = int(rank)
        self.alpha_lora = float(alpha_lora)
        self.scale = self.alpha_lora / self.rank
        self.first_layer, self.last_layer = int(first_layer), int(last_layer)
        self.modules = adapted_modules(model, self.first_layer, self.last_layer)
        self.a: dict[str, torch.nn.Parameter] = {}
        self.b: dict[str, torch.nn.Parameter] = {}
        for name, module in self.modules:
            device = module.weight.device
            a = torch.empty(self.rank, module.in_features, dtype=torch.float32, device=device)
            torch.nn.init.kaiming_uniform_(a, a=KAIMING_A)
            self.a[name] = torch.nn.Parameter(a)
            self.b[name] = torch.nn.Parameter(
                torch.zeros(module.out_features, self.rank, dtype=torch.float32, device=device))

    def update(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """`(alpha_lora / rank) * B (A x)` for one projection, in float32, before the mask."""
        a, b = self.a[name], self.b[name]
        return ((x.to(torch.float32) @ a.t()) @ b.t()) * self.scale

    def install(self) -> None:
        raise NotImplementedError

    def remove(self) -> None:
        raise NotImplementedError

    def parameters(self) -> list[torch.nn.Parameter]:
        """Every factor, `A` then `B` for each projection, in the order of `adapted_modules`."""
        out = []
        for name, _ in self.modules:
            out += [self.a[name], self.b[name]]
        return out

    def factors(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        return {name: (self.a[name], self.b[name]) for name, _ in self.modules}

    def load_factors(self, factors: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Copy saved factors in; the names must be exactly this adapter's and the shapes must match."""
        if set(factors) != set(self.a):
            missing, extra = sorted(set(self.a) - set(factors)), sorted(set(factors) - set(self.a))
            raise KeyError(f"the factors do not match the adapted modules: missing {missing[:3]}, "
                           f"extra {extra[:3]}")
        with torch.no_grad():
            for name, (a, b) in factors.items():
                if tuple(a.shape) != tuple(self.a[name].shape) or tuple(b.shape) != tuple(self.b[name].shape):
                    raise ValueError(f"{name}: factor shapes {tuple(a.shape)}, {tuple(b.shape)} do not match "
                                     f"{tuple(self.a[name].shape)}, {tuple(self.b[name].shape)}")
                self.a[name].copy_(a)
                self.b[name].copy_(b)
