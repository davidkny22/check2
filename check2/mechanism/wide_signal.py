"""The wide signal: the reader's three probabilities beside the model's own state at the read token.

For each conversation, `S = concat(p, n)`, in float32. `p` is the reader's output. `n` is
the output of layer 18 at the read token, normalised over its own values with no learned
parameter: `n = (h - mean(h)) / sqrt(var(h) + 1e-5)`, computed in float32, each row over
its own values and never over the batch. Layer 18 is the layer of the reader's one-layer
variant. No label and no fitted part enters `n`. The tiny random model has 8 layers, so on
it the state is the output of layer 4, a Gated DeltaNet layer between two write layers, as
layer 18 is on the full model.
"""

from __future__ import annotations

import torch

from check2.config.schema import LINEAR_PROBE_LAYERS
from check2.evaluation.scoring import SITUATIONS
from check2.mechanism.loading import FULL_LAYOUT, TINY_LAYOUT, Layout
from check2.mechanism.write import Writers

STATE_LAYER = LINEAR_PROBE_LAYERS[0]
TINY_STATE_LAYER = 4
NORM_EPSILON = 1e-5
PROBABILITIES = len(SITUATIONS)


class WideSignalError(ValueError):
    """A state or a signal that does not fit the wide signal's shape or its layer."""


def size(hidden_size: int) -> int:
    """The number of values in `S` for a model of the given width."""
    return PROBABILITIES + int(hidden_size)


def state_layer(layout: Layout) -> int:
    """The layer whose output at the read token is the state: 18 on the full model, 4 on the tiny one."""
    if layout == FULL_LAYOUT:
        return STATE_LAYER
    if layout == TINY_LAYOUT:
        return TINY_STATE_LAYER
    raise WideSignalError(f"no state layer is defined for the layout {layout}")


def wide_writers(layout: Layout, hidden_size: int, width: int) -> Writers:
    """Writers that take `S`, reading the state of the layout's state layer, at their starting values."""
    return Writers(layout.write_layers, hidden_size, width, input_size=size(hidden_size),
                   state_layer=state_layer(layout))


def state_position(layers, layer: int = STATE_LAYER) -> int:
    """The position of the state layer in a list of kept layers, which is not its index in the model."""
    layers = [int(j) for j in layers]
    if layer not in layers:
        raise WideSignalError(f"the kept layers {layers} do not include layer {layer}")
    return layers.index(layer)


def normalise(state: torch.Tensor) -> torch.Tensor:
    """`n` for each row of `state` `[B, d]`, in float32, over that row's own values."""
    if state.ndim != 2:
        raise WideSignalError(f"a state must be [batch, width], not {tuple(state.shape)}")
    h = state.to(torch.float32)
    mean = h.mean(dim=-1, keepdim=True)
    var = h.var(dim=-1, unbiased=False, keepdim=True)
    return (h - mean) / torch.sqrt(var + NORM_EPSILON)


def signal(p: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """`S = concat(p, n)`, `[B, 3 + d]` in float32, from `p` `[B, 3]` and the read token's state `[B, d]`."""
    if p.ndim != 2 or p.shape[1] != PROBABILITIES:
        raise WideSignalError(f"p must be [batch, {PROBABILITIES}], not {tuple(p.shape)}")
    if state.ndim != 2 or state.shape[0] != p.shape[0]:
        raise WideSignalError(f"the state {tuple(state.shape)} does not have one row per row of p {tuple(p.shape)}")
    n = normalise(state)
    return torch.cat([p.to(device=n.device, dtype=torch.float32), n], dim=-1)
