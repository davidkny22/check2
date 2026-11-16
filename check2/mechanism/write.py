"""The writer: one small network per write layer maps the signal `p` to a vector added to the stream.

For each write layer `j`, `v_j = Linear_b_j(GELU(Linear_a_j(p)))`, from 3 to the writer's
hidden width to the model's width. Writers built with a wider input take the wide signal
`S` of `check2.mechanism.wide_signal` in place of `p`, and are otherwise the same.
`Linear_b_j` starts with zero weight and zero bias, so an untrained writer adds nothing. A hook adds `mask * alpha * v_j` to the hidden state
entering layer `j`, where the control state's mask is 1 only at tokens after the read.
"""

from __future__ import annotations

import torch

from check2.config.schema import Writer as WriterSettings
from check2.mechanism.control import ControlState
from check2.mechanism.model_parts import decoder_layers

SIGNAL_SIZE = 3
WRITER_HIDDEN = WriterSettings.hidden
WRITE_HOOK_TAG = "check2_write_layer"


class Writers(torch.nn.Module):
    """The writer networks, one per write layer, with parameters in float32."""

    def __init__(self, layers, hidden_size: int, width: int = WRITER_HIDDEN, input_size: int = SIGNAL_SIZE,
                 state_layer: int | None = None) -> None:
        super().__init__()
        self.layers = tuple(int(j) for j in layers)
        self.input_size = int(input_size)
        # For writers that take the wide signal: the layer whose output at the read token they receive.
        self.state_layer = state_layer
        if self.wide and state_layer is None:
            raise ValueError("writers with a wider input than the three probabilities need the layer of their state")
        self.first = torch.nn.ModuleDict(
            {str(j): torch.nn.Linear(self.input_size, width, dtype=torch.float32) for j in self.layers})
        self.second = torch.nn.ModuleDict(
            {str(j): torch.nn.Linear(width, hidden_size, dtype=torch.float32) for j in self.layers})
        self.activation = torch.nn.GELU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """The last layer of every writer starts at zero weight and zero bias."""
        with torch.no_grad():
            for j in self.layers:
                self.second[str(j)].weight.zero_()
                self.second[str(j)].bias.zero_()

    @property
    def wide(self) -> bool:
        """Whether these writers take the wide signal rather than the three probabilities."""
        return self.input_size != SIGNAL_SIZE

    def forward(self, p: torch.Tensor) -> dict[int, torch.Tensor]:
        """Map the signal `[B, input_size]` to one vector `[B, d]` per write layer, in float32."""
        if p.ndim != 2 or p.shape[1] != self.input_size:
            raise ValueError(f"the writers take a signal of shape [batch, {self.input_size}], not {tuple(p.shape)}")
        p = p.to(torch.float32)
        return {j: self.second[str(j)](self.activation(self.first[str(j)](p))) for j in self.layers}


def _write_hook(control: ControlState, layer: int):
    def hook(module, args, kwargs):
        vector = control.write_vector(layer)
        if vector is None:
            return None
        hidden = args[0] if args else kwargs["hidden_states"]
        mask = control.mask_for(hidden)
        added = mask.unsqueeze(-1) * control.alpha * vector.to(hidden.device).unsqueeze(1)
        changed = hidden + added.to(hidden.dtype)
        if args:
            return (changed, *args[1:]), kwargs
        return args, {**kwargs, "hidden_states": changed}

    setattr(hook, WRITE_HOOK_TAG, layer)
    return hook


def install_write_hooks(model: torch.nn.Module, control: ControlState, layers=WriterSettings.layers) -> list:
    """Add the write to the hidden state entering each write layer; returns the hook handles."""
    blocks = decoder_layers(model)
    return [blocks[j].register_forward_pre_hook(_write_hook(control, j), with_kwargs=True) for j in layers]
