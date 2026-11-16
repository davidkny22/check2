"""Where the decoder layers sit in the model, whichever head class was loaded."""

from __future__ import annotations

import torch


def text_model(model: torch.nn.Module) -> torch.nn.Module:
    """The text decoder: the module that holds `embed_tokens`, `layers` and `norm`."""
    inner = model.model
    return getattr(inner, "language_model", inner)


def decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    return text_model(model).layers


def is_full_attention(layer: torch.nn.Module) -> bool:
    return hasattr(layer, "self_attn")
