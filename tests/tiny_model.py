"""A small random model with the architecture of Qwen3.5, for fast tests on a CPU.

It has the same layer pattern as the full model (three Gated DeltaNet layers, then one
full-attention layer, repeated), the same module names, and the library's built-in
DeltaNet path. Its weights are random, so it tests mask, padding, cache and loss logic
and nothing about language.
"""

from __future__ import annotations

import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

TINY_VOCAB_SIZE = 512
TINY_HIDDEN_SIZE = 32
TINY_NUM_LAYERS = 8
TINY_FULL_ATTENTION_LAYERS = (3, 7)


def tiny_config(num_layers: int = TINY_NUM_LAYERS) -> Qwen3_5TextConfig:
    """Return the configuration of the tiny model, with full attention at every fourth layer."""
    return Qwen3_5TextConfig(
        vocab_size=TINY_VOCAB_SIZE,
        hidden_size=TINY_HIDDEN_SIZE,
        intermediate_size=64,
        num_hidden_layers=num_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        max_position_embeddings=1024,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def tiny_model(seed: int = 0, num_layers: int = TINY_NUM_LAYERS) -> Qwen3_5ForCausalLM:
    """Return a freshly initialised tiny model in float32 on the CPU, in evaluation mode."""
    torch.manual_seed(seed)
    model = Qwen3_5ForCausalLM(tiny_config(num_layers))
    model.eval()
    return model
