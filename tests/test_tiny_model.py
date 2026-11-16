"""The tiny model behaves like the full one in the ways the fast tests rely on."""

from __future__ import annotations

import torch

from tests.tiny_model import (
    TINY_FULL_ATTENTION_LAYERS,
    TINY_HIDDEN_SIZE,
    TINY_NUM_LAYERS,
    TINY_VOCAB_SIZE,
    tiny_model,
)


def _ids(batch: int, length: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(3, TINY_VOCAB_SIZE, (batch, length), generator=generator)


def test_layer_pattern_matches_the_full_model() -> None:
    model = tiny_model()
    layers = model.model.layers
    assert len(layers) == TINY_NUM_LAYERS
    for index, layer in enumerate(layers):
        if index in TINY_FULL_ATTENTION_LAYERS:
            assert hasattr(layer, "self_attn") and not hasattr(layer, "linear_attn")
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                assert isinstance(getattr(layer.self_attn, name), torch.nn.Linear)
        else:
            assert hasattr(layer, "linear_attn") and not hasattr(layer, "self_attn")
            for name in ("in_proj_qkv", "in_proj_z", "out_proj"):
                assert isinstance(getattr(layer.linear_attn, name), torch.nn.Linear)
        for name in ("gate_proj", "up_proj", "down_proj"):
            assert isinstance(getattr(layer.mlp, name), torch.nn.Linear)


def test_hidden_states_have_one_entry_per_layer_plus_the_embedding() -> None:
    model = tiny_model()
    with torch.no_grad():
        out = model(input_ids=_ids(2, 12), output_hidden_states=True)
    assert len(out.hidden_states) == TINY_NUM_LAYERS + 1
    assert out.hidden_states[-1].shape == (2, 12, TINY_HIDDEN_SIZE)
    assert out.logits.shape == (2, 12, TINY_VOCAB_SIZE)


def test_a_cached_continuation_matches_one_full_pass() -> None:
    model = tiny_model()
    ids = _ids(1, 16)
    with torch.no_grad():
        full = model(input_ids=ids).logits
        prefix = model(input_ids=ids[:, :9], use_cache=True)
        rest = model(input_ids=ids[:, 9:], past_key_values=prefix.past_key_values, use_cache=True).logits
    assert torch.allclose(full[:, 9:], rest, atol=1e-5)


def test_gradients_reach_both_layer_types() -> None:
    model = tiny_model()
    model.train()
    out = model(input_ids=_ids(2, 10), labels=_ids(2, 10, seed=1))
    out.loss.backward()
    assert model.model.layers[0].linear_attn.in_proj_qkv.weight.grad.abs().sum() > 0
    assert model.model.layers[3].self_attn.q_proj.weight.grad.abs().sum() > 0
