"""The forming update: its modules, its mask, and the pipeline that sets it."""

from __future__ import annotations

import pytest
import torch
from transformers import AutoConfig, Qwen3_5ForCausalLM

from check2.data.tokenizer import tokenizer_folder
from check2.gates import mechanism_setup as setup
from check2.mechanism.control import ControlState
from check2.mechanism.generate import Pipeline, one_hot_situations
from check2.mechanism.loading import FULL_LAYOUT
from check2.training.forming.update import (FormingError, FormingUpdate, before_read_mask, factors_sha256,
                                            layers_for)
from tests.training.forming.helpers import additions_seen, random_update

ATTENTION = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
DELTANET = ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj")
MLP = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
CONTINUATION = [11, 12, 13]
PAD_WITH_A_NON_ZERO_EMBEDDING = 7


def test_on_the_full_model_the_update_covers_the_132_listed_projections_of_layers_3_to_23():
    config = AutoConfig.from_pretrained(str(tokenizer_folder())).text_config
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)
    update = FormingUpdate(model, "hooks", *layers_for(FULL_LAYOUT))
    expected = set()
    for index in range(3, 24):
        kind = ATTENTION if config.layer_types[index] == "full_attention" else DELTANET
        expected |= {f"layers.{index}.{name}" for name in kind + MLP}
    names = [name for name, _ in update.modules]
    assert len(names) == len(set(names)) == 132
    assert set(names) == expected
    full = [i for i in range(3, 24) if config.layer_types[i] == "full_attention"]
    assert len(full) == 6 and 6 * 7 + 15 * 6 == 132
    assert (update.adapter.rank, update.adapter.alpha_lora) == (32, 64.0)


def test_at_initialisation_b_is_zero_and_a_is_not(bench):
    update = FormingUpdate(bench.model, "hooks", *layers_for(bench.layout))
    for a, b in update.factors().values():
        assert bool((b == 0).all()) and float(a.abs().max()) > 0


def test_the_mask_is_one_at_real_tokens_up_to_the_read_column_and_zero_elsewhere():
    attention = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    read_column = torch.tensor([3, 3])
    whole = before_read_mask(attention, read_column, 6)
    assert whole.tolist() == [[0, 0, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0]]
    # a call over the last two columns lies after the read token
    assert before_read_mask(attention, read_column, 2).tolist() == [[0, 0], [0, 0]]
    # in decoding, one column beyond the read token
    longer = torch.cat([attention, torch.ones(2, 1, dtype=torch.long)], 1)
    assert before_read_mask(longer, read_column, 1).tolist() == [[0], [0]]


def test_setting_the_state_of_an_update_that_is_not_installed_is_refused(bench):
    update = FormingUpdate(bench.model, "hooks", *layers_for(bench.layout))
    with pytest.raises(FormingError):
        update.set_call(torch.ones(1, 3, dtype=torch.long), torch.tensor([1]), 3)


def _run(bench, batch, forming=None, steps=3):
    with Pipeline(bench.model, bench.tokenizer, ControlState(), forming=forming) as pipeline:
        prefilled, outputs = setup.prompt_pass_outputs(pipeline, batch, None)
        tokens = [r.token_ids for r in pipeline.generate(prefilled, None, steps)]
        recomputed = pipeline.teacher_forced_logits(prefilled, None, [CONTINUATION] * len(batch), cached=False)
    return prefilled, outputs, tokens, recomputed


def test_at_initialisation_the_formed_model_equals_the_original_at_every_token(bench, episodes):
    for batch in episodes.values():
        _, plain, plain_tokens, plain_scores = _run(bench, batch)
        update = FormingUpdate(bench.model, "hooks", *layers_for(bench.layout))
        update.install()
        try:
            _, formed, formed_tokens, formed_scores = _run(bench, batch, forming=update)
        finally:
            update.remove()
        for layer in plain:
            for call in range(2):
                assert float((plain[layer][call] - formed[layer][call]).abs().max()) == 0.0
        assert formed_tokens == plain_tokens
        for a, b in zip(plain_scores, formed_scores):
            assert torch.equal(a.top_values, b.top_values) and torch.equal(a.top_ids, b.top_ids)


def test_with_random_factors_each_module_adds_only_at_real_tokens_up_to_the_read_token(bench, episodes):
    batch = episodes["short"]
    update = random_update(bench)
    update.install()
    try:
        with Pipeline(bench.model, bench.tokenizer, ControlState(), forming=update) as pipeline:
            # The tiny model's own pad id has a zero embedding, which would hide an addition at
            # padding; this one does not.
            pipeline.pad_id = PAD_WITH_A_NON_ZERO_EMBEDDING
            assert float(bench.model.get_input_embeddings().weight[pipeline.pad_id].abs().max()) > 0
            with additions_seen(update) as seen:
                prefilled = pipeline.prefill(batch)
                pipeline.teacher_forced_logits(prefilled, None, [CONTINUATION] * len(batch), cached=False)
    finally:
        update.remove()
    prefix_width = prefilled.prefix_ids.shape[1]
    rest = prefilled.rest_ids.shape[1]
    attention = torch.cat([prefilled.prefix_attention, torch.ones(len(batch), rest + len(CONTINUATION) - 1,
                                                                  dtype=torch.long)], 1)
    columns = torch.arange(attention.shape[1])
    region = (columns.unsqueeze(0) <= prefilled.read_column.unsqueeze(1)) & attention.bool()
    assert bool((region[:, prefix_width:] == 0).all())
    assert torch.equal(region[:, :prefix_width], prefilled.prefix_attention.bool())
    assert len(update.modules) == 31
    for name, calls in seen.items():
        prefix_call, whole_call = calls
        # the call over tokens 0..r: real tokens all changed, padding untouched
        real = prefilled.prefix_attention.bool()
        assert float(prefix_call[~real].abs().max()) == 0.0, name
        assert bool((prefix_call[real].norm(dim=-1) > 0).all()), name
        # the whole sequence in one call: nothing after r, nothing at padding
        assert float(whole_call[~region].abs().max()) == 0.0, name
        assert bool((whole_call[region].norm(dim=-1) > 0).all()), name


def test_in_cached_decoding_the_update_adds_exactly_zero(bench, episodes, monkeypatch):
    batch = episodes["snap"]
    update = random_update(bench)
    masks = []
    original = update.control.set

    def spy(mask, *args, **kwargs):
        masks.append(mask.clone())
        return original(mask, *args, **kwargs)
    monkeypatch.setattr(update.control, "set", spy)
    update.install()
    try:
        with Pipeline(bench.model, bench.tokenizer, ControlState(), forming=update) as pipeline:
            prefilled = pipeline.prefill(batch)
            with additions_seen(update) as seen:
                pipeline.generate(prefilled, one_hot_situations(batch), 4)
    finally:
        update.remove()
    prefix_mask, rest_mask, *decoding = masks
    assert torch.equal(prefix_mask, prefilled.prefix_attention.float())
    assert bool((rest_mask == 0).all()) and len(decoding) == 3
    assert all(m.shape == (len(batch), 1) and bool((m == 0).all()) for m in decoding)
    for name, calls in seen.items():
        assert len(calls) == 4, name
        assert all(float(c.abs().max()) == 0.0 for c in calls), name
    assert update.control.mask is None


def test_the_factors_hash_follows_every_value(bench):
    update = random_update(bench)
    factors = {k: (a.clone(), b.clone()) for k, (a, b) in update.factors().items()}
    first = factors_sha256(factors)
    assert first == update.factors_sha256()
    name = sorted(factors)[-1]
    factors[name][0].view(-1)[-1] += 1e-6
    assert factors_sha256(factors) != first
