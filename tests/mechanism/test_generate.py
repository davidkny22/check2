"""Generation split at the read token, on the tiny model."""

from __future__ import annotations

import pytest
import torch

from check2.gates import cached_equals_uncached, faults, padding_invariance
from check2.gates import mechanism_setup as setup
from check2.mechanism.control import ControlState
from check2.mechanism.generate import Pipeline, PipelineError, Reply, one_hot_situations

# Float32 sums over a batch of another shape round differently in the last bits.
BATCH_SHAPE_TOLERANCE = 1e-5


@pytest.fixture(scope="module")
def additions(bench, episodes):
    control = ControlState()
    writers, adapter = setup.random_additions(bench, control, episodes["snap"] + episodes["short"])
    return control, writers, adapter


def _masks_seen(monkeypatch) -> list[torch.Tensor]:
    seen = []
    original = ControlState.set

    def spy(self, mask, *args, **kwargs):
        seen.append(mask.clone())
        return original(self, mask, *args, **kwargs)
    monkeypatch.setattr(ControlState, "set", spy)
    return seen


def test_the_mask_is_zero_over_the_prefix_and_one_in_decoding(bench, episodes, additions, monkeypatch):
    control, writers, adapter = additions
    batch = episodes["short"]
    seen = _masks_seen(monkeypatch)
    with Pipeline(bench.model, bench.tokenizer, control, writers=writers, adapter=adapter) as pipeline:
        prefilled = pipeline.prefill(batch)
        pipeline.generate(prefilled, one_hot_situations(batch), 3)
    prefix, rest, *decoding = seen
    assert prefix.shape == prefilled.prefix_ids.shape and bool((prefix == 0).all())
    assert rest.shape == prefilled.rest_ids.shape and bool((rest == 1).all())
    assert len(decoding) == 2 and all(m.shape == (len(batch), 1) and bool((m == 1).all()) for m in decoding)
    assert prefilled.prefix_attention.sum(1).tolist() != [prefilled.prefix_ids.shape[1]] * len(batch)


def test_the_read_token_sits_in_the_last_prefix_column(bench, episodes):
    batch = episodes["snap"]
    with setup.plain_pipeline(bench) as pipeline:
        prefilled = pipeline.prefill(batch)
    width = prefilled.prefix_ids.shape[1]
    assert prefilled.read_column.tolist() == [width - 1] * len(batch)
    for b, e in enumerate(batch):
        ids = prefilled.prompt_ids[b]
        assert prefilled.prefix_ids[b, -1].item() == ids[e["read_index"]]
        assert prefilled.rest_ids[b].tolist() == ids[e["read_index"] + 1:]


def test_a_rows_logits_in_a_padded_batch_equal_its_logits_alone(bench, episodes, additions):
    control, writers, adapter = additions
    for fmt, batch in episodes.items():
        p = one_hot_situations(batch)
        with Pipeline(bench.model, bench.tokenizer, control, writers=writers, adapter=adapter) as pipeline:
            together = pipeline.prompt_pass(pipeline.prefill(batch), p)
            for b, e in enumerate(batch):
                alone = pipeline.prompt_pass(pipeline.prefill([e]), p[b:b + 1])[0]
                assert torch.allclose(alone, together[b], rtol=0, atol=BATCH_SHAPE_TOLERANCE), (fmt, b)
                assert alone.argmax() == together[b].argmax()


def test_padded_logits_differ_from_alone_under_the_padding_fault(bench, episodes, additions):
    control, writers, adapter = additions
    batch = episodes["short"]
    p = one_hot_situations(batch)
    worst = 0.0
    with faults.mask_from_column_zero():
        with Pipeline(bench.model, bench.tokenizer, control, writers=writers, adapter=adapter) as pipeline:
            together = pipeline.prompt_pass(pipeline.prefill(batch), p)
            for b, e in enumerate(batch):
                alone = pipeline.prompt_pass(pipeline.prefill([e]), p[b:b + 1])[0]
                worst = max(worst, float((alone - together[b]).abs().max()))
    assert worst > 1000 * BATCH_SHAPE_TOLERANCE


def test_the_appended_read_runs_under_the_additions(bench, episodes, additions):
    control, writers, adapter = additions
    batch = episodes["short"]
    p = one_hot_situations(batch)
    replies = [Reply(text=" The answer stays as it was.", token_ids=[], ended=True) for _ in batch]
    with setup.plain_pipeline(bench) as plain:
        none = plain.appended_letter_logits(plain.prefill(batch), None, replies)
    with Pipeline(bench.model, bench.tokenizer, control, writers=writers, adapter=adapter) as pipeline:
        prefilled = pipeline.prefill(batch)
        added = pipeline.appended_letter_logits(prefilled, p, replies)
        letters = pipeline.appended_read(prefilled, p, replies)
    assert ((added - none).abs().amax(-1) > 1e-3).all()
    assert letters == ["ABCD"[i] for i in added.argmax(-1).tolist()]


def test_the_appended_read_with_the_write_off_and_no_adapter_is_the_plain_read(bench, episodes, additions):
    control, writers, _ = additions
    batch = episodes["snap"]
    replies = [Reply(text="B", token_ids=[], ended=True) for _ in batch]
    with setup.plain_pipeline(bench) as plain:
        none = plain.appended_letter_logits(plain.prefill(batch), None, replies)
    with Pipeline(bench.model, bench.tokenizer, control, writers=writers) as pipeline:
        off = pipeline.appended_letter_logits(pipeline.prefill(batch), one_hot_situations(batch), replies, alpha=0.0)
    assert torch.equal(off, none)


def test_the_perfect_signal_is_the_one_hot_situation(bench, episodes):
    batch = episodes["snap"]
    with setup.plain_pipeline(bench) as pipeline:
        prefilled = pipeline.prefill(batch)
        p, p_reader = pipeline.signal(prefilled, "perfect")
        with pytest.raises(PipelineError):
            pipeline.signal(prefilled, "reader")
    order = ("user_right", "hold", "recheck")
    assert p_reader is None
    assert [order[i] for i in p.argmax(-1).tolist()] == [e["situation"] for e in batch]
    assert torch.equal(p.sum(-1), torch.ones(len(batch)))


class _MeanReader(torch.nn.Module):
    """A stand-in reader: the mean of the window's states, read as three numbers."""

    layers = (2, 3)

    def forward(self, h, mask):
        pooled = (h * mask[:, None, :, None]).sum(2) / mask.sum(-1)[:, None, None]
        return torch.softmax(pooled.sum(1)[:, :3], -1)


def test_the_reader_sees_the_window_with_the_read_token_last(bench, episodes):
    batch = episodes["short"]
    with Pipeline(bench.model, bench.tokenizer, ControlState(), reader=_MeanReader()) as pipeline:
        prefilled = pipeline.prefill(batch)
        p, p_reader = pipeline.signal(prefilled, "perfect")
    assert p_reader is not None and p_reader.shape == (len(batch), 3)
    width = prefilled.window_mask.shape[1]
    assert prefilled.window_states.shape[:3] == (len(batch), 2, width)
    assert prefilled.window_mask[:, -1].all()
    with setup.layer_outputs(bench.model, [3]) as seen, setup.plain_pipeline(bench) as plain:
        plain.prefill(batch)
    assert torch.equal(prefilled.window_states[:, 1, -1], seen[3][0][:, -1])
    assert not torch.equal(p, p_reader)


def test_generation_ends_on_the_end_of_turn_token(bench, episodes, monkeypatch):
    batch = episodes["snap"][:2]
    with setup.plain_pipeline(bench) as pipeline:
        prefilled = pipeline.prefill(batch)
        first = pipeline.generate(prefilled, None, 5)
        stop = first[0].token_ids[2]
        monkeypatch.setattr(pipeline, "end_of_turn_id", stop)
        replies = pipeline.generate(prefilled, None, 5)
    assert replies[0].token_ids == first[0].token_ids[:3] and replies[0].ended
    assert len(first[1].token_ids) == 5


def _fewer(episodes):
    # The short format's cap of 256 tokens dominates the time on the tiny model, which never ends a reply.
    return {"snap": episodes["snap"], "short": episodes["short"][:2]}


def test_padding_invariance_is_green_and_red_under_its_fault(bench, episodes):
    assert padding_invariance.check(bench, _fewer(episodes))["green"]
    assert not padding_invariance.check(bench, _fewer(episodes), plant=True)["green"]


def test_cached_equals_uncached_is_green_and_red_under_its_fault(bench, episodes):
    assert cached_equals_uncached.check(bench, _fewer(episodes))["green"]
    assert not cached_equals_uncached.check(bench, _fewer(episodes), plant=True)["green"]


def test_unequal_tails_are_refused(bench, episodes):
    broken = dict(episodes["snap"][0], read_index=episodes["snap"][0]["read_index"] - 1)
    with setup.plain_pipeline(bench) as pipeline, pytest.raises(PipelineError, match="different numbers"):
        pipeline.prefill([episodes["snap"][1], broken])


def _whole_sequence_outputs(bench, control, writers, adapter, batch, plant=False):
    """Every layer's output in one call over prompt and a short continuation, with no cache."""
    p = one_hot_situations(batch)
    continuation = [[5, 6, 7]] * len(batch)
    with faults.planted("gating", plant):
        with Pipeline(bench.model, bench.tokenizer, control, writers=writers, adapter=adapter) as pipeline:
            prefilled = pipeline.prefill(batch)
            with setup.layer_outputs(bench.model) as seen:
                pipeline.teacher_forced_logits(prefilled, p, continuation, cached=False)
    return prefilled, {layer: calls[0] for layer, calls in seen.items()}


def test_the_whole_sequence_call_leaves_tokens_up_to_the_read_alone(bench, episodes, additions):
    control, writers, adapter = additions
    batch = episodes["short"]
    prefilled, plain = _whole_sequence_outputs(bench, ControlState(), None, None, batch)
    width = prefilled.prefix_ids.shape[1]
    real = prefilled.prefix_attention.bool()
    for arm in ((writers, None), (None, adapter)):
        _, added = _whole_sequence_outputs(bench, control, *arm, batch)
        for layer in plain:
            assert torch.equal(plain[layer][:, :width][real], added[layer][:, :width][real]), layer
        assert not torch.equal(plain[3][:, width], added[3][:, width])
        _, faulty = _whole_sequence_outputs(bench, control, *arm, batch, plant=True)
        assert not torch.equal(plain[3][:, width - 1], faulty[3][:, width - 1])
