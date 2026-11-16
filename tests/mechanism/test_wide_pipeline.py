"""The pipeline with wide writers: each row's own state at its own read token, on every path that writes."""

from __future__ import annotations

import pytest
import torch

from check2.gates import mechanism_setup as setup
from check2.mechanism import wide_signal
from check2.mechanism.control import ControlState
from check2.mechanism.generate import Pipeline, PipelineError, Reply, one_hot_situations


def _random_wide_writers(bench, seed: int = 3, scale: float = 0.05):
    torch.manual_seed(seed)
    writers = wide_signal.wide_writers(bench.layout, bench.model.config.hidden_size, 32)
    with torch.no_grad():
        for parameter in writers.parameters():
            parameter.normal_().mul_(scale)
    return writers


def _expected_s(bench, episode: dict, p_row: torch.Tensor) -> torch.Tensor:
    """`S` from a pass of this conversation alone, unpadded, and the formula written here."""
    ids = bench.tokenizer([episode["prompt"]])["input_ids"][0][:episode["read_index"] + 1]
    with torch.no_grad():
        out = bench.model(input_ids=torch.tensor([ids]), output_hidden_states=True)
    h = out.hidden_states[wide_signal.TINY_STATE_LAYER + 1][0, -1].to(torch.float32)
    n = (h - h.mean()) / torch.sqrt(((h - h.mean()) ** 2).mean() + 1e-5)
    return torch.cat([p_row.to(torch.float32), n])


def _spy(writers, monkeypatch) -> list[torch.Tensor]:
    seen = []
    original = type(writers).forward

    def spy(self, signal):
        seen.append(signal.detach().clone())
        return original(self, signal)
    monkeypatch.setattr(type(writers), "forward", spy)
    return seen


def test_each_row_of_a_padded_batch_receives_its_own_state_on_every_path(bench, episodes, monkeypatch):
    writers = _random_wide_writers(bench)
    for fmt, batch in episodes.items():
        p = torch.softmax(torch.randn(len(batch), 3), -1)
        expected = torch.stack([_expected_s(bench, e, p[b]) for b, e in enumerate(batch)])
        seen = _spy(writers, monkeypatch)
        with Pipeline(bench.model, bench.tokenizer, ControlState(), writers=writers) as pipeline:
            prefilled = pipeline.prefill(batch)
            assert len(set(prefilled.prefix_attention.sum(1).tolist())) > 1, "the rows must differ in length"
            pipeline.prompt_pass(prefilled, p)
            replies = pipeline.generate(prefilled, p, 2)
            forced = [r.token_ids for r in replies]
            pipeline.teacher_forced_logits(prefilled, p, forced, cached=True)
            pipeline.teacher_forced_logits(prefilled, p, forced, cached=False)
            pipeline.appended_letter_logits(prefilled, p, [Reply("x", [], True) for _ in batch])
        monkeypatch.undo()
        assert len(seen) == 5
        for signal in seen:
            assert signal.shape == (len(batch), wide_signal.size(bench.model.config.hidden_size))
            assert torch.allclose(signal, expected, rtol=1e-4, atol=1e-5), fmt


def test_a_wide_run_at_strength_zero_gives_the_original_replies(bench, episodes):
    writers = _random_wide_writers(bench, scale=1.0)
    for fmt, batch in episodes.items():
        with setup.plain_pipeline(bench) as plain:
            original = plain.generate(plain.prefill(batch), None, 6)
        with Pipeline(bench.model, bench.tokenizer, ControlState(), writers=writers) as pipeline:
            prefilled = pipeline.prefill(batch)
            off = pipeline.generate(prefilled, one_hot_situations(batch), 6, alpha=0.0)
            on = pipeline.generate(prefilled, one_hot_situations(batch), 6, alpha=1.0)
        assert [r.token_ids for r in off] == [r.token_ids for r in original], fmt
        assert [r.token_ids for r in on] != [r.token_ids for r in original], fmt


class _Reader(torch.nn.Module):
    def __init__(self, layers) -> None:
        super().__init__()
        self.layers = tuple(layers)

    def forward(self, h, mask):
        return torch.softmax(h[:, 0, -1, :3].to(torch.float32), -1)


def test_the_state_comes_from_the_readers_kept_outputs_and_a_reader_without_its_layer_is_refused(bench, episodes):
    writers = _random_wide_writers(bench)
    batch = episodes["short"]
    with pytest.raises(PipelineError):
        Pipeline(bench.model, bench.tokenizer, ControlState(), writers=writers, reader=_Reader((2, 3)))
    with Pipeline(bench.model, bench.tokenizer, ControlState(), writers=writers, reader=_Reader((2, 4))) as pipeline:
        from_reader = pipeline.prefill(batch)
    with Pipeline(bench.model, bench.tokenizer, ControlState(), writers=writers) as pipeline:
        own = pipeline.prefill(batch)
    assert torch.equal(from_reader.read_state, own.read_state)
    assert torch.equal(from_reader.read_state, from_reader.window_states[:, 1, -1])
