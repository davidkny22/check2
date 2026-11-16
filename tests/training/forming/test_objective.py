"""The forming step on the tiny model: its KL term at the start, where its gradient goes, its batches."""

from __future__ import annotations

import copy

import torch

from check2.reader.model import OUTPUTS
from check2.training.forming.draws import step_draws
from check2.training.forming.objective import forming_step, make_example
from check2.training.forming.update import FormingUpdate, layers_for
from tests.training.forming.helpers import random_update, tiny_reader

WEIGHTS = torch.tensor([0.8, 1.1, 1.3])


def _examples(bench, episodes):
    return [make_example(e, bench.tokenizer) for e in episodes["snap"] + episodes["short"]]


def test_the_examples_carry_the_window_and_the_situation(bench, episodes):
    for e, x in zip(episodes["snap"] + episodes["short"], _examples(bench, episodes)):
        assert x.read_index == e["read_index"] and len(x.prompt_ids) == e["n_prompt_tokens"]
        assert 0 < x.window_start <= x.read_index
        assert OUTPUTS[x.label] == e["situation"]


def _first_step(bench, examples, weights):
    torch.manual_seed(0)
    update = FormingUpdate(bench.model, "hooks", *layers_for(bench.layout))
    reader = tiny_reader(bench).eval()
    update.install()
    try:
        losses = forming_step(bench.model, update, reader, examples, weights, len(examples), 0)
    finally:
        update.remove()
    return losses, update


def test_the_kl_term_is_exactly_zero_at_the_first_step(bench, episodes):
    examples = _examples(bench, episodes)
    losses, update = _first_step(bench, examples, WEIGHTS)
    assert losses.kl.tolist() == [0.0] * len(examples)
    assert bool((losses.cross_entropy > 0).all())
    # A's gradient is zero while B is. The KL term leaves only a rounding residual on B where
    # the two distributions agree; the reader's term must change every B's gradient.
    without_reader_term, kl_only = _first_step(bench, examples, torch.zeros(3))
    assert without_reader_term.cross_entropy.tolist() == [0.0] * len(examples)
    for name, (a, b) in update.factors().items():
        assert float(a.grad.abs().max()) == 0.0, name
        assert not torch.equal(b.grad, kl_only.factors()[name][1].grad), name


def test_the_gradient_reaches_every_factor_and_every_reader_weight_and_no_model_weight(bench, episodes):
    update = random_update(bench)
    reader = tiny_reader(bench)
    examples = _examples(bench, episodes)
    before = copy.deepcopy(bench.model.state_dict())
    update.install()
    try:
        losses = forming_step(bench.model, update, reader, examples, WEIGHTS, len(examples), 0)
    finally:
        update.remove()
    assert bool((losses.kl > 0).all())
    for name, (a, b) in update.factors().items():
        for factor in (a, b):
            assert factor.grad is not None and bool(torch.isfinite(factor.grad).all()), name
            assert float(factor.grad.norm()) > 0, name
    for name, parameter in reader.named_parameters():
        assert parameter.grad is not None and float(parameter.grad.norm()) > 0, name
    assert all(p.grad is None and not p.requires_grad for p in bench.model.parameters())
    optimizer = torch.optim.AdamW(update.parameters() + list(reader.parameters()), lr=1e-2)
    optimizer.step()
    after = bench.model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_the_step_s_gradient_does_not_depend_on_its_split_into_forward_batches(bench, episodes):
    examples = _examples(bench, episodes)

    def gradients(parts):
        update = random_update(bench)
        reader = tiny_reader(bench).eval()
        update.install()
        try:
            for part in parts:
                forming_step(bench.model, update, reader, part, WEIGHTS, len(examples), 0)
        finally:
            update.remove()
        return [p.grad.clone() for p in update.parameters() + list(reader.parameters())]

    whole = gradients([examples])
    split = gradients([examples[:3], examples[3:5], examples[5:]])
    for a, b in zip(whole, split):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-7)


def test_the_same_seed_gives_the_same_batches():
    episodes = [{"episode_id": f"e{i:03d}"} for i in range(50)]
    first = [[e["episode_id"] for e in step] for step in step_draws(episodes, 0, 6, 16)]
    again = [[e["episode_id"] for e in step] for step in step_draws(list(reversed(episodes)), 0, 6, 16)]
    other = [[e["episode_id"] for e in step] for step in step_draws(episodes, 1, 6, 16)]
    assert first == again and first != other
    assert all(len(step) == 16 for step in first)
