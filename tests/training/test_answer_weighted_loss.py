"""The answer-weighted cross-entropy of an episode: the formula computed directly, and the mean where it does not apply."""

from __future__ import annotations

import pytest
import torch

from check2.training import loss as loss_module
from check2.training.loss import LossError, answer_weighted_losses, answer_weights, episode_losses
from tests.evaluation import faults

VOCAB = 11
WEIGHT = 0.5
ONE_BEFORE = "answer-weight-one-position-early"
ONE_BEFORE_FAULT = ("check2.training.loss", "answer_weights", "    answer[rows, answer_position] = True\n",
                    "    answer[rows, answer_position - 1] = True\n")


def _case(lengths: list[int], width: int, seed: int = 0):
    """Rows whose reply positions are left-aligned, as the training forms lay them out."""
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(len(lengths), width, VOCAB, generator=generator)
    labels = torch.randint(0, VOCAB, (len(lengths), width), generator=generator)
    mask = torch.zeros(len(lengths), width, dtype=torch.bool)
    for b, n in enumerate(lengths):
        mask[b, :n] = True
    return logits, labels, mask


def _per_token(logits, labels, row: int, length: int) -> list[float]:
    out = []
    for t in range(length):
        scores = logits[row, t].double()
        out.append(float(torch.logsumexp(scores, 0) - scores[labels[row, t]]))
    return out


def _formula(per_token: list[float], answer: int, weight: float) -> float:
    others = [x for t, x in enumerate(per_token) if t != answer]
    return weight * per_token[answer] + (1 - weight) * sum(others) / len(others)


def _weighted_rows_match_the_formula() -> None:
    lengths, answers = [150, 9, 3, 2], [146, 0, 1, 0]
    logits, labels, mask = _case(lengths, 150)
    got = answer_weighted_losses(logits, labels, mask, torch.tensor(answers), WEIGHT)
    assert got.shape == (4,)
    for row, (n, a) in enumerate(zip(lengths, answers)):
        want = _formula(_per_token(logits, labels, row, n), a, WEIGHT)
        assert abs(float(got[row]) - want) <= 1e-6 * abs(want), row


def test_a_weighted_row_equals_the_formula_computed_directly():
    _weighted_rows_match_the_formula()


def test_the_weight_one_position_early_turns_the_formula_check_red(monkeypatch):
    monkeypatch.setitem(faults.FAULTS, ONE_BEFORE, ONE_BEFORE_FAULT)
    faults.fails_under(ONE_BEFORE, _weighted_rows_match_the_formula)


def test_the_answer_carries_half_and_a_row_s_weights_sum_to_one():
    lengths, answers = [150, 9, 3, 2], [146, 0, 1, 0]
    _, _, mask = _case(lengths, 150)
    weights = answer_weights(mask, torch.tensor(answers), WEIGHT).double()
    assert torch.allclose(weights.sum(-1), torch.ones(4, dtype=torch.float64), atol=1e-6)
    assert bool((weights[~mask] == 0).all())
    for row, (n, a) in enumerate(zip(lengths, answers)):
        assert float(weights[row, a]) == WEIGHT
        others = [float(weights[row, t]) for t in range(n) if t != a]
        assert all(abs(x - (1 - WEIGHT) / (n - 1)) <= 1e-9 for x in others), row
    # With every token's loss the same, the weighted loss is that loss.
    flat = torch.zeros(4, 150, VOCAB)
    labels = torch.zeros(4, 150, dtype=torch.long)
    got = answer_weighted_losses(flat, labels, mask, torch.tensor(answers), WEIGHT)
    assert torch.allclose(got, torch.full((4,), float(torch.log(torch.tensor(float(VOCAB))))), atol=1e-6)


def test_at_a_weight_of_zero_every_row_is_the_mean_bit_for_bit_through_the_mean_s_own_function(monkeypatch):
    lengths, answers = [150, 9, 3, 2], [146, 0, 1, -1]
    logits, labels, mask = _case(lengths, 150)
    want = episode_losses(logits, labels, mask)
    calls = []
    real = loss_module.episode_losses

    def counted(*args):
        calls.append(args)
        return real(*args)

    monkeypatch.setattr(loss_module, "episode_losses", counted)
    monkeypatch.setattr(loss_module, "_answer_weighted", None)  # calling it would raise
    got = answer_weighted_losses(logits, labels, mask, torch.tensor(answers), 0.0)
    assert torch.equal(got, want)
    assert len(calls) == 1 and all(a is b for a, b in zip(calls[0], (logits, labels, mask)))


def test_a_row_with_no_answer_position_keeps_the_mean_bit_for_bit_beside_weighted_rows():
    # Rows 0 and 2 are snap-format replies of three and two positions; rows 1 and 3 are short-format.
    lengths, answers = [3, 150, 2, 40], [-1, 146, -1, 37]
    logits, labels, mask = _case(lengths, 150)
    means = episode_losses(logits, labels, mask)
    got = answer_weighted_losses(logits, labels, mask, torch.tensor(answers), WEIGHT)
    assert torch.equal(got[0], means[0]) and torch.equal(got[2], means[2])
    for row in (1, 3):
        want = _formula(_per_token(logits, labels, row, lengths[row]), answers[row], WEIGHT)
        assert abs(float(got[row]) - want) <= 1e-6 * abs(want)
        assert not torch.equal(got[row], means[row])
    # With no row holding an answer position, the batch is the mean's.
    assert torch.equal(answer_weighted_losses(logits, labels, mask, torch.tensor([-1] * 4), WEIGHT), means)


def test_the_gradient_reaches_each_row_s_logits_with_its_own_weights():
    lengths, answers = [3, 12], [-1, 8]
    logits, labels, mask = _case(lengths, 12)
    logits.requires_grad_(True)
    answer_weighted_losses(logits, labels, mask, torch.tensor(answers), WEIGHT).sum().backward()
    per_position = logits.grad.abs().sum(-1)
    assert bool((per_position[~mask] == 0).all()) and bool((per_position[mask] > 0).all())
    reference = logits.detach().clone().requires_grad_(True)
    per_token = torch.nn.functional.cross_entropy(reference[1, :12], labels[1, :12], reduction="none")
    others = torch.cat([per_token[:8], per_token[9:]])
    (WEIGHT * per_token[8] + (1 - WEIGHT) * others.mean()).backward()
    assert torch.allclose(logits.grad[1], reference.grad[1], atol=1e-7)


@pytest.mark.parametrize("lengths,answers", [([5], [5]), ([5], [7]), ([1], [0])])
def test_an_answer_position_outside_the_reply_or_alone_in_it_is_refused(lengths, answers):
    logits, labels, mask = _case(lengths, 8)
    with pytest.raises(LossError):
        answer_weighted_losses(logits, labels, mask, torch.tensor(answers), WEIGHT)
