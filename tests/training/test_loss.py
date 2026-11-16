"""The step loss: per-episode means over the reply and the end-of-turn token, each episode weighing the same."""

from __future__ import annotations

import math

import pytest
import torch

from check2.training.loss import LossError, batch_objective, episode_kl, episode_losses

VOCAB = 11


def _case(lengths: list[int], width: int, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(len(lengths), width, VOCAB, generator=generator)
    labels = torch.randint(0, VOCAB, (len(lengths), width), generator=generator)
    mask = torch.zeros(len(lengths), width, dtype=torch.bool)
    for b, n in enumerate(lengths):
        mask[b, width - n:] = True  # the reply and its end-of-turn token close the row
    return logits, labels, mask


def _by_hand(logits, labels, mask) -> list[float]:
    out = []
    for b in range(logits.shape[0]):
        terms = []
        for t in range(logits.shape[1]):
            if mask[b, t]:
                row = logits[b, t].double()
                terms.append(float(torch.logsumexp(row, 0) - row[labels[b, t]]))
        out.append(sum(terms) / len(terms))
    return out


def test_an_episode_loss_is_the_mean_over_its_reply_and_end_of_turn_tokens():
    logits, labels, mask = _case([2, 5, 9], 12)
    got = episode_losses(logits, labels, mask)
    assert got.shape == (3,)
    assert torch.allclose(got.double(), torch.tensor(_by_hand(logits, labels, mask), dtype=torch.float64), atol=1e-6)


def test_a_prompt_token_label_changes_nothing_and_the_end_of_turn_label_counts():
    # Twelve positions of one sequence; the last four predict three reply tokens and the end-of-turn token.
    logits, labels, mask = _case([4], 12)
    before = episode_losses(logits, labels, mask)
    prompt_changed = labels.clone()
    prompt_changed[0, :8] = (prompt_changed[0, :8] + 3) % VOCAB
    assert torch.equal(episode_losses(logits, prompt_changed, mask), before)
    end_changed = labels.clone()
    end_changed[0, -1] = (end_changed[0, -1] + 1) % VOCAB
    assert not torch.equal(episode_losses(logits, end_changed, mask), before)


def test_a_two_token_episode_and_a_150_token_episode_weigh_the_same():
    width = 150
    logits = torch.zeros(2, width, VOCAB)
    labels = torch.zeros(2, width, dtype=torch.long)
    mask = torch.zeros(2, width, dtype=torch.bool)
    mask[0, :2] = True
    mask[1, :] = True
    logits[0, :2, 0] = 4.0    # the snap episode is predicted well, the short one uniformly
    losses = episode_losses(logits, labels, mask)
    low, uniform = math.log(VOCAB - 1 + math.exp(4.0)) - 4.0, math.log(VOCAB)
    assert losses.tolist() == pytest.approx([low, uniform], abs=1e-6)
    step = batch_objective(losses, 2)
    assert float(step) == pytest.approx((low + uniform) / 2, abs=1e-6)
    token_mean = (2 * low + 150 * uniform) / 152
    assert abs(float(step) - token_mean) > 0.5


def test_each_episode_receives_the_same_total_gradient_weight():
    logits, labels, mask = _case([2, 150], 150)
    logits.requires_grad_(True)
    batch_objective(episode_losses(logits, labels, mask), 2).backward()
    # The softmax gradient's per-position mass is 1 - p(label); summed per episode and divided by its count.
    p = logits.detach().softmax(-1)
    weight = (1 - p.gather(-1, labels.unsqueeze(-1)).squeeze(-1)) * mask
    expected = weight.sum(-1) / mask.sum(-1) / 2
    got = -logits.grad.gather(-1, labels.unsqueeze(-1)).squeeze(-1).sum(-1)
    assert torch.allclose(got, expected, atol=1e-6)


def test_the_kl_term_is_kl_of_the_reference_from_the_pass_averaged_per_episode():
    reference, labels, mask = _case([3, 7], 8, seed=1)
    logits, _, _ = _case([3, 7], 8, seed=2)
    got = episode_kl(reference, logits, mask)
    p, q = reference.double().softmax(-1), logits.double().softmax(-1)
    per_token = (p * (p.log() - q.log())).sum(-1)
    expected = (per_token * mask).sum(-1) / mask.sum(-1)
    assert torch.allclose(got.double(), expected, atol=1e-6)
    assert torch.all(episode_kl(reference, reference, mask).abs() < 1e-6)


def test_the_reference_carries_no_gradient():
    reference, _, mask = _case([3], 5)
    reference.requires_grad_(True)
    logits, _, _ = _case([3], 5, seed=3)
    logits.requires_grad_(True)
    episode_kl(reference, logits, mask).sum().backward()
    assert reference.grad is None and logits.grad is not None


def test_an_episode_with_nothing_to_predict_is_refused():
    logits, labels, mask = _case([0, 2], 4)
    with pytest.raises(LossError):
        episode_losses(logits, labels, mask)


def test_a_batch_larger_than_its_step_is_refused():
    with pytest.raises(LossError):
        batch_objective(torch.ones(4), 2)
