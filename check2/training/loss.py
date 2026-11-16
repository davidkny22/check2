"""The loss of a training step, per episode, so that it does not depend on how the step is split.

An episode's loss is the mean of its per-token losses over the reply's tokens and the
end-of-turn token. A step's loss is the mean of its episodes' losses, so a snap episode of
two tokens and a short episode of 150 weigh the same. A forward batch back-propagates the
sum of its episodes' losses divided by the step's episode count; summed over the forward
batches of a step, that is the step's loss whatever the split. The KL term is averaged the
same way.

In the answer-weighted form, an episode with an answer token takes
`w * l[a] + (1 - w) * mean(l[t] for t != a)` in place of the mean, where `l` holds its
per-token losses over the same positions and `a` is the position that predicts the answer
token. An episode with no answer token, and every episode at `w = 0`, takes the mean.

Every function takes logits at the reply positions only, `[B, L, V]` in float32, where row
`b`'s position `t` predicts `labels[b, t]` and `reply_mask[b, t]` says whether that position
predicts one of the episode's reply tokens or its end-of-turn token.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class LossError(ValueError):
    """Logits, labels and mask that do not fit together, or an episode with nothing to predict."""


def _counts(reply_mask: torch.Tensor) -> torch.Tensor:
    counts = reply_mask.sum(-1)
    if bool((counts == 0).any()):
        raise LossError("an episode has no reply position; every episode predicts at least its end-of-turn token")
    return counts


def episode_losses(logits: torch.Tensor, labels: torch.Tensor, reply_mask: torch.Tensor) -> torch.Tensor:
    """Each episode's mean cross-entropy over its reply tokens and end-of-turn token; `[B]`."""
    if logits.shape[:2] != labels.shape or labels.shape != reply_mask.shape:
        raise LossError(f"logits {tuple(logits.shape)}, labels {tuple(labels.shape)} and mask "
                        f"{tuple(reply_mask.shape)} do not fit together")
    mask = reply_mask.bool()
    counts = _counts(mask)
    # A position outside the reply has no label; any valid id stands in, and the mask removes it.
    safe = labels.masked_fill(~mask, 0)
    per_token = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), safe.reshape(-1),
                                reduction="none").view(labels.shape)
    return (per_token * mask).sum(-1) / counts


def answer_weights(reply_mask: torch.Tensor, answer_position: torch.Tensor, weight: float) -> torch.Tensor:
    """Each episode's weights over its reply positions, `[B, L]`: `weight` at its answer position and the
    rest shared equally by its other reply positions, so that a row sums to 1.

    `answer_position` is `[B]`; every episode has an answer position among its reply positions and at
    least one other reply position.
    """
    mask = reply_mask.bool()
    counts = _counts(mask)
    if answer_position.shape != counts.shape:
        raise LossError(f"answer positions {tuple(answer_position.shape)} for {counts.shape[0]} episodes")
    rows = torch.arange(mask.shape[0], device=mask.device)
    inside = (answer_position >= 0) & (answer_position < mask.shape[1])
    if not bool(inside.all()) or not bool(mask[rows, answer_position].all()):
        raise LossError("an answer position is not one of its episode's reply positions")
    if bool((counts < 2).any()):
        raise LossError("an episode has an answer position and no other reply position")
    answer = torch.zeros_like(mask)
    answer[rows, answer_position] = True
    others = mask & ~answer
    return weight * answer + (1 - weight) * others / (counts - 1).unsqueeze(-1)


def _answer_weighted(logits: torch.Tensor, labels: torch.Tensor, reply_mask: torch.Tensor,
                     answer_position: torch.Tensor, weight: float) -> torch.Tensor:
    if logits.shape[:2] != labels.shape or labels.shape != reply_mask.shape:
        raise LossError(f"logits {tuple(logits.shape)}, labels {tuple(labels.shape)} and mask "
                        f"{tuple(reply_mask.shape)} do not fit together")
    weights = answer_weights(reply_mask, answer_position, weight)
    safe = labels.masked_fill(~reply_mask.bool(), 0)
    per_token = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), safe.reshape(-1),
                                reduction="none").view(labels.shape)
    return (per_token * weights).sum(-1)


def answer_weighted_losses(logits: torch.Tensor, labels: torch.Tensor, reply_mask: torch.Tensor,
                           answer_position: torch.Tensor, weight: float) -> torch.Tensor:
    """Each episode's cross-entropy with its answer token at `weight` of it; `[B]`.

    `answer_position` is `[B]`: the reply position that predicts the episode's answer token, or a
    negative number for an episode with no answer token. Such an episode, and every episode at a
    weight of zero, takes `episode_losses`.
    """
    has_answer = answer_position >= 0
    if weight == 0 or not bool(has_answer.any()):
        return episode_losses(logits, labels, reply_mask)
    if bool(has_answer.all()):
        return _answer_weighted(logits, labels, reply_mask, answer_position, weight)
    out = torch.zeros(logits.shape[0], dtype=logits.dtype, device=logits.device)
    out[has_answer] = _answer_weighted(logits[has_answer], labels[has_answer], reply_mask[has_answer],
                                       answer_position[has_answer], weight)
    out[~has_answer] = episode_losses(logits[~has_answer], labels[~has_answer], reply_mask[~has_answer])
    return out


def episode_kl(reference_logits: torch.Tensor, logits: torch.Tensor, reply_mask: torch.Tensor) -> torch.Tensor:
    """Each episode's mean of `KL(softmax(reference) || softmax(logits))` over its reply positions; `[B]`."""
    if reference_logits.shape != logits.shape or logits.shape[:2] != reply_mask.shape:
        raise LossError(f"reference {tuple(reference_logits.shape)}, logits {tuple(logits.shape)} and mask "
                        f"{tuple(reply_mask.shape)} do not fit together")
    mask = reply_mask.bool()
    counts = _counts(mask)
    reference = F.log_softmax(reference_logits.detach(), dim=-1)
    per_token = F.kl_div(F.log_softmax(logits, dim=-1), reference, log_target=True, reduction="none").sum(-1)
    return (per_token * mask).sum(-1) / counts


def batch_objective(losses: torch.Tensor, step_size: int) -> torch.Tensor:
    """What a forward batch back-propagates: the sum of its episodes' losses divided by the step's episode count."""
    if step_size < losses.shape[0]:
        raise LossError(f"a forward batch of {losses.shape[0]} episodes is larger than its step of {step_size}")
    return losses.sum() / step_size
