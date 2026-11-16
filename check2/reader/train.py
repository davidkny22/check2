"""Fitting a reader on cached states: weighted cross-entropy against the situation, AdamW, batches
of 256, at most 30 epochs, stopping when the `heldout` loss has not improved for 3 epochs and
keeping the best epoch; or a fixed number of epochs with no `heldout` set.
"""
import copy
import random
from dataclasses import dataclass, field

import torch
from torch import nn

from check2.reader.cache import StateBank
from check2.reader.model import OUTPUTS, VARIANTS, Reader

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
BATCH_SIZE = 256
MAX_EPOCHS = 30
PATIENCE = 3


@dataclass
class Fit:
    reader: Reader
    best_epoch: int
    epochs_run: int
    train_loss: list[float] = field(default_factory=list)
    heldout_loss: list[float] = field(default_factory=list)


def make_reader(variant: str, hidden_size: int, seed: int) -> Reader:
    """A fresh reader of the named variant, its initial weights drawn from `seed`."""
    torch.manual_seed(seed)
    return Reader(hidden_size=hidden_size, **VARIANTS[variant])


def class_weights(labels: list[int]) -> torch.Tensor:
    """Weights inversely proportional to each class's frequency among `labels`."""
    counts = torch.bincount(torch.tensor(labels), minlength=len(OUTPUTS)).double()
    if (counts == 0).any():
        missing = [OUTPUTS[i] for i in range(len(OUTPUTS)) if counts[i] == 0]
        raise ValueError(f"the training episodes hold no {missing} episode")
    return (counts.sum() / (len(OUTPUTS) * counts)).float()


def layer_index(reader: Reader, bank_layers: list[int]) -> list[int]:
    return [list(bank_layers).index(layer) for layer in reader.layers]


def _batches(ids: list[str], size: int) -> list[list[str]]:
    return [ids[i:i + size] for i in range(0, len(ids), size)]


@torch.no_grad()
def predict(reader: Reader, bank: StateBank, bank_layers: list[int], episode_ids: list[str],
            tail_position: int, batch_size: int = BATCH_SIZE) -> torch.Tensor:
    """`p` for each episode, `[N, 3]`, on the CPU, with the reader in evaluation mode."""
    reader.eval()
    index = layer_index(reader, bank_layers)
    out = [reader(*bank.window(chunk, tail_position, index)).cpu() for chunk in _batches(episode_ids, batch_size)]
    return torch.cat(out)


@torch.no_grad()
def mean_loss(reader: Reader, bank: StateBank, bank_layers: list[int], episode_ids: list[str],
              labels: list[int], weights: torch.Tensor, tail_position: int) -> float:
    """The weighted cross-entropy over the episodes, in evaluation mode."""
    reader.eval()
    index = layer_index(reader, bank_layers)
    weights = weights.to(bank.device)
    total, weight_sum = 0.0, 0.0
    for chunk, label_chunk in zip(_batches(episode_ids, BATCH_SIZE), _batches(labels, BATCH_SIZE)):
        y = torch.tensor(label_chunk, device=bank.device)
        losses = nn.functional.cross_entropy(reader.logits(*bank.window(chunk, tail_position, index)), y,
                                             weight=weights, reduction="none")
        total += float(losses.sum())
        weight_sum += float(weights[y].sum())
    return total / weight_sum


def fit(variant: str, bank: StateBank, bank_layers: list[int], train_ids: list[str],
        label_of: dict[str, str], tail_position: int, seed: int,
        heldout_ids: list[str] | None = None, epochs: int = MAX_EPOCHS) -> Fit:
    """Fit a reader of `variant` on `train_ids`.

    With `heldout_ids`, training stops when the `heldout` loss has not improved for `PATIENCE`
    epochs, or at `epochs`, and the reader of the best epoch is kept. Without them, it runs
    exactly `epochs` epochs and keeps the last. Each epoch visits the training episodes in an
    order drawn from one `random.Random(seed)`.
    """
    train_ids = sorted(train_ids)
    labels = {e: OUTPUTS.index(label_of[e]) for e in train_ids + sorted(heldout_ids or [])}
    weights = class_weights([labels[e] for e in train_ids])
    reader = make_reader(variant, bank.flat.shape[-1], seed).to(bank.device)
    index = layer_index(reader, bank_layers)
    optimiser = torch.optim.AdamW(reader.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(bank.device))
    rng = random.Random(seed)
    result = Fit(reader=reader, best_epoch=0, epochs_run=0)
    best_loss, best_state = float("inf"), None
    for epoch in range(1, epochs + 1):
        reader.train()
        order = rng.sample(train_ids, len(train_ids))
        total, count = 0.0, 0
        for chunk in _batches(order, BATCH_SIZE):
            y = torch.tensor([labels[e] for e in chunk], device=bank.device)
            loss = loss_fn(reader.logits(*bank.window(chunk, tail_position, index)), y)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total += float(loss) * len(chunk)
            count += len(chunk)
        result.train_loss.append(total / count)
        result.epochs_run = epoch
        if heldout_ids is None:
            continue
        held = sorted(heldout_ids)
        loss = mean_loss(reader, bank, bank_layers, held, [labels[e] for e in held], weights, tail_position)
        result.heldout_loss.append(loss)
        if loss < best_loss:
            best_loss, best_state, result.best_epoch = loss, copy.deepcopy(reader.state_dict()), epoch
        elif epoch - result.best_epoch >= PATIENCE:
            break
    if heldout_ids is None:
        result.best_epoch = result.epochs_run
    else:
        reader.load_state_dict(best_state)
    reader.eval()
    return result
