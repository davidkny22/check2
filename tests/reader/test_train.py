"""Fitting a reader: class weights, early stopping on the held-out loss, the best epoch kept."""
import pytest
import torch

from check2.reader.model import OUTPUTS
from check2.reader.train import MAX_EPOCHS, PATIENCE, class_weights, fit, mean_loss, predict
from tests.reader.helpers import random_bank

LAYERS = [14, 15, 18, 19, 22, 23]
D = 8


def _task(n_train=96, n_held=48, seed=0):
    """Made-up states whose read token carries the situation in one direction of layer 18."""
    situations = OUTPUTS
    lengths, label_of = {}, {}
    g = torch.Generator().manual_seed(seed)
    for i in range(n_train + n_held):
        eid = f"{'t' if i < n_train else 'h'}{i:03d}"
        lengths[eid] = 9 + int(torch.randint(1, 6, (1,), generator=g))
        label_of[eid] = situations[i % 3]
    bank = random_bank(lengths, len(LAYERS), D, seed)
    for eid, n in lengths.items():
        start = bank.starts[eid]
        bank.flat[start:start + n, LAYERS.index(18), OUTPUTS.index(label_of[eid])] += 4.0
    train = sorted(e for e in lengths if e.startswith("t"))
    held = sorted(e for e in lengths if e.startswith("h"))
    return bank, label_of, train, held


def test_class_weights_are_inversely_proportional_to_frequency():
    w = class_weights([0, 0, 0, 1, 2, 2])
    # 6 episodes over 3 classes: weights 6 / (3 * count)
    assert w.tolist() == pytest.approx([6 / 9, 6 / 3, 6 / 6])
    with pytest.raises(ValueError):
        class_weights([0, 1, 1])


def test_a_fit_learns_a_signal_present_in_its_states():
    bank, label_of, train, held = _task()
    result = fit("six_layers", bank, LAYERS, train, label_of, 0, seed=0, heldout_ids=held)
    p = predict(result.reader, bank, LAYERS, held, 0)
    labels = torch.tensor([OUTPUTS.index(label_of[e]) for e in held])
    assert (p.argmax(-1) == labels).float().mean() > 0.9
    assert 1 <= result.best_epoch <= result.epochs_run <= MAX_EPOCHS


def test_early_stopping_keeps_the_best_epoch():
    bank, label_of, train, held = _task()
    result = fit("linear_probe", bank, LAYERS, train, label_of, 2, seed=0, heldout_ids=held)
    losses = result.heldout_loss
    assert result.best_epoch == 1 + min(range(len(losses)), key=losses.__getitem__)
    assert result.epochs_run == MAX_EPOCHS or result.epochs_run - result.best_epoch == PATIENCE
    weights = class_weights([OUTPUTS.index(label_of[e]) for e in train])
    kept = mean_loss(result.reader, bank, LAYERS, held, [OUTPUTS.index(label_of[e]) for e in held], weights, 2)
    assert kept == pytest.approx(losses[result.best_epoch - 1], rel=1e-5)


def test_a_fixed_count_refit_reproduces_the_kept_epoch():
    bank, label_of, train, held = _task()
    stopped = fit("deltanet_only", bank, LAYERS, train, label_of, 1, seed=0, heldout_ids=held)
    refit = fit("deltanet_only", bank, LAYERS, train, label_of, 1, seed=0, epochs=stopped.best_epoch)
    assert refit.epochs_run == refit.best_epoch == stopped.best_epoch
    for a, b in zip(stopped.reader.state_dict().values(), refit.reader.state_dict().values()):
        assert torch.equal(a, b)
