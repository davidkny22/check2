"""Cross-fitting: each fold's outputs come from a fit that never saw the fold's questions."""
import torch

from check2.reader.crossfit import cross_fit, load_reader, save_reader
from check2.reader.folds import fold_parts
from check2.reader.inputs import Inputs
from check2.reader.model import OUTPUTS
from check2.reader.train import fit, predict
from tests.reader.helpers import random_bank

LAYERS = [14, 15, 18, 19, 22, 23]


def _inputs(n_questions=20):
    train, lengths = [], {}
    for q in range(n_questions):
        for form, situation in (("bare", "hold"), ("named", "user_right"), ("reasoned", "recheck")):
            eid = f"q/{q}:{form}:snap"
            train.append({"episode_id": eid, "qid": f"q/{q}", "form": form, "situation": situation})
            lengths[eid] = 10 + q % 4
    heldout = [{"episode_id": "h/0:bare:snap", "qid": "h/0", "form": "bare", "situation": "hold"}]
    lengths["h/0:bare:snap"] = 11
    return Inputs(train=train, heldout=heldout, bank=random_bank(lengths, len(LAYERS), 8), bank_layers=LAYERS)


FOLDS = [sorted(f"q/{q}" for q in range(n_start, 20, 5)) for n_start in range(5)]


def test_each_fold_is_scored_by_a_fit_on_the_other_four():
    inputs = _inputs()
    p, fold_fits = cross_fit(inputs, FOLDS, "six_layers", 0, epochs=2)
    assert sorted(p) == sorted(e["episode_id"] for e in inputs.train)
    for k, f in enumerate(fold_fits):
        assert f["scored_questions"] == FOLDS[k]
        assert set(f["trained_questions"]).isdisjoint(FOLDS[k])
        assert set(f["trained_questions"]) | set(FOLDS[k]) == {e["qid"] for e in inputs.train}
        # the outputs of fold k are the ones its own fit gives
        trained, scored = fold_parts(inputs.train, FOLDS, k)
        refit = fit("six_layers", inputs.bank, LAYERS, trained, inputs.label_of, 0, seed=0, epochs=2)
        expected = predict(refit.reader, inputs.bank, LAYERS, scored, 0)
        assert torch.equal(torch.tensor([p[e] for e in scored]), expected)


def test_a_saved_reader_loads_with_the_same_outputs(tmp_path):
    inputs = _inputs()
    result = fit("attention_only", inputs.bank, LAYERS, inputs.train_ids, inputs.label_of, 3, seed=0, epochs=1)
    save_reader(result.reader, tmp_path / "reader.pt")
    loaded = load_reader(tmp_path / "reader.pt")
    assert loaded.layers == [15, 19, 23] and not loaded.training
    ids = inputs.train_ids[:7]
    assert torch.equal(predict(result.reader, inputs.bank, LAYERS, ids, 3),
                       predict(loaded, inputs.bank, LAYERS, ids, 3))
    assert predict(loaded, inputs.bank, LAYERS, ids, 3).shape == (7, len(OUTPUTS))
