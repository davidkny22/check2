"""The consumer step with the answer token's weight, on the tiny model, in both training forms.

The expected answer position of a made-up short reply is worked out here from its text,
which ends in `Final answer: (X)`, and the tokeniser's offsets; the expected loss of a row
comes from that episode run alone and unpadded.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from check2.data.tokenizer import get_tokenizer
from check2.gates import training_setup
from check2.mechanism.control import ControlState
from check2.mechanism.generate import Pipeline, one_hot_situations
from check2.training import objectives
from check2.training.answer_token import weighting_for
from check2.training.batches import plan_batches
from check2.training.bench import drawn, examples as make_examples
from check2.training.examples import read_targets
from check2.training.forms import SharedPrefix, WholeSequence
from tests.evaluation import faults
from tests.training.made_up_targets import training_candidates

WEIGHT = 0.5
FORMS = (SharedPrefix, WholeSequence)
# Float32 sums over a batch of another shape round differently in the last bits.
SPLIT_TOLERANCE = 1e-5
ROW_TOLERANCE = 1e-5
WEIGHTED_SNAP = "answer-weight-on-snap-rows"
WEIGHTED_SNAP_FAULT = ("check2.training.objectives", "_cross_entropy_pass",
                       "positions = answer.positions_of(prepared.examples).to(logits.device)\n",
                       "positions = answer.positions_of(prepared.examples).to(logits.device).clamp(min=0)\n")


@pytest.fixture(scope="module")
def chosen() -> list[dict]:
    candidates = training_candidates()
    return (drawn([e for e in candidates if e["format"] == "snap"], 8)
            + drawn([e for e in candidates if e["format"] == "short"], 8))


@pytest.fixture(scope="module")
def pipeline(bench):
    control = ControlState()
    writers, adapter = training_setup.random_additions(bench, control)
    with Pipeline(bench.model, bench.tokenizer, control, writers=writers, adapter=adapter) as built:
        yield built


@pytest.fixture(scope="module")
def batch(bench, chosen, targets_file):
    return make_examples(chosen, bench.tokenizer, targets_file)


@pytest.fixture(scope="module")
def replies(targets_file) -> dict[str, str]:
    return {i: target["reply"] for i, target in read_targets(targets_file).items()}


@pytest.fixture(scope="module")
def signal(chosen) -> dict[str, torch.Tensor]:
    return {e["episode_id"]: row for e, row in zip(chosen, one_hot_situations(chosen))}


@pytest.fixture(scope="module")
def weighting(bench, batch, replies):
    return weighting_for(WEIGHT, batch, replies, bench.tokenizer)


def _p(signal, batch) -> torch.Tensor:
    return torch.stack([signal[e.episode_id] for e in batch])


def _position_by_hand(reply: str) -> int:
    """The token that holds the letter of a made-up short reply, which ends in `(X)`."""
    assert reply[-3] == "(" and reply[-1] == ")" and reply[-2] in "ABCD" and reply.count("Final answer:") == 1
    offsets = get_tokenizer()(reply, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    holding = [t for t, (start, end) in enumerate(offsets) if start <= len(reply) - 2 < end]
    assert len(holding) == 1
    return holding[0]


def _alone(pipeline, example, p_row: torch.Tensor) -> torch.Tensor:
    """The write-on pass's per-token losses of one episode run alone, on its whole unpadded sequence."""
    ids = torch.tensor([example.sequence_ids])
    after = (torch.arange(ids.shape[1]) > example.read_index).float().unsqueeze(0)
    with torch.no_grad():
        pipeline.control.set(after, pipeline.writers(p_row[None]), 1.0, (), adapter_active=True)
        try:
            hidden = pipeline.body(input_ids=ids).last_hidden_state
        finally:
            pipeline.control.clear()
        first = len(example.prompt_ids) - 1
        logits = pipeline.model.lm_head(hidden[0, first:first + len(example.target_ids)]).float()
        return F.cross_entropy(logits, torch.tensor(example.target_ids), reduction="none").double()


def test_the_weighting_holds_the_hand_worked_position_of_each_short_episode(batch, replies, weighting):
    short = [e for e in batch if e.format == "short"]
    assert len(short) == 8 and set(weighting.positions) == {e.episode_id for e in short}
    for e in short:
        assert weighting.positions[e.episode_id] == _position_by_hand(replies[e.episode_id]), e.episode_id


@pytest.mark.parametrize("form_class", FORMS, ids=lambda c: c.name)
def test_a_short_row_s_cross_entropy_is_the_formula_on_that_episode_alone(pipeline, batch, replies, signal,
                                                                         weighting, form_class):
    # The 16 rows in one padded batch: snap and short rows side by side, each short row at its own offset.
    losses = objectives.consumer_step(form_class(pipeline), batch, _p(signal, batch), 1.0, len(batch), weighting)
    for row, e in enumerate(batch):
        per_token = _alone(pipeline, e, signal[e.episode_id])
        want = float(per_token.mean())
        if e.format == "short":
            a = _position_by_hand(replies[e.episode_id])
            others = torch.cat([per_token[:a], per_token[a + 1:]])
            want = float(WEIGHT * per_token[a] + (1 - WEIGHT) * others.mean())
            assert abs(want - float(per_token.mean())) > 100 * ROW_TOLERANCE * abs(want), "the weight must show"
        assert abs(float(losses.cross_entropy[row]) - want) <= ROW_TOLERANCE * abs(want), (e.episode_id, e.format)


def _snap_rows_and_the_kl_term_are_unchanged(pipeline, batch, signal, weighting, form_class) -> None:
    form = form_class(pipeline)
    assert any(e.format == "snap" and len(e.target_ids) >= 3 for e in batch), "a snap row the weight would change"
    for part in plan_batches(batch, float("inf"), form.batch_tokens, max_rows=8) + [batch[4:12]]:
        plain = objectives.consumer_step(form, part, _p(signal, part), 1.0, len(batch))
        weighted = objectives.consumer_step(form, part, _p(signal, part), 1.0, len(batch), weighting)
        assert torch.equal(weighted.kl, plain.kl)
        for row, e in enumerate(part):
            if e.format == "snap":
                assert torch.equal(weighted.cross_entropy[row], plain.cross_entropy[row]), e.episode_id
            else:
                assert not torch.equal(weighted.cross_entropy[row], plain.cross_entropy[row]), e.episode_id
        assert torch.equal(weighted.total, weighted.cross_entropy + weighted.kl)


@pytest.mark.parametrize("form_class", FORMS, ids=lambda c: c.name)
def test_snap_rows_and_the_kl_term_are_the_same_numbers_with_and_without_the_weight(pipeline, batch, signal,
                                                                                    weighting, form_class):
    _snap_rows_and_the_kl_term_are_unchanged(pipeline, batch, signal, weighting, form_class)


def test_the_weighted_form_on_snap_rows_turns_the_check_red(pipeline, batch, signal, weighting, monkeypatch):
    monkeypatch.setitem(faults.FAULTS, WEIGHTED_SNAP, WEIGHTED_SNAP_FAULT)
    faults.fails_under(WEIGHTED_SNAP, lambda: _snap_rows_and_the_kl_term_are_unchanged(
        pipeline, batch, signal, weighting, SharedPrefix))


def _step(form, batch, signal, rows: int, parameters, weighting):
    training_setup.clear_gradients(parameters)
    total = torch.zeros((), dtype=torch.float64)
    for part in plan_batches(batch, float("inf"), form.batch_tokens, max_rows=rows):
        total += objectives.consumer_step(form, part, _p(signal, part), 1.0, len(batch),
                                          weighting).total.double().sum()
    return float(total) / len(batch), training_setup.gradients(parameters)


@pytest.mark.parametrize("form_class", FORMS, ids=lambda c: c.name)
def test_the_weighted_step_s_gradient_does_not_depend_on_the_split(pipeline, batch, signal, weighting, form_class):
    form = form_class(pipeline)
    parameters = training_setup.trained_parameters(pipeline)
    one_loss, one = _step(form, batch, signal, 16, parameters, weighting)
    plain_loss, plain = _step(form, batch, signal, 16, parameters, None)
    assert abs(one_loss - plain_loss) > 100 * SPLIT_TOLERANCE * abs(plain_loss), "the weight must change the step"
    for rows in (8, 4, 3):
        loss, grads = _step(form, batch, signal, rows, parameters, weighting)
        assert abs(loss - one_loss) <= SPLIT_TOLERANCE * abs(one_loss)
        for name, g in one.items():
            assert float(g.norm()) > 0.0, f"{name} received no gradient"
            assert float((grads[name] - g).norm()) <= SPLIT_TOLERANCE * float(g.norm()), name
