"""The training forms and objectives on the tiny model, with random non-zero additions."""

from __future__ import annotations

import pytest
import torch

from check2.gates import mechanism_setup as setup
from check2.gates import split_training, training_setup
from check2.mechanism.control import ControlState
from check2.mechanism.generate import Pipeline, one_hot_situations
from check2.mechanism.write import Writers
from check2.training import forms as forms_module
from check2.training import objectives
from check2.training.batches import plan_batches
from check2.training.bench import drawn, examples as make_examples
from check2.training.forms import SharedPrefix, WholeSequence
from tests.evaluation import faults
from tests.training.made_up_targets import training_candidates

# Float32 sums over a batch of another shape round differently in the last bits.
SPLIT_TOLERANCE = 1e-5
FORMS = (SharedPrefix, WholeSequence)
# The KL term's weight on the one repeat of consumer training after a failed Off gate.
REPEAT_BETA = 4.0
# The reviewer's fault: the KL term's backward pass without its weight, while the reported loss keeps it.
BETA_DROPPED = "beta-dropped-from-kl-backward"
BETA_DROPPED_FAULT = ("check2.training.objectives", "consumer_step",
                      "_backward(form, batch_objective(beta * kl, step_size))",
                      "_backward(form, batch_objective(kl, step_size))")


@pytest.fixture(scope="module")
def chosen() -> list[dict]:
    candidates = training_candidates()
    return (drawn([e for e in candidates if e["format"] == "snap"], 8)
            + drawn([e for e in candidates if e["format"] == "short"], 8))


@pytest.fixture(scope="module")
def pipeline(bench, chosen):
    control = ControlState()
    writers, adapter = setup.random_additions(bench, control, chosen)
    with Pipeline(bench.model, bench.tokenizer, control, writers=writers, adapter=adapter) as built:
        yield built


@pytest.fixture(scope="module")
def batch(bench, chosen, targets_file):
    return make_examples(chosen, bench.tokenizer, targets_file)


@pytest.fixture(scope="module")
def signal(chosen) -> dict[str, torch.Tensor]:
    return {e["episode_id"]: row for e, row in zip(chosen, one_hot_situations(chosen))}


def _p(signal, batch) -> torch.Tensor:
    return torch.stack([signal[e.episode_id] for e in batch])


@pytest.fixture(scope="module")
def gate_scale_pipeline(bench):
    """Writers and adapter factors at the scale of the gates on the training forms, where the KL term is small."""
    control = ControlState()
    writers, adapter = training_setup.random_additions(bench, control)
    with Pipeline(bench.model, bench.tokenizer, control, writers=writers, adapter=adapter) as built:
        yield built


def _step(form, batch, signal, rows: int, parameters, beta: float = 1.0):
    training_setup.clear_gradients(parameters)
    total = torch.zeros((), dtype=torch.float64)
    for part in plan_batches(batch, float("inf"), form.batch_tokens, max_rows=rows):
        total += objectives.consumer_step(form, part, _p(signal, part), beta, len(batch)).total.double().sum()
    return float(total) / len(batch), training_setup.gradients(parameters)


def _beta_4_step_matches_the_plain_reference(pipeline, batch, signal, form_class) -> None:
    """The consumer step at `beta` 4 in two forward batches of 8 gives the reference's gradient, parameter by parameter.

    At this scale the KL term is a small part of the loss, so only the gradients can show
    whether its weight reached the backward pass.
    """
    form = form_class(pipeline)
    assert [len(part) for part in plan_batches(batch, float("inf"), form.batch_tokens, max_rows=8)] == [8, 8]
    parameters = training_setup.trained_parameters(pipeline)
    training_setup.clear_gradients(parameters)
    training_setup.one_at_a_time(pipeline, batch, _p(signal, batch), REPEAT_BETA, len(batch))
    reference = training_setup.gradients(parameters)
    _, got = _step(form, batch, signal, 8, parameters, beta=REPEAT_BETA)
    for name, want in reference.items():
        assert float(want.norm()) > 0.0, f"{name} received no gradient in the reference"
        difference = float((got[name] - want).norm()) / float(want.norm())
        assert difference <= split_training.TOLERANCE, (name, difference)


@pytest.mark.parametrize("form_class", FORMS, ids=lambda c: c.name)
def test_the_consumer_step_at_beta_4_matches_the_plain_reference(gate_scale_pipeline, batch, signal, form_class):
    _beta_4_step_matches_the_plain_reference(gate_scale_pipeline, batch, signal, form_class)


def test_beta_dropped_from_the_kl_backward_pass_turns_the_beta_4_check_red(gate_scale_pipeline, batch, signal,
                                                                           monkeypatch):
    monkeypatch.setitem(faults.FAULTS, BETA_DROPPED, BETA_DROPPED_FAULT)
    faults.fails_under(BETA_DROPPED, lambda: _beta_4_step_matches_the_plain_reference(
        gate_scale_pipeline, batch, signal, SharedPrefix))


@pytest.mark.parametrize("form_class", FORMS, ids=lambda c: c.name)
def test_the_step_gradient_does_not_depend_on_the_split(pipeline, batch, signal, form_class):
    form = form_class(pipeline)
    parameters = training_setup.trained_parameters(pipeline)
    one_loss, one = _step(form, batch, signal, 16, parameters)
    for rows in (8, 4):
        loss, grads = _step(form, batch, signal, rows, parameters)
        assert abs(loss - one_loss) <= SPLIT_TOLERANCE * abs(one_loss)
        for name, g in one.items():
            assert float(g.norm()) > 0.0, f"{name} received no gradient"
            assert float((grads[name] - g).norm()) <= SPLIT_TOLERANCE * float(g.norm()), name


def test_a_pass_s_control_state_holds_until_its_backward_pass_has_finished(pipeline, batch, signal):
    control = pipeline.control
    seen = []

    def snapshot(kind):
        def hook(grad):
            seen.append((kind, control.mask is not None, control.adapter_active, sorted(control.vectors)))
        return hook

    name = pipeline.adapter.modules[0][0]
    write_layer = pipeline.writers.layers[0]
    handles = [pipeline.adapter.b[name].register_hook(snapshot("adapter")),
               pipeline.writers.second[str(write_layer)].weight.register_hook(snapshot("writer"))]
    try:
        part = batch[:4]
        objectives.consumer_step(SharedPrefix(pipeline), part, _p(signal, part), 1.0, len(part))
    finally:
        for handle in handles:
            handle.remove()
    writes = sorted(pipeline.writers.layers)
    assert seen == [("adapter", True, True, []), ("adapter", True, True, writes), ("writer", True, True, writes)] \
        or seen == [("adapter", True, True, []), ("writer", True, True, writes), ("adapter", True, True, writes)]
    assert control.mask is None


def test_the_rebinding_lines_are_what_lets_the_backward_pass_run(pipeline, batch, signal, monkeypatch):
    monkeypatch.setattr(forms_module, "rebind_recurrent_state", lambda cache: None)
    part = batch[:2]
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        objectives.consumer_step(SharedPrefix(pipeline), part, _p(signal, part), 1.0, len(part))
    assert pipeline.control.mask is None


@pytest.mark.parametrize("form_class", FORMS, ids=lambda c: c.name)
def test_the_logits_cover_the_reply_and_the_end_of_turn_token_only(pipeline, batch, form_class):
    form = form_class(pipeline)
    prepared = form.prepare(batch)
    width = max(len(e.target_ids) for e in batch)
    for row, e in enumerate(batch):
        n = len(e.target_ids)
        assert prepared.labels[row, :n].tolist() == list(e.target_ids)
        assert prepared.reply_mask[row].tolist() == [True] * n + [False] * (width - n)
    logits = form.forward(prepared, adapter_on=False, write_on=False, p=None)
    form.end_pass()
    assert logits.shape[:2] == (len(batch), width) and logits.dtype == torch.float32


@pytest.mark.parametrize("form_class", FORMS, ids=lambda c: c.name)
def test_the_forms_predict_the_same_reply_as_one_unpadded_episode(bench, pipeline, batch, form_class):
    form = form_class(pipeline)
    with torch.no_grad():
        logits = form.forward(form.prepare(batch[6:10]), adapter_on=True, write_on=False, p=None)
        form.end_pass()
        for row, e in enumerate(batch[6:10]):
            ids = torch.tensor([e.sequence_ids])
            after = (torch.arange(ids.shape[1]) > e.read_index).float().unsqueeze(0)
            pipeline.control.set(after, {}, 1.0, (), adapter_active=True)
            hidden = pipeline.body(input_ids=ids).last_hidden_state
            pipeline.control.clear()
            first = len(e.prompt_ids) - 1
            alone = bench.model.lm_head(hidden[0, first:first + len(e.target_ids)])
            assert torch.allclose(logits[row, :len(e.target_ids)], alone, atol=1e-5)


def test_connector_and_adapter_only_steps_train_only_their_parts(bench, pipeline, batch, signal):
    part = batch[:3]
    writers = Writers(bench.layout.write_layers, bench.model.config.hidden_size)
    with torch.no_grad():
        for parameter in writers.parameters():
            parameter.normal_()
    with Pipeline(bench.model, bench.tokenizer, ControlState(), writers=writers) as connector:
        losses = objectives.connector_step(SharedPrefix(connector), part, _p(signal, part), 16)
        assert losses.kl is None and torch.equal(losses.total, losses.cross_entropy)
        assert all(p.grad is not None and float(p.grad.norm()) > 0 for p in writers.parameters())
    pipeline.adapter.remove()
    try:
        with Pipeline(bench.model, bench.tokenizer, pipeline.control, adapter=pipeline.adapter) as only:
            training_setup.clear_gradients(training_setup.trained_parameters(pipeline))
            objectives.adapter_only_step(WholeSequence(only), part, 16)
            assert all(p.grad is not None for p in pipeline.adapter.parameters())
            assert all(p.grad is None for p in pipeline.writers.parameters())
    finally:
        pipeline.adapter.install()


def test_a_step_without_the_parts_its_objective_trains_is_refused(bench, pipeline, batch, signal):
    with Pipeline(bench.model, bench.tokenizer, ControlState()) as bare:
        with pytest.raises(objectives.ObjectiveError):
            objectives.consumer_step(SharedPrefix(bare), batch[:1], _p(signal, batch[:1]), 1.0, 16)
    with pytest.raises(objectives.ObjectiveError):
        objectives.connector_step(SharedPrefix(pipeline), batch[:1], _p(signal, batch[:1]), 16)
