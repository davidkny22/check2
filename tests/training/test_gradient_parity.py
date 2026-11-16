"""Gradient parity's parts that run here: the seeded additions, the per-layer grouping, and the comparison."""

from __future__ import annotations

import pytest
import torch

from check2.gates import gradient_parity, training_setup
from check2.mechanism.control import ControlState
from check2.mechanism.generate import one_hot_situations
from check2.training import bench as training_bench
from check2.training.forms import SharedPrefix
from check2.training.objectives import consumer_step


@pytest.fixture(scope="module")
def drawn(targets_file):
    args = training_setup.parser("gradient parity").parse_args(["--model", "tiny", "--targets", str(targets_file)])
    return training_setup.draw(args, gradient_parity.EPISODES_PER_FORMAT)


def test_both_processes_draw_the_same_additions(bench):
    first = gradient_parity.small_random_additions(bench, ControlState())
    values = {n: p.detach().clone() for n, p in training_setup.trained_parameters(first).items()}
    first.close()
    second = gradient_parity.small_random_additions(bench, ControlState())
    try:
        again = training_setup.trained_parameters(second)
        assert set(again) == set(values)
        assert all(torch.equal(again[n].detach(), values[n]) for n in values)
        assert all(0.005 < float(v.std()) < 0.02 for v in values.values() if v.numel() > 100)
    finally:
        second.close()


def test_a_writer_counts_with_its_write_layer():
    assert training_setup.layer_of("writer.first.15.weight") == 15
    assert training_setup.layer_of("writer.second.3.bias") == 3
    assert training_setup.layer_of("adapter.layers.12.mlp.up_proj.A") == 12
    reference = {"writer.second.3.bias": torch.tensor([1.0, 0.0]), "adapter.layers.3.mlp.up_proj.A": torch.tensor([0.0, 1.0]),
                 "adapter.layers.4.mlp.up_proj.B": torch.tensor([1.0, 1.0])}
    fast = {"writer.second.3.bias": torch.tensor([1.0, 0.0]), "adapter.layers.3.mlp.up_proj.A": torch.tensor([0.0, -1.0]),
            "adapter.layers.4.mlp.up_proj.B": torch.tensor([2.0, 2.0])}
    cosines = gradient_parity.per_layer_cosines(fast, reference)
    assert cosines == {3: pytest.approx(0.0), 4: pytest.approx(1.0)}


def test_the_training_form_and_the_reference_agree_layer_by_layer_in_float32(bench, drawn, targets_file):
    chosen = drawn["snap"] + drawn["short"]
    examples = training_bench.examples(chosen, bench.tokenizer, targets_file)
    p = one_hot_situations(chosen)
    pipeline = gradient_parity.small_random_additions(bench, ControlState())
    try:
        parameters = training_setup.trained_parameters(pipeline)
        training_setup.clear_gradients(parameters)
        form = SharedPrefix(pipeline)
        for start in (0, 4):
            consumer_step(form, examples[start:start + 4], p[start:start + 4], 1.0, 8)
        fast = training_setup.gradients(parameters)
        training_setup.clear_gradients(parameters)
        training_setup.one_at_a_time(pipeline, examples, p, 1.0, 8)
        reference = training_setup.gradients(parameters)
    finally:
        pipeline.close()
    cosines = gradient_parity.per_layer_cosines(fast, reference)
    assert sorted(cosines) == list(range(bench.layout.adapter_first, bench.layout.adapter_last + 1))
    assert min(cosines.values()) > 1 - 1e-6


def test_the_gate_has_no_tiny_model_form():
    with pytest.raises(SystemExit):
        gradient_parity.main(["--model", "tiny", "--part", "compare", "--out", "x"])
