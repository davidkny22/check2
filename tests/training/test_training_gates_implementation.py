"""The adapter implementation switch on the two training-form gates, on the tiny model."""

from __future__ import annotations

import json
import shutil

import pytest
import torch

from check2 import record
from check2.gates import gradient_parity, split_training, training_setup
from check2.mechanism.adapter.implementations import IMPLEMENTATIONS
from check2.mechanism.adapter.wrappers import GatedProjection
from check2.mechanism.control import ControlState
from check2.mechanism.model_parts import decoder_layers

OTHERS = ["wrappers", "library"]


@pytest.fixture(scope="module")
def episodes(targets_file):
    args = training_setup.parser("split training").parse_args(["--model", "tiny", "--targets", str(targets_file)])
    return training_setup.draw(args, split_training.EPISODES_PER_FORMAT)


def _recording_pipelines(monkeypatch, module) -> list[dict]:
    """Replace the gate module's `Pipeline` by one that records what each pipeline was built with."""
    built: list[dict] = []
    real = module.Pipeline

    class Recording(real):
        def __init__(self, model, *args, **kwargs):
            super().__init__(model, *args, **kwargs)
            first = decoder_layers(model)[self.adapter.first_layer]
            built.append({"adapter": type(self.adapter), "projection": first.mlp.gate_proj})

    monkeypatch.setattr(module, "Pipeline", Recording)
    return built


def _installed_as(projection, implementation: str) -> bool:
    if implementation == "wrappers":
        return isinstance(projection, GatedProjection)
    if implementation == "library":
        return isinstance(getattr(projection, "base_layer", None), torch.nn.Linear)
    return type(projection) is torch.nn.Linear and len(projection._forward_hooks) > 0


@pytest.mark.parametrize("implementation", OTHERS)
def test_every_implementation_gets_the_same_seeded_values(bench, implementation):
    _, hooks = training_setup.random_additions(bench, ControlState())
    _, other = training_setup.random_additions(bench, ControlState(), implementation)
    assert type(other) is IMPLEMENTATIONS[implementation]
    assert set(other.factors()) == set(hooks.factors())
    for name, (a, b) in hooks.factors().items():
        assert torch.equal(other.a[name].detach(), a.detach()) and torch.equal(other.b[name].detach(), b.detach())
        assert other.a[name].requires_grad and other.b[name].requires_grad


@pytest.mark.parametrize("implementation", OTHERS)
def test_the_command_line_switch_reaches_the_pipeline(bench, targets_file, implementation, monkeypatch, tmp_path):
    monkeypatch.setattr(record, "RESULTS", str(tmp_path))
    built = _recording_pipelines(monkeypatch, split_training)
    code = split_training.main(["--model", "tiny", "--targets", str(targets_file), "--limit", "1",
                                "--implementation", implementation])
    assert [b["adapter"] for b in built] == [IMPLEMENTATIONS[implementation]]
    assert _installed_as(built[0]["projection"], implementation)
    saved = json.loads(next(tmp_path.rglob("*.json")).read_text(encoding="utf-8"))
    assert saved["values"]["implementation"] == implementation
    assert saved["values"]["reference_zero_gradients"] == []
    assert code == 0


@pytest.mark.parametrize("implementation", OTHERS)
def test_split_training_is_green_in_both_forms_and_red_under_its_planted_fault(bench, episodes, targets_file,
                                                                               implementation, monkeypatch):
    built = _recording_pipelines(monkeypatch, split_training)
    values = split_training.check(bench, episodes, targets_file, implementation=implementation)
    assert values["implementation"] == implementation and values["reference_zero_gradients"] == []
    for name, form in values["forms"].items():
        assert form["error"] is None and form["green"], (name, form)
    assert values["green"] and values["training_form"] == "shared_prefix"
    planted = split_training.check(bench, episodes, targets_file, plant=True, implementation=implementation)
    shared = planted["forms"]["shared_prefix"]
    assert not shared["green"] and shared["max_gradient_relative_difference"] > 10 * split_training.TOLERANCE
    assert not planted["green"]
    assert [b["adapter"] for b in built] == [IMPLEMENTATIONS[implementation]] * 2
    assert all(_installed_as(b["projection"], implementation) for b in built)


@pytest.mark.parametrize("implementation", OTHERS)
def test_gradient_parity_parts_build_and_name_the_implementation(bench, targets_file, implementation, monkeypatch,
                                                                 tmp_path):
    monkeypatch.setattr(record, "RESULTS", str(tmp_path / "results"))
    built = _recording_pipelines(monkeypatch, gradient_parity)
    out = tmp_path / "parts"
    args = training_setup.parser("gradient parity").parse_args(
        ["--model", "tiny", "--targets", str(targets_file), "--limit", "1", "--implementation", implementation])
    args.out, args.form = str(out), None
    assert gradient_parity.save_part(args, "builtin") == 0
    assert [b["adapter"] for b in built] == [IMPLEMENTATIONS[implementation]]
    assert _installed_as(built[0]["projection"], implementation)
    facts = json.loads((out / "builtin.json").read_text(encoding="utf-8"))
    assert facts["implementation"] == implementation

    # The same gradients stood in for the fast part: the comparison is green only when both parts name one implementation.
    shutil.copy(out / "builtin.pt", out / "fast.pt")
    fast = {**facts, "part": "fast", "dtype": "bfloat16", "fast_kernel_imported": True}
    (out / "fast.json").write_text(json.dumps(fast), encoding="utf-8")
    compare_args = training_setup.parser("gradient parity").parse_args(["--model", "tiny"])
    compare_args.out = str(out)
    assert gradient_parity.compare(compare_args) == 0
    (out / "fast.json").write_text(json.dumps({**fast, "implementation": "hooks"}), encoding="utf-8")
    assert gradient_parity.compare(compare_args) == 1
    records = sorted((tmp_path / "results").rglob("*.json"))
    last = json.loads(records[-1].read_text(encoding="utf-8"))["values"]
    assert last["same_implementation"] is False
    assert last["implementation"] == {"fast": "hooks", "builtin": implementation}
