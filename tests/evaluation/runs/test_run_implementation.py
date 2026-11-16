"""A run builds the adapter in the implementation its config names, and every implementation gives the same replies."""

from __future__ import annotations

import pytest
import torch

from check2.evaluation.runs import RunError, execution, output, parts, run
from check2.mechanism.adapter.implementations import IMPLEMENTATIONS
from check2.mechanism.adapter.wrappers import GatedProjection
from check2.mechanism.model_parts import decoder_layers

from .conftest import config

TRAINED = {"adapter.kind": "consumer", "writer.alpha": 1.0}


def _recording_build(monkeypatch) -> dict:
    """Wrap the runner's `build_pipeline`: record the adapter it built, what sits on one projection, and every token."""
    seen: dict = {"adapters": [], "projections": [], "tokens": []}

    def recording(*args, **kwargs):
        pipeline = parts.build_pipeline(*args, **kwargs)
        seen["adapters"].append(type(pipeline.adapter))
        if pipeline.adapter is not None:
            layer = decoder_layers(pipeline.model)[pipeline.adapter.first_layer]
            seen["projections"].append((layer.mlp.gate_proj, len(layer.mlp.gate_proj._forward_hooks)))
        generate = pipeline.generate

        def keeping(*a, **k):
            replies = generate(*a, **k)
            seen["tokens"] += [list(r.token_ids) for r in replies]
            return replies

        pipeline.generate = keeping
        return pipeline

    monkeypatch.setattr(execution, "build_pipeline", recording)
    return seen


def _installed_as(seen_projection, implementation: str) -> bool:
    """Whether the projection, as seen when the pipeline was built, carries the named implementation."""
    projection, forward_hooks = seen_projection
    if implementation == "wrappers":
        return isinstance(projection, GatedProjection)
    if implementation == "library":
        return isinstance(getattr(projection, "base_layer", None), torch.nn.Linear)
    return type(projection) is torch.nn.Linear and forward_hooks > 0


def _run(bench, reader, cfg, run_id, folder):
    return run(cfg, run_id, bench.model, bench.tokenizer, bench.layout, reader, None, episode_folder=folder)


@pytest.mark.parametrize("implementation", ["wrappers", "library"])
def test_a_run_gives_the_hook_run_s_replies_token_for_token(bench, reader, results, episode_folder, checkpoints,
                                                           implementation, monkeypatch):
    original_seen = _recording_build(monkeypatch)
    _run(bench, reader, config(), "original", episode_folder)
    hooks_seen = _recording_build(monkeypatch)
    hooks = _run(bench, reader, config(**TRAINED, **{"adapter.implementation": "hooks"}), "hooks", episode_folder)
    other_seen = _recording_build(monkeypatch)
    other = _run(bench, reader, config(**TRAINED, **{"adapter.implementation": implementation}), implementation,
                 episode_folder)
    assert hooks_seen["adapters"] == [IMPLEMENTATIONS["hooks"]]
    assert other_seen["adapters"] == [IMPLEMENTATIONS[implementation]]
    assert _installed_as(hooks_seen["projections"][0], "hooks")
    assert _installed_as(other_seen["projections"][0], implementation)
    assert len(other_seen["tokens"]) == 20 and other_seen["tokens"] == hooks_seen["tokens"]
    assert output.read_records(other) == output.read_records(hooks)
    # The adapter acts in this implementation: alone, it moves the replies off the original model's.
    alone_seen = _recording_build(monkeypatch)
    _run(bench, reader, config(**{"adapter.kind": "adapter_only", "writer.alpha": 0.0,
                                  "adapter.implementation": implementation}), f"{implementation}-alone", episode_folder)
    assert alone_seen["adapters"] == [IMPLEMENTATIONS[implementation]]
    assert alone_seen["tokens"] != original_seen["tokens"]


def test_a_config_whose_implementation_is_not_written_out_is_refused(bench, reader, results, episode_folder,
                                                                     checkpoints):
    with pytest.raises(RunError, match="writes the chosen implementation out"):
        _run(bench, reader, config(**TRAINED, **{"adapter.implementation": "selected"}), "selected", episode_folder)
    assert not (results / "selected").exists()
    assert checkpoints.calls == []
