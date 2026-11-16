"""The timed training step runs with no adapter and with every implementation, on the tiny model."""

from __future__ import annotations

import pytest

from check2.training import bench as training_bench
from check2.training.forms import FORMS
from check2.training.step_timing import time_step
from tests.training.made_up_targets import training_candidates


@pytest.mark.parametrize("adapter", training_bench.ADAPTER_CHOICES)
def test_one_step_is_timed_with_each_adapter(bench, targets_file, adapter):
    episodes = training_bench.drawn([e for e in training_candidates() if e["format"] == "short"], 4)
    examples = training_bench.examples(episodes, bench.tokenizer, targets_file)
    pipeline = training_bench.training_pipeline(bench, adapter)
    try:
        values = time_step(FORMS["shared_prefix"](pipeline), examples, 1.0, float("inf"))
    finally:
        pipeline.close()
    assert values["seconds"] > 0 and len(values["warm_up_seconds"]) == 2
    assert values["batches"] == [{"rows": 4, "tokens": values["batches"][0]["tokens"]}]
    assert all(p.grad is not None for p in pipeline.writers.parameters())
    if pipeline.adapter is not None:
        assert all(p.grad is not None for p in pipeline.adapter.parameters())
