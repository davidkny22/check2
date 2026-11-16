"""A forming step whose forward batch runs out of memory runs again from its start at half the budget."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

from check2 import record
from check2.config.schema import Config
from check2.reader.crossfit import save_reader
from check2.training import checkpoints
from check2.training.forming import training
from check2.training.forming.objective import batch_tokens, make_example
from tests.training.fake_drive import FolderDrive
from tests.training.forming.helpers import random_update, tiny_reader

WEIGHTS = torch.tensor([0.8, 1.1, 1.3])


def _failing_once(monkeypatch, on_call: int) -> list[int]:
    """`forming_step` raises the out-of-memory error at its `on_call`-th call, after the earlier calls ran."""
    calls = []
    real = training.forming_step

    def step(*args, **kwargs):
        calls.append(1)
        if len(calls) == on_call:
            raise torch.OutOfMemoryError("made-up: the card is full")
        return real(*args, **kwargs)
    monkeypatch.setattr(training, "forming_step", step)
    return calls


def _one_step(bench, examples, budget):
    update = random_update(bench)
    reader = tiny_reader(bench).eval()
    parameters = update.parameters() + list(reader.parameters())
    # Plain gradient descent: Adam's first step moves each weight by about its learning rate times the sign of
    # its gradient, so a gradient near zero, rounded differently in another split of the step, flips a whole step.
    optimizer = torch.optim.SGD(parameters, lr=1e-2)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    update.install()
    try:
        losses = training.run_step(bench.model, update, reader, examples, WEIGHTS, 0, budget, optimizer, schedule)
    finally:
        update.remove()
    return losses, [p.detach().clone() for p in parameters]


def test_a_step_that_runs_out_of_memory_ends_with_the_weights_of_a_step_that_does_not(bench, episodes, monkeypatch):
    examples = [make_example(e, bench.tokenizer) for e in episodes["snap"] + episodes["short"]]
    budget = batch_tokens(examples) / 2  # at least two forward batches, so the first has gradients to clear
    clean, expected = _one_step(bench, examples, budget)
    calls = _failing_once(monkeypatch, on_call=2)
    retried, got = _one_step(bench, examples, budget)
    assert clean.out_of_memory == [] and len(calls) > 2
    assert len(retried.out_of_memory) == 1 and retried.out_of_memory[0]["token_budget"] == budget
    got, expected = torch.cat([g.flatten() for g in got]), torch.cat([e.flatten() for e in expected])
    assert float((got - expected).norm()) <= 1e-6 * float(expected.norm())
    assert retried.total == pytest.approx(clean.total, rel=1e-6)


def test_the_epoch_record_counts_each_out_of_memory_event(bench, tmp_path, monkeypatch):
    monkeypatch.setattr(record, "RESULTS", str(tmp_path / "results"))
    reader_path = tmp_path / "reader" / "reader.pt"
    save_reader(tiny_reader(bench), reader_path)
    base = Config()
    config = replace(base, signal_forming="formed_model_only", reader=replace(base.reader, read_position=0),
                     writer=replace(base.writer, alpha=0.0), adapter=replace(base.adapter, implementation="hooks"))
    storage = checkpoints.Storage(tmp_path / "session", "checkpoints", FolderDrive(tmp_path / "drive"))
    _failing_once(monkeypatch, on_call=2)
    options = training.Options(storage=storage, reader_path=reader_path, episode_limit=32, epochs=1,
                               token_budget=6000)
    training.train(config, options, bench=bench)
    (path,) = (tmp_path / "results" / "measurements" / training.RECORD_NAME).glob("*.json")
    values = json.loads(path.read_text(encoding="utf-8"))["values"]
    assert values["out_of_memory_count"] == 1
    (event,) = values["out_of_memory_events"]
    assert event["token_budget"] == 6000 and event["rows"] > 1 and event["step"] in (0, 1)


def test_a_batch_of_one_row_that_runs_out_of_memory_stops_the_run(bench, episodes, monkeypatch):
    examples = [make_example(e, bench.tokenizer) for e in episodes["snap"][:1]]
    _failing_once(monkeypatch, on_call=1)
    with pytest.raises(training.OutOfMemory):
        _one_step(bench, examples, float("inf"))
