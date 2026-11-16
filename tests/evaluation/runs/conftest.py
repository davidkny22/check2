"""Runs on the tiny model: a few real `heldout` episodes, small batches, short caps, stand-in parts.

The reader is the package's own, at random weights, saved and loaded as a run loads it. The
checkpoint loader is a stand-in with the interface the runner calls:
`check2.training.checkpoints.load_for_run(config, epoch) -> (writers_state, factors)`.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from dataclasses import replace

import pytest
import torch

from check2 import record
from check2.config.schema import Config
from check2.data import paths
from check2.evaluation.runs import episodes as run_episodes
from check2.gates import mechanism_setup as setup
from check2.mechanism.control import ControlState
from check2.mechanism.generate import REPLY_CAPS
from check2.reader.crossfit import load_reader, save_reader
from check2.reader.model import Reader

PER_FORMAT = 8
SMALL_BATCH = 2
SMALL_CAPS = {"snap": 6, "short": 8}


@pytest.fixture(scope="session")
def bench() -> setup.Bench:
    return setup.load("tiny", exact=True)


TINY_READ_LAYERS = [2, 5]


@pytest.fixture(scope="session")
def reader(bench, tmp_path_factory):
    """The reader of the pooled variant on two of the tiny model's layers, seeded, saved and loaded back."""
    torch.manual_seed(0)
    fitted = Reader(TINY_READ_LAYERS, bench.model.config.hidden_size)
    with torch.no_grad():
        fitted.u.normal_()
    path = tmp_path_factory.mktemp("reader") / "reader.pt"
    save_reader(fitted, path)
    return load_reader(path)


@pytest.fixture(scope="session")
def small_episodes() -> list[dict]:
    """The shortest `heldout` episodes, `PER_FORMAT` per format, with every form present."""
    with (paths.EPISODES / "heldout.jsonl").open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    chosen = []
    for fmt in ("snap", "short"):
        mine = sorted((e for e in rows if e["format"] == fmt), key=lambda e: (e["n_prompt_tokens"], e["episode_id"]))
        for form in ("bare", "named", "reasoned"):
            chosen += [e for e in mine if e["form"] == form][:PER_FORMAT // 4 + (2 if form == "bare" else 1)]
    return chosen


@pytest.fixture()
def episode_folder(tmp_path, small_episodes):
    folder = tmp_path / "episodes"
    folder.mkdir()
    with (folder / "heldout.jsonl").open("w", encoding="utf-8") as f:
        for e in small_episodes:
            f.write(json.dumps(e) + "\n")
    return folder


@pytest.fixture()
def results(tmp_path, monkeypatch):
    """Run folders go under a temporary folder; batches are small and replies short."""
    folder = tmp_path / "results"
    monkeypatch.setattr(record, "RESULTS", str(folder))
    monkeypatch.setattr(run_episodes, "BATCH_SIZE", SMALL_BATCH)
    for fmt, cap in SMALL_CAPS.items():
        monkeypatch.setitem(REPLY_CAPS, fmt, cap)
    return folder


def config(**changes) -> Config:
    """A heldout run's settings at read position 0; `changes` maps 'section.key' or 'key' to a value."""
    base = Config()
    base = replace(base, data=replace(base.data, eval_split="heldout"), reader=replace(base.reader, read_position=0),
                   adapter=replace(base.adapter, implementation="hooks"), writer=replace(base.writer, alpha=0.0))
    for key, value in changes.items():
        if "." in key:
            section, name = key.split(".")
            base = replace(base, **{section: replace(getattr(base, section), **{name: value})})
        else:
            base = replace(base, **{key: value})
    return base


@pytest.fixture(scope="session")
def trained(bench, small_episodes):
    """Random non-zero writers and adapter factors, at the scale the mechanism's gates use."""
    writers, adapter = setup.random_additions(bench, ControlState(), small_episodes)
    return writers.state_dict(), {name: (a.detach().clone(), b.detach().clone())
                                  for name, (a, b) in adapter.factors().items()}


@pytest.fixture()
def checkpoints(monkeypatch, trained):
    """A stand-in `check2.training.checkpoints` whose answer the test sets; records the calls."""
    try:
        importlib.import_module("check2.training")
    except ModuleNotFoundError:
        monkeypatch.setitem(sys.modules, "check2.training", types.ModuleType("check2.training"))
    module = types.ModuleType("check2.training.checkpoints")
    module.calls = []
    module.answer = trained

    def load_for_run(cfg, epoch=None):
        module.calls.append((cfg, epoch))
        return module.answer
    module.load_for_run = load_for_run
    monkeypatch.setitem(sys.modules, "check2.training.checkpoints", module)
    return module
