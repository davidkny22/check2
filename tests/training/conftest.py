"""The tiny model, and a targets file of made-up replies for real `train` episodes."""

from __future__ import annotations

import pytest

from check2.gates import mechanism_setup as setup
from tests.training import made_up_targets


@pytest.fixture(scope="session")
def bench() -> setup.Bench:
    return setup.load("tiny", exact=True)


@pytest.fixture(scope="session")
def targets_file(tmp_path_factory):
    """A targets file with one made-up reply for every `named` and `reasoned` `train` episode."""
    return made_up_targets.write(tmp_path_factory.mktemp("targets") / "targets.jsonl")
