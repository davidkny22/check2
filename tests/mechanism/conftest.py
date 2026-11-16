"""The tiny model with the folded tokenizer, and a few real episodes of each format."""

from __future__ import annotations

import pytest

from check2.data import paths
from check2.gates import mechanism_setup as setup


@pytest.fixture(scope="session")
def bench() -> setup.Bench:
    return setup.load("tiny", exact=True)


@pytest.fixture(scope="session")
def episodes() -> dict[str, list[dict]]:
    """Four `heldout` episodes per format, drawn as the gates draw them."""
    return setup.draw(paths.EPISODES / "heldout.jsonl", 16, 0, 4)
