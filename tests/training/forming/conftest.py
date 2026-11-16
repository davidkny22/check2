"""A few real episodes of each format, drawn as the gates draw them."""

from __future__ import annotations

import pytest

from check2.data import paths
from check2.gates import mechanism_setup as setup


@pytest.fixture(scope="session")
def episodes() -> dict[str, list[dict]]:
    """Four `heldout` episodes per format at the read position `tail[0]`."""
    return setup.draw(paths.EPISODES / "heldout.jsonl", 16, 0, 4)
