"""Which episodes each step of the forming training takes.

The episodes are the reader's own training episodes. Each step draws its episodes uniformly
with replacement, in the form of the build's seeded draws: the candidates sorted by episode
id, and one `random.Random(seed)` for the whole training, from which each step takes
`rng.choices(candidates, k=episodes_per_step)`.
"""

from __future__ import annotations

import random
from collections.abc import Iterator


class DrawError(ValueError):
    """No episodes to draw from, or an episode id that repeats."""


def ordered(episodes: list[dict]) -> list[dict]:
    if not episodes:
        raise DrawError("no episodes to draw from")
    out = sorted(episodes, key=lambda e: e["episode_id"])
    repeated = [a["episode_id"] for a, b in zip(out, out[1:]) if a["episode_id"] == b["episode_id"]]
    if repeated:
        raise DrawError(f"episode ids repeat: {repeated[:3]}")
    return out


def step_draws(episodes: list[dict], seed: int, steps: int, episodes_per_step: int,
               generator: random.Random | None = None) -> Iterator[list[dict]]:
    """Each step's episodes. A training that resumes passes its generator as it was after the steps taken."""
    candidates = ordered(episodes)
    rng = generator if generator is not None else random.Random(seed)
    for _ in range(steps):
        yield rng.choices(candidates, k=episodes_per_step)
