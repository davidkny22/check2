"""Which episodes each training step takes.

The training episodes are the `named` and `reasoned` episodes of `train` that have a
target reply. Each step's episodes are drawn with weights inversely proportional to the
size of their (situation, form, format) cell, so that every cell has the same expected
share of the draws. The draw follows the form of the build's seeded draws: the candidates
sorted by episode id, and one `random.Random(seed)` for the whole training, from which each
step takes `rng.choices(candidates, weights, k=episodes_per_step)`. An epoch is the number
of training episodes divided by the episodes of a step, rounded down.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterator

TRAINING_FORMS = ("named", "reasoned")
CELL_KEYS = ("situation", "form", "format")


class SamplingError(ValueError):
    """Episodes that cannot be drawn from: none, a repeated id, or an episode without a cell."""


def cell_of(episode: dict) -> tuple[str, str, str]:
    """The episode's (situation, form, format)."""
    missing = [key for key in CELL_KEYS if episode.get(key) is None]
    if missing:
        raise SamplingError(f"{episode.get('episode_id')}: no {', '.join(missing)}")
    return tuple(episode[key] for key in CELL_KEYS)


def training_episodes(episodes: list[dict], target_ids) -> list[dict]:
    """The `named` and `reasoned` episodes of `train` that have a target, sorted by episode id."""
    targets = set(target_ids)
    chosen = [e for e in episodes if e["split"] == "train" and e["form"] in TRAINING_FORMS
              and e["episode_id"] in targets]
    return sorted(chosen, key=lambda e: e["episode_id"])


def _ordered(episodes: list[dict]) -> list[dict]:
    if not episodes:
        raise SamplingError("no episodes to draw from")
    ordered = sorted(episodes, key=lambda e: e["episode_id"])
    repeated = [a["episode_id"] for a, b in zip(ordered, ordered[1:]) if a["episode_id"] == b["episode_id"]]
    if repeated:
        raise SamplingError(f"episode ids repeat: {repeated[:3]}")
    return ordered


def cell_weights(episodes: list[dict]) -> list[float]:
    """Each episode's weight, `1 / size of its cell`, in the order of the episodes sorted by id."""
    ordered = _ordered(episodes)
    sizes = Counter(cell_of(e) for e in ordered)
    return [1.0 / sizes[cell_of(e)] for e in ordered]


def steps_per_epoch(count: int, episodes_per_step: int) -> int:
    """The steps of one epoch: the training episodes divided by the episodes of a step, rounded down."""
    steps = count // episodes_per_step
    if steps < 1:
        raise SamplingError(f"{count} episodes make no step of {episodes_per_step}")
    return steps


def step_batches(episodes: list[dict], seed: int, steps: int, episodes_per_step: int = 16,
                 generator: random.Random | None = None) -> Iterator[list[dict]]:
    """Each step's episodes, drawn with replacement with weights that balance the cells.

    The draws come from `random.Random(seed)`. A training that resumes passes, as
    `generator`, that generator restored to the state it had after the steps already
    taken, and `steps` the steps that remain.
    """
    ordered = _ordered(episodes)
    weights = cell_weights(ordered)
    rng = generator if generator is not None else random.Random(seed)
    for _ in range(steps):
        yield rng.choices(ordered, weights=weights, k=episodes_per_step)
