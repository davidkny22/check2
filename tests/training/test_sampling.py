"""The draw of each step's episodes: balanced cells, and the same batches from the same seed."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from check2.data import paths
from check2.gates import mechanism_setup as setup
from check2.training import sampling
from check2.training.examples import read_targets
from tests.training import made_up_targets

DRAWS = 20_000
STEP = 16


@pytest.fixture(scope="module")
def candidates() -> list[dict]:
    """The training episodes of the build's own files: `named` and `reasoned` `train` episodes with a target."""
    episodes = setup.read_episodes(paths.EPISODES / "train.jsonl")
    return sampling.training_episodes(episodes, read_targets(paths.TARGETS))


def test_the_training_episodes_are_named_and_reasoned_train_episodes_with_a_target(candidates):
    targets = read_targets(paths.TARGETS)
    expected = sorted(e["episode_id"] for e in made_up_targets.training_candidates() if e["episode_id"] in targets)
    assert [e["episode_id"] for e in candidates] == expected
    assert {e["form"] for e in candidates} == {"named", "reasoned"}


def test_each_cell_has_the_same_expected_share_over_20000_draws(candidates):
    sizes = Counter(sampling.cell_of(e) for e in candidates)
    assert len(sizes) > 1 and len(set(sizes.values())) > 1, "the cells must differ in size for the check to bite"
    # The formula: each episode's weight is 1 / its cell's size, so each cell's expected share is 1 / cells.
    ordered = sorted(candidates, key=lambda e: e["episode_id"])
    weights = sampling.cell_weights(candidates)
    total = sum(weights)
    for e, w in zip(ordered, weights):
        assert w / total == pytest.approx(1.0 / (len(sizes) * sizes[sampling.cell_of(e)]), rel=1e-12)
    steps = DRAWS // STEP
    drawn = Counter(sampling.cell_of(e) for batch in sampling.step_batches(candidates, 0, steps) for e in batch)
    expected = 1.0 / len(sizes)
    # Three standard deviations of a binomial share at 20,000 draws.
    tolerance = 3 * (expected * (1 - expected) / DRAWS) ** 0.5
    for cell in sizes:
        assert abs(drawn[cell] / DRAWS - expected) <= tolerance, (cell, drawn[cell] / DRAWS, expected)


def test_the_same_seed_gives_the_same_batches_and_another_seed_others(candidates):
    def ids(seed):
        return [[e["episode_id"] for e in b] for b in sampling.step_batches(candidates, seed, 20)]

    assert ids(0) == ids(0)
    assert ids(0) != ids(1)
    assert all(len(b) == STEP for b in ids(0))


def test_a_restored_generator_continues_the_same_draw(candidates):
    whole = [[e["episode_id"] for e in b] for b in sampling.step_batches(candidates, 0, 10)]
    rng = random.Random(0)
    first = [[e["episode_id"] for e in b] for b in sampling.step_batches(candidates, 0, 4, generator=rng)]
    saved = rng.getstate()
    resumed = random.Random(12345)
    resumed.setstate(saved)
    rest = [[e["episode_id"] for e in b] for b in sampling.step_batches(candidates, 0, 6, generator=resumed)]
    assert first + rest == whole


def test_an_epoch_is_the_episodes_divided_by_the_step_rounded_down():
    assert sampling.steps_per_epoch(3932, 16) == 245
    assert sampling.steps_per_epoch(320, 16) == 20
    with pytest.raises(sampling.SamplingError):
        sampling.steps_per_epoch(15, 16)


def test_a_repeated_episode_id_is_refused(candidates):
    with pytest.raises(sampling.SamplingError):
        list(sampling.step_batches(candidates[:3] + candidates[:1], 0, 1))
