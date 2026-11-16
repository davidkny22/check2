"""The splits: drawn once, reproducible, whole duplicate groups, and independent of reasons."""
import json
import random
from dataclasses import asdict, replace

import pytest

from check2.data import paths
from check2.data.pairs import all_pairs
from check2.data.pool import complying, duplicate_groups, load_pool, normalise
from check2.data.splits import build_splits, load_splits


@pytest.fixture(scope="module")
def pool():
    return load_pool()


@pytest.fixture(scope="module")
def pairs(pool):
    return all_pairs(pool)


@pytest.fixture(scope="module")
def drawn(pool, pairs):
    return build_splits(pool, pairs)


def test_the_draw_repeats_and_matches_the_saved_file(pool, pairs, drawn):
    again = build_splits(pool, pairs)
    assert again.heldout == drawn.heldout
    assert again.folds == drawn.folds
    assert asdict(load_splits()) == asdict(drawn)


def test_the_draw_does_not_depend_on_reasons(pool, pairs, drawn):
    without = [replace(p, reason=None) for p in pairs]
    assert asdict(build_splits(pool, without)) == asdict(drawn)


def test_the_saved_second_part_is_used_as_the_file_gives_it(drawn):
    second = json.loads(paths.TEST_SECOND_PART.read_text(encoding="utf-8"))
    assert drawn.test_second == sorted(second["wrong_first"] + second["right_first"])
    assert "college_physics/93" in drawn.test_second


def test_twins_of_test_questions_are_in_no_split(pool, drawn):
    good = complying(pool)
    named_twins = ["college_physics/18", "college_physics/41", "astronomy/48", "college_physics/93"]
    texts = {normalise(good[q].question) for q in named_twins}
    expected = sorted(q for q in good if q not in named_twins and normalise(good[q].question) in texts)
    assert len(expected) == 4
    assert drawn.in_no_split == expected


def test_no_duplicate_group_straddles_two_splits_or_two_folds(pool, drawn):
    where = {}
    for name in ("test_first", "test_second", "heldout", "train", "in_no_split"):
        for qid in getattr(drawn, name):
            assert qid not in where
            where[qid] = name
    fold_of = {qid: k for k, fold in enumerate(drawn.folds) for qid in fold}
    for group in duplicate_groups(pool):
        homes = {where[q] for q in group}
        assert len(homes - {"in_no_split"}) == 1, group
        if homes == {"train"}:
            assert len({fold_of[q] for q in group}) == 1, group


def test_heldout_and_folds_follow_the_stated_draw(pool, pairs, drawn):
    """The draw recomputed here from the specification's sentences, independently of the code."""
    good = complying(pool)
    test = set(drawn.test_first) | set(drawn.test_second)
    rest = [q for q in sorted(good) if q not in test and q not in drawn.in_no_split]
    unit_of = {q: (q,) for q in rest}
    for g in duplicate_groups(pool):
        members = tuple(q for q in g if q in unit_of)
        for q in members:
            unit_of[q] = members
    units = sorted(set(unit_of.values()))
    with_pair = {p.qid for p in pairs}

    def stratum(u):
        q = good[u[0]]
        return 0 if q.a1 != q.gold else (1 if q.qid in with_pair else 2)

    rng = random.Random(2027)
    held = []
    for s in range(3):
        members = sorted(u for u in units if stratum(u) == s)
        held += rng.sample(members, round(0.1 * len(members)))
    assert drawn.heldout == sorted(q for u in held for q in u)
    train_units = [u for u in units if u not in set(held)]
    order = random.Random(2031).sample(sorted(train_units), len(train_units))
    folds = [sorted(q for u in order[k::5] for q in u) for k in range(5)]
    assert drawn.folds == folds
