"""The split isolation check is green on the built files and red on each kind of leak."""
import copy
from dataclasses import replace

import pytest

from check2.data.pool import complying, load_pool
from check2.data.splits import load_splits
from check2.gates.split_isolation import check, check_folds, load_episode_files, plant


@pytest.fixture(scope="module")
def pool():
    return load_pool()


@pytest.fixture(scope="module")
def splits():
    return load_splits()


@pytest.fixture(scope="module")
def episode_files():
    return load_episode_files()


def test_green_on_the_built_files(pool, splits, episode_files):
    result = check(splits=splits, pool=pool, episode_files=episode_files, targets=[], cache_index={})
    assert result.green, result.findings


def test_red_when_a_test_text_is_planted_in_train(pool, splits, episode_files):
    files = copy.deepcopy(episode_files)
    plant(files, splits, complying(pool))
    result = check(splits=splits, pool=pool, episode_files=files, targets=[], cache_index={})
    assert not result.green
    assert any("belongs to ['test']" in f for f in result.findings)


def test_red_when_an_id_is_in_two_splits(pool, splits, episode_files):
    leaked = replace(splits, train=sorted(splits.train + [splits.test_first[0]]))
    result = check(splits=leaked, pool=pool, episode_files={}, targets=[], cache_index={})
    assert not result.green


def test_red_when_a_twin_of_a_test_question_is_in_train(pool, splits):
    twin = splits.in_no_split[0]
    leaked = replace(splits, train=sorted(splits.train + [twin]), in_no_split=splits.in_no_split[1:],
                     folds=[sorted(splits.folds[0] + [twin])] + splits.folds[1:])
    result = check(splits=leaked, pool=pool, episode_files={}, targets=[], cache_index={})
    assert any("normalised text" in f for f in result.findings)


def test_red_when_a_duplicate_group_straddles_two_folds(pool, splits):
    good = complying(pool)
    by_text = {}
    for qid in splits.train:
        by_text.setdefault("".join(good[qid].question.lower().split()), []).append(qid)
    group = next(ids for ids in by_text.values() if len(ids) > 1)
    folds = [[q for q in f if q not in group] for f in splits.folds]
    folds[0].append(group[0])
    folds[1].append(group[1])
    assert not check_folds(folds, pool).green
    assert check_folds(splits.folds, pool).green


def test_red_when_a_target_or_cache_entry_leaks(pool, splits, episode_files):
    test_episode = episode_files["test.jsonl"][0]["episode_id"]
    train_episode = episode_files["train.jsonl"][0]["episode_id"]
    assert check(splits=splits, pool=pool, episode_files=episode_files,
                 targets=[{"episode_id": train_episode}], cache_index={train_episode: [0, 0, 5]}).green
    assert not check(splits=splits, pool=pool, episode_files=episode_files,
                     targets=[{"episode_id": test_episode}], cache_index={}).green
    assert not check(splits=splits, pool=pool, episode_files=episode_files, targets=[],
                     cache_index={"nowhere/1:bare:snap:-": [0, 0, 5]}).green
