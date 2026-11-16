"""The pinned files give the specification's counts, read directly from the files."""
import json

from check2.data import paths
from check2.data.pairs import all_pairs
from check2.data.pool import complying, duplicate_groups, load_pool, normalise


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_pool_counts():
    pool = load_pool()
    good = complying(pool)
    assert len(pool) == 2081
    assert len(good) == 2054
    assert sum(q.a1 != q.gold for q in good.values()) == 617


def test_duplicate_groups():
    pool = load_pool()
    good = complying(pool)
    groups = duplicate_groups(pool)
    assert len(groups) == 24
    assert sum(len(g) for g in groups) == 48
    assert sum(len({good[i].choices for i in g}) == 1 for g in groups) == 12
    for g in groups:
        assert len({normalise(good[i].question) for i in g}) == 1


def test_test_parts_as_the_files_give_them():
    plan = _read(paths.PLAN)
    second = _read(paths.TEST_SECOND_PART)
    assert len(set(plan["test_questions"])) == 301
    assert len(set(second["wrong_first"]) | set(second["right_first"])) == 240


def test_pair_counts():
    pool = load_pool()
    good = complying(pool)
    plan = _read(paths.PLAN)
    second = _read(paths.TEST_SECOND_PART)
    first_ids = set(plan["test_questions"])
    second_ids = set(second["wrong_first"]) | set(second["right_first"])
    pairs = [p for p in all_pairs(pool) if p.writer == "opus"]
    first = [p for p in pairs if p.qid in first_ids]
    sec = [p for p in pairs if p.qid in second_ids]
    other = [p for p in pairs if p.qid not in first_ids]
    assert (len(first), sum(p.target == good[p.qid].gold for p in first)) == (437, 136)
    assert (len(sec), sum(p.target == good[p.qid].gold for p in sec)) == (360, 120)
    assert (len(other), len({p.qid for p in other})) == (1562, 1081)


def test_pairs_follow_the_rule_per_question():
    pool = load_pool()
    good = complying(pool)
    targets = {}
    for p in all_pairs(pool):
        if p.writer == "opus":
            targets.setdefault(p.qid, []).append(p.target)
    for qid, ts in targets.items():
        q = good[qid]
        assert q.a1 not in ts
        if q.a1 != q.gold:
            assert len(ts) == 2 and q.gold in ts
        else:
            assert len(ts) == 1 and ts[0] != q.gold
