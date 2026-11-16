"""Writing and review jobs, the contradiction rule, and the style score."""
import json
import random

import pytest

from check2.data import jobs as jobs_module
from check2.data import paths
from check2.data.jobs import apply_reviews, contradicted, review_jobs
from check2.data.pairs import Pair
from check2.data.pool import Question
from check2.data.splits import Splits
from check2.data.style import style_score


def _q(qid, gold="A", a1="A"):
    return Question(qid=qid, subject="s", question=f"Question {qid}?", choices=("w", "x", "y", "z"),
                    gold=gold, a1=a1, free_text=a1, complies=True, p1=(0.25,) * 4)


def test_crosswriter_jobs_draw_300_first_part_pairs():
    entries = json.loads(paths.CROSSWRITER_JOBS.read_text(encoding="utf-8"))
    assert len(entries) == 300
    assert len({e["job_id"] for e in entries}) == 300
    assert all(set(e) == {"job_id", "question", "options", "argue_for"} for e in entries)


def test_contradiction_rule():
    gold = {"argue_for": "A", "keyed_option": "A"}
    wrong = {"argue_for": "B", "keyed_option": "A"}
    assert contradicted(gold, {"sound": "unsound"})
    assert not contradicted(gold, {"sound": "sound"})
    assert not contradicted(gold, {"sound": "unclear"})
    assert contradicted(wrong, {"sound": "sound"})
    assert not contradicted(wrong, {"sound": "unsound"})
    assert not contradicted(wrong, {"sound": "unclear"})


def test_review_jobs_refuse_while_a_reason_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "JOBS", tmp_path)
    pool = [_q("s/1")]
    splits = Splits(["s/1"], [], [], [], [], [[], [], [], [], []])
    with pytest.raises(ValueError):
        review_jobs(pool, [Pair("s/1", "B", None, "opus", "t1_000")], splits)


def test_review_jobs_and_applying_reviews(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "JOBS", tmp_path / "jobs")
    monkeypatch.setattr(paths, "REVIEWS", tmp_path / "reviews")
    monkeypatch.setattr(paths, "REMOVED_PAIRS", tmp_path / "reviews" / "removed_pairs.json")
    monkeypatch.setattr(jobs_module, "REVIEW_SAMPLE_SIZE", 4)
    test_ids = [f"s/{i}" for i in range(250)]
    train_ids = [f"t/{i}" for i in range(10)]
    pool = [_q(q) for q in test_ids + train_ids]
    pairs = [Pair(q, "B", f"reason {q}", "opus", None) for q in test_ids + train_ids]
    splits = Splits(test_ids, [], [], train_ids, [], [train_ids, [], [], [], []])
    files = review_jobs(pool, pairs, splits)
    assert files == {"review_test_1.json": 200, "review_test_2.json": 50, "review_sample.json": 4}
    sample = json.loads((tmp_path / "jobs" / "review_sample.json").read_text(encoding="utf-8"))
    expected = random.Random(2029).sample(sorted(train_ids), 4)
    assert [e["job_id"] for e in sample] == [f"{q}:B" for q in expected]
    (tmp_path / "reviews").mkdir()
    reviews = [{"job_id": "s/0:B", "supports_own_option": True, "sound": "sound", "flaw": "",
                "plausible": "plausible"},
               {"job_id": "s/1:B", "supports_own_option": True, "sound": "unclear", "flaw": "",
                "plausible": "weak"},
               {"job_id": f"{expected[0]}:B", "supports_own_option": True, "sound": "unsound",
                "flaw": "x", "plausible": "weak"}]
    (tmp_path / "reviews" / "review_a.json").write_text(json.dumps({"reviews": reviews}), encoding="utf-8")
    counts = apply_reviews()
    removed = json.loads((tmp_path / "reviews" / "removed_pairs.json").read_text(encoding="utf-8"))
    assert [(r["qid"], r["target"]) for r in removed["removed"]] == [("s/0", "B")]
    assert counts["contradicted"] == 1 and counts["unclear_kept"] == 1
    assert counts["sample_reviewed"] == 1 and counts["sample_contradicted"] == 0


def test_style_score_separates_marked_text_and_not_unmarked_text():
    rng = random.Random(0)
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    groups = [f"q{i // 2}" for i in range(200)]
    gold = [i % 2 == 0 for i in range(200)]
    marked = [" ".join(rng.choices(words, k=12)) + (" indeed" if g else " however") for g in gold]
    assert style_score(marked, gold, groups) > 0.95
    plain = [" ".join(rng.choices(words, k=12)) for _ in gold]
    assert 0.3 < style_score(plain, gold, groups) < 0.7
