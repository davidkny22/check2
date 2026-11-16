"""Writer and review job files, checking and joining reasons, and applying reviews."""
import random

import pytest

from check2.data.expansion.jobs import review_jobs, review_sample, writer_jobs
from check2.data.expansion.keep import keep_questions
from check2.data.expansion.reasons import join_reasons, reason_faults
from check2.data.expansion.reviews import apply_reviews, without_removed


def _q(n, key="A", subject="s", question=None):
    return {"qid": f"expansion/{subject}/{n}", "subject": subject, "question": question or f"Q{n}?",
            "choices": [f"o{n}{i}" for i in range(4)], "key": key}


def _pairs(qs):
    return [{"qid": q["qid"], "subject": q["subject"], "target": t, "key_target": t == q["key"], "a1": None}
            for q in qs for t in (q["key"], "D")]


def test_writer_jobs_come_in_files_of_150_in_the_existing_form():
    qs = [_q(n) for n in range(100)]
    files = writer_jobs(qs, _pairs(qs))
    assert [len(f) for f in files] == [150, 50]
    job = files[0][0]
    assert set(job) == {"job_id", "question", "options", "argue_for"}
    assert job["job_id"] == "expansion/s/0:A" and job["options"]["B"] == "o01"


def test_review_jobs_sample_200_with_seed_2042_and_carry_the_key():
    qs = [_q(n, "B") for n in range(300)]
    pairs = [{**p, "reason": "r"} for p in _pairs(qs)]
    sample = review_sample(pairs)
    want = random.Random(2042).sample(sorted(pairs, key=lambda p: (p["qid"], p["target"])), 200)
    assert sample == want
    files = review_jobs(qs, pairs)
    assert sum(len(f) for f in files) == 200
    assert set(files[0][0]) == {"job_id", "question", "options", "keyed_option", "argue_for", "reason"}
    assert files[0][0]["keyed_option"] == "B"
    with pytest.raises(ValueError, match="no reason"):
        review_jobs(qs, [{**pairs[0], "reason": None}] + pairs[1:])


def test_reason_faults_catch_missing_repeated_unknown_and_wrong_length():
    jobs = [{"job_id": "a"}, {"job_id": "b"}, {"job_id": "c"}]
    ok = " ".join(["word"] * 25)
    assert reason_faults(jobs, {"reasons": [{"job_id": j, "reason": ok} for j in "abc"]}) == []
    faults = reason_faults(jobs, {"reasons": [{"job_id": "a", "reason": ok}, {"job_id": "a", "reason": ok},
                                              {"job_id": "x", "reason": "too short"}]})
    assert any("twice" in f for f in faults)
    assert any("missing" in f for f in faults)
    assert any("unknown" in f for f in faults)
    assert any(f.startswith("x: 2 words") for f in faults)
    assert reason_faults(jobs[:1], {"reasons": [{"job_id": "a", "reason": " ".join(["w"] * 41)}]}) == ["a: 41 words"]


def test_reasons_join_back_to_their_pairs():
    qs = [_q(0, "B")]
    pairs = _pairs(qs)
    joined = join_reasons(pairs, [{"writer": "opus", "reasons": [{"job_id": "expansion/s/0:B", "reason": "because"}]}])
    assert [(p["target"], p["reason"], p["writer"]) for p in joined] == [("B", "because", "opus"), ("D", None, None)]


def test_reviews_remove_contradicted_pairs_and_count_the_share():
    entries = [{"job_id": "q/1:A", "keyed_option": "A", "argue_for": "A"},
               {"job_id": "q/1:C", "keyed_option": "A", "argue_for": "C"},
               {"job_id": "q/2:B", "keyed_option": "B", "argue_for": "B"},
               {"job_id": "q/2:D", "keyed_option": "B", "argue_for": "D"}]
    reviews = [{"job_id": "q/1:A", "sound": "unsound", "supports_own_option": True, "flaw": "x", "plausible": ""},
               {"job_id": "q/1:C", "sound": "sound", "supports_own_option": True, "flaw": "", "plausible": "plausible"},
               {"job_id": "q/2:B", "sound": "unclear", "supports_own_option": True, "flaw": "", "plausible": ""},
               {"job_id": "q/2:D", "sound": "unsound", "supports_own_option": False, "flaw": "y", "plausible": "weak"}]
    removed, counts = apply_reviews(entries, reviews)
    assert [(r["qid"], r["target"]) for r in removed] == [("q/1", "A"), ("q/1", "C")]
    assert counts["contradicted"] == 2 and counts["contradicted_share"] == 0.5
    assert counts["does_not_support_own_option"] == 1
    pairs = [{"qid": "q/1", "target": "A"}, {"qid": "q/1", "target": "C"}, {"qid": "q/2", "target": "B"}]
    assert without_removed(pairs, removed) == [{"qid": "q/2", "target": "B"}]
    with pytest.raises(ValueError, match="no review"):
        apply_reviews(entries, reviews[:3])


def test_keep_drops_a_repeat_of_an_earlier_subject_and_applies_curation():
    gen = {"beta": [_q(0, "A", "beta", "Shared question?"), _q(1, "A", "beta", "Beta question?")],
           "alpha": [_q(0, "A", "alpha", "shared  question?"), _q(1, "B", "alpha", "Alpha question?")]}
    cur = {s: {"answered_by": "sonnet", "answers": [
        {"qid": q["qid"], "answer": q["key"], "problem": "none", "confidence": "high", "note": ""} for q in qs]}
        for s, qs in gen.items()}
    questions, counts = keep_questions(gen, cur)
    assert [q["qid"] for q in questions] == ["expansion/alpha/0", "expansion/alpha/1", "expansion/beta/1"]
    assert counts["beta"]["dropped_repeat_of_earlier_subject"] == 1
    assert counts["alpha"]["kept"] == 2 and counts["beta"]["kept"] == 1
    assert set(questions[0]) == {"qid", "subject", "question", "choices", "key", "options_reordered"}
