"""The review sample drawn over the final pairs: which judgements carry, new review files,
and what is removed."""
import random

import pytest

from check2.data.expansion import jobs as jobs_module
from check2.data.expansion.jobs import sample_jobs, write_review_jobs
from check2.data.expansion.reviews import carried_judgements, sample_outcome


def _job(jid, keyed="A", argue="A", reason="r"):
    return {"job_id": jid, "question": "Q?", "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
            "keyed_option": keyed, "argue_for": argue, "reason": reason}


def _review(jid, sound):
    return {"job_id": jid, "supports_own_option": True, "sound": sound, "flaw": "", "plausible": ""}


def test_the_sample_is_drawn_over_the_pairs_given_sorted_by_id_then_target():
    qs = [{"qid": f"e/s/{n}", "question": f"Q{n}?", "choices": ["a", "b", "c", "d"], "key": "C"}
          for n in range(150)]
    pairs = [{"qid": q["qid"], "target": t, "reason": f"{q['qid']} {t}"} for q in reversed(qs) for t in ("C", "A")]
    by_hand = sorted([(p["qid"], p["target"]) for p in pairs])
    want = [f"{qid}:{t}" for qid, t in random.Random(2042).sample(by_hand, 200)]
    sample = sample_jobs(qs, pairs)
    assert [j["job_id"] for j in sample] == want
    assert sample[0]["reason"] == sample[0]["job_id"].replace(":", " ")
    assert sample[0]["keyed_option"] == "C"


def test_a_judgement_carries_only_to_the_same_pair_with_the_same_reason_text():
    judged = [(_job("q/1:A", reason="first"), _review("q/1:A", "sound")),
              (_job("q/2:B", argue="B", reason="old text"), _review("q/2:B", "sound"))]
    jobs = [_job("q/1:A", reason="first"), _job("q/2:B", argue="B", reason="new text"), _job("q/3:A")]
    carried, to_review = carried_judgements(jobs, judged)
    assert carried == {"q/1:A": _review("q/1:A", "sound")}
    assert [j["job_id"] for j in to_review] == ["q/2:B", "q/3:A"]


def test_a_pair_judged_twice_with_the_same_reason_is_refused():
    judged = [(_job("q/1:A"), _review("q/1:A", "sound")), (_job("q/1:A"), _review("q/1:A", "unsound"))]
    with pytest.raises(ValueError, match="judged twice"):
        carried_judgements([_job("q/1:A")], judged)


def test_new_review_files_are_numbered_after_the_existing_ones_and_never_replace_them(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs_module.paths, "REVIEW_JOBS", tmp_path)
    for n in (1, 2, 10):
        (tmp_path / f"review_{n}.json").write_text("[]", encoding="utf-8")
    written = write_review_jobs([_job(f"q/{n}:A") for n in range(51)])
    assert written == {"review_11.json": 50, "review_12.json": 1}
    assert (tmp_path / "review_10.json").read_text(encoding="utf-8") == "[]"
    assert write_review_jobs([]) == {}


def test_the_share_is_the_samples_and_earlier_contradicted_pairs_stay_removed():
    # Every pair with a reason: q/1 to q/5. The first review judged q/1:B (contradicted: a
    # reason for another option rated sound) and q/2:A (sound, for the key). The sample is
    # q/2:A, q/3:A and q/4:B; q/3:A is judged unsound for the key, so contradicted.
    jobs = [_job("q/1:B", argue="B"), _job("q/2:A"), _job("q/3:A"), _job("q/4:B", argue="B"), _job("q/5:A")]
    first = [(_job("q/1:B", argue="B"), _review("q/1:B", "sound")), (_job("q/2:A"), _review("q/2:A", "sound"))]
    new = [(_job("q/3:A"), _review("q/3:A", "unsound")), (_job("q/4:B", argue="B"), _review("q/4:B", "unsound"))]
    sample = [jobs[1], jobs[2], jobs[3]]
    sample_removed, removed, counts = sample_outcome(jobs, sample, first + new)
    assert [(r["qid"], r["target"]) for r in sample_removed] == [("q/3", "A")]
    assert counts["reviewed"] == 3 and counts["contradicted"] == 1
    assert counts["contradicted_share"] == pytest.approx(1 / 3)
    assert [(r["qid"], r["target"], r["in_review_sample"]) for r in removed] == [("q/1", "B", False),
                                                                                 ("q/3", "A", True)]


def test_a_sampled_pair_whose_reason_changed_needs_a_new_review():
    jobs = [_job("q/1:A", reason="written again")]
    judged = [(_job("q/1:A", reason="earlier"), _review("q/1:A", "unsound"))]
    with pytest.raises(ValueError, match="no judgement of their current reason"):
        sample_outcome(jobs, jobs, judged)
    # A removed pair is never one whose contradicted judgement was of an earlier reason.
    sample_removed, removed, _ = sample_outcome(jobs, jobs, judged + [(jobs[0], _review("q/1:A", "sound"))])
    assert removed == [] and sample_removed == []
