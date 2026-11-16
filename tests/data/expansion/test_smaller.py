"""The unwritten jobs split into smaller job files."""
import json

import pytest

from check2.data.expansion import smaller
from check2.data.expansion.smaller import split_jobs, write_split


def _jobs(subject, n, start=0):
    return [{"job_id": f"expansion/{subject}/{i}:A", "question": "Q?", "options": {}, "argue_for": "A"}
            for i in range(start, start + n)]


def test_the_split_puts_every_job_in_exactly_one_file_of_at_most_the_size():
    # 47 of "alpha", 12 of "beta", 25 of "gamma", 5 of "delta"; files of 30. Built by hand:
    # alpha fills one file of 30 and leaves 17; the remainders, largest first, are gamma 25,
    # alpha 17, beta 12, delta 5. gamma opens a file (25), alpha opens one (17), beta joins
    # alpha's (29), delta joins gamma's (30).
    jobs = _jobs("alpha", 47) + _jobs("beta", 12) + _jobs("gamma", 25) + _jobs("delta", 5)
    out = split_jobs(jobs, 30)
    ids = [j["job_id"] for f in out for j in f]
    assert sorted(ids) == sorted(j["job_id"] for j in jobs) and len(ids) == len(set(ids))
    assert [len(f) for f in out] == [30, 30, 29]
    assert [j["job_id"] for j in out[0]] == [f"expansion/alpha/{i}:A" for i in range(30)]
    assert {smaller.subject_of(j["job_id"]) for j in out[1]} == {"gamma", "delta"}
    assert [j["job_id"] for j in out[2]] == ([f"expansion/alpha/{i}:A" for i in range(30, 47)]
                                             + [f"expansion/beta/{i}:A" for i in range(12)])


def test_a_split_into_files_of_ten_keeps_job_contents_unchanged():
    jobs = _jobs("alpha", 3) + _jobs("beta", 9)
    out = split_jobs(jobs, 10)
    assert [len(f) for f in out] == [9, 3]
    assert sorted((j for f in out for j in f), key=lambda j: j["job_id"]) == sorted(jobs, key=lambda j: j["job_id"])


def test_written_files_are_new_and_never_replace_one(tmp_path, monkeypatch):
    monkeypatch.setattr(smaller.paths, "JOBS", tmp_path)
    made = write_split(_jobs("alpha", 12), 10, "small")
    assert made == {"small_001.json": 10, "small_002.json": 2}
    assert len(json.loads((tmp_path / "small_002.json").read_text(encoding="utf-8"))) == 2
    with pytest.raises(ValueError, match="exist already"):
        write_split(_jobs("beta", 2), 10, "small")
