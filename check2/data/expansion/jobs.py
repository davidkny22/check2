"""Writer job files of 150 pairs, and review job files of a seeded sample of pairs.

Both are in the forms the reviewed set's jobs use: a writer job is `job_id`, `question`,
`options` and `argue_for`; a review job adds `keyed_option` and the `reason`. A new review
job file never replaces one already written, so every earlier judgement keeps its job.
"""
import random

from check2.data.expansion import files, paths
from check2.data.expansion.paths import JOB_FILE_SIZE, REVIEW_SAMPLE_SIZE, REVIEW_SEED
from check2.data.paths import LETTERS

REVIEW_FILE_SIZE = 50


def job_id(pair: dict) -> str:
    return f"{pair['qid']}:{pair['target']}"


def _options(q: dict) -> dict[str, str]:
    return dict(zip(LETTERS, q["choices"]))


def writer_jobs(questions: list[dict], pairs: list[dict]) -> list[list[dict]]:
    """The pairs, in their order, as writer jobs in files of 150."""
    by_id = {q["qid"]: q for q in questions}
    jobs = [{"job_id": job_id(p), "question": by_id[p["qid"]]["question"],
             "options": _options(by_id[p["qid"]]), "argue_for": p["target"]} for p in pairs]
    return [jobs[i:i + JOB_FILE_SIZE] for i in range(0, len(jobs), JOB_FILE_SIZE)]


def write_writer_jobs(questions: list[dict], pairs: list[dict]) -> dict[str, int]:
    out = {}
    for n, chunk in enumerate(writer_jobs(questions, pairs), start=1):
        path = paths.JOBS / f"writer_{n:03d}.json"
        files.write_json(path, chunk)
        out[path.name] = len(chunk)
    return out


def review_sample(pairs: list[dict]) -> list[dict]:
    """200 pairs drawn with `random.Random(2042)` from every pair sorted by id, then target."""
    candidates = sorted(pairs, key=lambda p: (p["qid"], p["target"]))
    return random.Random(REVIEW_SEED).sample(candidates, min(REVIEW_SAMPLE_SIZE, len(candidates)))


def review_job(question: dict, pair: dict) -> dict:
    """One pair and its reason as a review job, with the key the writer did not see."""
    return {"job_id": job_id(pair), "question": question["question"], "options": _options(question),
            "keyed_option": question["key"], "argue_for": pair["target"], "reason": pair["reason"]}


def _require_reasons(pairs: list[dict]) -> None:
    missing = [job_id(p) for p in pairs if not p.get("reason")]
    if missing:
        raise ValueError(f"{len(missing)} pairs have no reason; review jobs are made once every reason is in")


def review_jobs_of(questions: list[dict], pairs: list[dict]) -> list[dict]:
    """Every pair, in its order, as a review job. Each pair must have its reason."""
    _require_reasons(pairs)
    by_id = {q["qid"]: q for q in questions}
    return [review_job(by_id[p["qid"]], p) for p in pairs]


def sample_jobs(questions: list[dict], pairs: list[dict]) -> list[dict]:
    """The sampled pairs, in draw order, as review jobs. Every pair given must have its
    reason: the sample is drawn over the pairs with a reason, not over all pairs."""
    _require_reasons(pairs)
    return review_jobs_of(questions, review_sample(pairs))


def review_jobs(questions: list[dict], pairs: list[dict]) -> list[list[dict]]:
    """The sampled pairs, in draw order, as review jobs in files of 50."""
    jobs = sample_jobs(questions, pairs)
    return [jobs[i:i + REVIEW_FILE_SIZE] for i in range(0, len(jobs), REVIEW_FILE_SIZE)]


def _file_number(name: str) -> int:
    return int(name.removesuffix(".json").rsplit("_", 1)[1])


def write_review_jobs(jobs: list[dict]) -> dict[str, int]:
    """Write `jobs` in files of 50, numbered after the review job files already there, which
    are never overwritten."""
    taken = [_file_number(p.name) for p in paths.REVIEW_JOBS.glob("review_*.json")]
    start = max(taken, default=0) + 1
    out = {}
    for n, i in enumerate(range(0, len(jobs), REVIEW_FILE_SIZE), start=start):
        path = paths.REVIEW_JOBS / f"review_{n}.json"
        files.write_json(path, jobs[i:i + REVIEW_FILE_SIZE])
        out[path.name] = len(jobs[i:i + REVIEW_FILE_SIZE])
    return out
