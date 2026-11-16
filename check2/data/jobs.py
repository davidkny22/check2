"""Writing jobs for the cross-writer reasons, review jobs, and applying the reviews."""
import json
import random
from pathlib import Path

from check2.data import paths
from check2.data.paths import LETTERS
from check2.data.pairs import Pair, crosswriter_draw, first_part_pairs
from check2.data.pool import Question, complying, load_pool
from check2.data.splits import Splits

REVIEW_FILE_SIZE = 200
REVIEW_SAMPLE_SEED = 2029
REVIEW_SAMPLE_SIZE = 200
REVIEW_SAMPLE_LIMIT = 0.02


def _options(q: Question) -> dict[str, str]:
    return dict(zip(LETTERS, q.choices))


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8",
                    newline="\n")


def crosswriter_jobs(pairs: list[Pair]) -> list[dict]:
    """The writer's jobs for the cross-writer reasons; written to `data/jobs/crosswriter.json`."""
    by_id = complying(load_pool())
    test_first = json.loads(paths.PLAN.read_text(encoding="utf-8"))["test_questions"]
    jobs = [{"job_id": job_id, "question": by_id[p.qid].question, "options": _options(by_id[p.qid]),
             "argue_for": p.target}
            for job_id, p in crosswriter_draw(first_part_pairs(pairs, test_first))]
    _write_json(paths.CROSSWRITER_JOBS, jobs)
    return jobs


def review_job_id(pair: Pair) -> str:
    return f"{pair.qid}:{pair.target}"


def _review_entry(pair: Pair, q: Question) -> dict:
    return {"job_id": review_job_id(pair), "question": q.question, "options": _options(q),
            "keyed_option": q.gold, "argue_for": pair.target, "reason": pair.reason}


def review_jobs(pool: list[Question], pairs: list[Pair], splits: Splits) -> dict:
    """Every `test` reason in files of at most 200, and a seeded sample of 200 other reasons."""
    missing = [p for p in pairs if p.reason is None]
    if missing:
        raise ValueError(f"{len(missing)} pairs have no reason yet; review jobs are made only "
                         f"once every reason is written")
    by_id = complying(pool)
    test, others = set(splits.test), set(splits.heldout) | set(splits.train)
    writer_pairs = sorted((p for p in pairs if p.writer == "opus"), key=lambda p: (p.qid, p.target))
    test_pairs = [p for p in writer_pairs if p.qid in test]
    files = {}
    for n, start in enumerate(range(0, len(test_pairs), REVIEW_FILE_SIZE), start=1):
        chunk = [_review_entry(p, by_id[p.qid]) for p in test_pairs[start:start + REVIEW_FILE_SIZE]]
        path = paths.JOBS / f"review_test_{n}.json"
        _write_json(path, chunk)
        files[path.name] = len(chunk)
    candidates = [p for p in writer_pairs if p.qid in others]
    sample = random.Random(REVIEW_SAMPLE_SEED).sample(candidates, REVIEW_SAMPLE_SIZE)
    _write_json(paths.JOBS / "review_sample.json", [_review_entry(p, by_id[p.qid]) for p in sample])
    files["review_sample.json"] = len(sample)
    return files


def contradicted(entry: dict, review: dict) -> bool:
    """A gold-target reason rated unsound, or a wrong-target reason rated sound."""
    if entry["argue_for"] == entry["keyed_option"]:
        return review["sound"] == "unsound"
    return review["sound"] == "sound"


def _count(values: list) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items()))


def apply_reviews() -> dict:
    """Read every review, list the contradicted pairs in `removed_pairs.json`, return the counts."""
    entries: dict[str, tuple[dict, str]] = {}
    for path in sorted(paths.JOBS.glob("review_*.json")):
        for e in json.loads(path.read_text(encoding="utf-8")):
            entries[e["job_id"]] = (e, path.name)
    reviews: dict[str, dict] = {}
    for path in sorted(paths.REVIEWS.glob("review_*.json")):
        for r in json.loads(path.read_text(encoding="utf-8"))["reviews"]:
            if r["job_id"] not in entries:
                raise ValueError(f"{path.name}: review of unknown job {r['job_id']}")
            if r["job_id"] in reviews:
                raise ValueError(f"{path.name}: job {r['job_id']} is reviewed twice")
            reviews[r["job_id"]] = r
    removed, sample_total, sample_bad = [], 0, 0
    for job_id in sorted(reviews):
        entry, source = entries[job_id]
        bad = contradicted(entry, reviews[job_id])
        if source == "review_sample.json":
            sample_total += 1
            sample_bad += bad
        if bad:
            qid, target = job_id.rsplit(":", 1)
            removed.append({"qid": qid, "target": target, "writer": "opus", "review_job_id": job_id,
                            "gold_target": entry["argue_for"] == entry["keyed_option"],
                            "sound": reviews[job_id]["sound"]})
    _write_json(paths.REMOVED_PAIRS, {"removed": removed})
    values = list(reviews.values())
    share = sample_bad / sample_total if sample_total else None
    return {
        "reviewed": len(values), "not_reviewed": len(entries) - len(values),
        "sound": _count([r["sound"] for r in values]),
        "supports_own_option": _count([r["supports_own_option"] for r in values]),
        "plausible": _count([r.get("plausible") for r in values
                             if entries[r["job_id"]][0]["argue_for"] != entries[r["job_id"]][0]["keyed_option"]]),
        "contradicted": len(removed),
        "unclear_kept": sum(r["sound"] == "unclear" for r in values),
        "sample_reviewed": sample_total, "sample_contradicted": sample_bad,
        "sample_contradicted_share": share,
        "sample_above_limit": None if share is None else share > REVIEW_SAMPLE_LIMIT,
    }
