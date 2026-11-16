"""Applying the soundness reviews of the sampled pairs: which judgements still hold for a
pair's current reason, the share contradicted, and removal.

A pair is contradicted when its reason argues for the key and is rated unsound, or argues
for another option and is rated sound; the rule is the reviewed set's.
"""
from check2.data.expansion.jobs import job_id
from check2.data.jobs import contradicted


def _reviews_by_job(review_jobs: list[dict], reviews: list[dict]) -> dict[str, dict]:
    """Each job's one review, keyed by job id; a review of an unknown job, a job reviewed
    twice or a job with no review is refused."""
    entries = {e["job_id"]: e for e in review_jobs}
    seen: dict[str, dict] = {}
    for r in reviews:
        if r["job_id"] not in entries:
            raise ValueError(f"review of unknown job {r['job_id']}")
        if r["job_id"] in seen:
            raise ValueError(f"job {r['job_id']} is reviewed twice")
        seen[r["job_id"]] = r
    missing = sorted(set(entries) - set(seen))
    if missing:
        raise ValueError(f"{len(missing)} sampled pairs have no review")
    return seen


def judged_jobs(review_jobs: list[dict], reviews: list[dict]) -> list[tuple[dict, dict]]:
    """One review file's judgements, each as (the job as the reviewer saw it, its review)."""
    seen = _reviews_by_job(review_jobs, reviews)
    return [(e, seen[e["job_id"]]) for e in review_jobs]


def carried_judgements(jobs: list[dict], judged: list[tuple[dict, dict]]) -> tuple[dict[str, dict], list[dict]]:
    """Split `jobs` into those already judged and those still to review.

    A judgement carries to a job only when the reviewer saw that same job: the same pair,
    question, options, key and reason text. A pair whose reason was written again keeps no
    judgement of its earlier reason. Returns the carried reviews by job id, and the jobs, in
    their order, that have none."""
    by_id: dict[str, list[tuple[dict, dict]]] = {}
    for seen, r in judged:
        by_id.setdefault(seen["job_id"], []).append((seen, r))
    carried: dict[str, dict] = {}
    for job in jobs:
        matches = [r for seen, r in by_id.get(job["job_id"], []) if seen == job]
        if len(matches) > 1:
            raise ValueError(f"job {job['job_id']} is judged twice with the same reason")
        if matches:
            carried[job["job_id"]] = matches[0]
    return carried, [j for j in jobs if j["job_id"] not in carried]


def apply_reviews(review_jobs: list[dict], reviews: list[dict]) -> tuple[list[dict], dict]:
    """The contradicted pairs as (qid, target) records, and the review's counts."""
    entries = {e["job_id"]: e for e in review_jobs}
    seen = _reviews_by_job(review_jobs, reviews)
    removed = []
    for jid in sorted(seen):
        e, r = entries[jid], seen[jid]
        if contradicted(e, r):
            qid, target = jid.rsplit(":", 1)
            removed.append({"qid": qid, "target": target, "key_target": e["argue_for"] == e["keyed_option"],
                            "sound": r["sound"], "flaw": r.get("flaw", "")})
    key_reviews = [seen[j] for j in seen if entries[j]["argue_for"] == entries[j]["keyed_option"]]
    other_reviews = [seen[j] for j in seen if entries[j]["argue_for"] != entries[j]["keyed_option"]]
    counts = {
        "reviewed": len(seen),
        "key_target": {v: sum(r["sound"] == v for r in key_reviews) for v in ("sound", "unsound", "unclear")},
        "other_target": {v: sum(r["sound"] == v for r in other_reviews) for v in ("sound", "unsound", "unclear")},
        "other_plausible": {v: sum(r.get("plausible") == v for r in other_reviews)
                            for v in ("plausible", "weak", "nonsense")},
        "does_not_support_own_option": sum(not r["supports_own_option"] for r in seen.values()),
        "contradicted": len(removed),
        "contradicted_share": len(removed) / len(seen) if seen else None,
    }
    return removed, counts


def sample_outcome(jobs: list[dict], sample: list[dict],
                   judged: list[tuple[dict, dict]]) -> tuple[list[dict], list[dict], dict]:
    """The review's result over the sample, and every pair to remove.

    `jobs` is every pair with a reason, as review jobs; `sample` is the sampled jobs. Every
    sampled job must carry a judgement of its current reason. Returns the contradicted pairs
    of the sample, every pair whose current reason any review found contradicted (each marked
    with whether it is in the sample), and the counts over the sample alone."""
    carried, _ = carried_judgements(jobs, judged)
    unjudged = [j["job_id"] for j in sample if j["job_id"] not in carried]
    if unjudged:
        raise ValueError(f"{len(unjudged)} sampled pairs have no judgement of their current reason")
    sample_removed, counts = apply_reviews(sample, [carried[j["job_id"]] for j in sample])
    judged_now = [j for j in jobs if j["job_id"] in carried]
    removed, _ = apply_reviews(judged_now, [carried[j["job_id"]] for j in judged_now])
    in_sample = {j["job_id"] for j in sample}
    removed = [{**r, "in_review_sample": f"{r['qid']}:{r['target']}" in in_sample,
                "review_file": carried[f"{r['qid']}:{r['target']}"].get("review_file")} for r in removed]
    return sample_removed, removed, counts


def without_removed(pairs: list[dict], removed: list[dict]) -> list[dict]:
    gone = {f"{r['qid']}:{r['target']}" for r in removed}
    return [p for p in pairs if job_id(p) not in gone]
