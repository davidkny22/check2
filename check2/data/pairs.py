"""Pairs (question, target) with their written reasons, and the situation of an episode."""
import json
import random
from dataclasses import dataclass
from pathlib import Path

from check2.data import paths
from check2.data.paths import LETTERS
from check2.data.pool import Question, complying

TRAINING_BATCHES = (1, 2, 3, 4, 5, 6)
MEASURED_REASON_BATCHES = (1, 2, 3, 4, 6)
WRITTEN_REASON_FILES = ("5a", "5b", "5c")
CROSSWRITER_SEED = 2028
CROSSWRITER_COUNT = 300


@dataclass(frozen=True)
class Pair:
    qid: str
    target: str
    reason: str | None
    writer: str
    job_id: str | None

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.qid, self.target, self.writer)


def situation(form: str, a1: str, gold: str, target: str | None) -> str:
    """The situation of an episode, from the table of the conversation's form."""
    if form == "bare":
        return "user_right" if a1 != gold else "hold"
    if form in ("named", "reasoned"):
        if target is None:
            raise ValueError(f"a {form} episode needs a target")
        if target == gold:
            return "user_right"
        return "hold" if a1 == gold else "recheck"
    raise ValueError(f"no situation is defined for the form {form!r}")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _reasons_by_job(path: Path, job_ids: list[str]) -> dict[str, str]:
    """A reasons file's reasons by job id, refusing a missing, doubled or unknown job id."""
    entries = _read_json(path)["reasons"]
    seen = [e["job_id"] for e in entries]
    doubled = sorted({j for j in seen if seen.count(j) > 1})
    missing = sorted(set(job_ids) - set(seen))
    unknown = sorted(set(seen) - set(job_ids))
    if doubled or missing or unknown:
        raise ValueError(f"{path.name}: doubled {doubled[:5]}, missing {missing[:5]}, "
                         f"unknown {unknown[:5]} job ids")
    return {e["job_id"]: e["reason"] for e in entries}


def _check_job_text(job: dict, q: Question) -> None:
    options = [job["options"][letter] for letter in LETTERS]
    if job["question"] != q.question or options != list(q.choices):
        raise ValueError(f"job {job['job_id']} does not carry the text of {q.qid}")


def _batch_pairs(pool: dict[str, Question], plan_jobs: dict[str, dict]) -> list[Pair]:
    pairs = []
    for batch in TRAINING_BATCHES:
        jobs = _read_json(paths.MEASURED_JOBS / f"train_batch{batch}.json")
        reasons: dict[str, str] = {}
        if batch in MEASURED_REASON_BATCHES:
            reasons = _reasons_by_job(paths.MEASURED_JOBS / f"reasons_train_batch{batch}.json",
                                      [j["job_id"] for j in jobs])
        else:
            for part in WRITTEN_REASON_FILES:
                reason_file = paths.REASONS / f"reasons_train_batch{part}.json"
                if reason_file.exists():
                    part_jobs = _read_json(paths.JOBS / f"train_batch{part}.json")
                    reasons.update(_reasons_by_job(reason_file, [j["job_id"] for j in part_jobs]))
            unknown = sorted(set(reasons) - {j["job_id"] for j in jobs})
            if unknown:
                raise ValueError(f"reasons for jobs outside batch {batch}: {unknown[:5]}")
        for job in jobs:
            pairs.append(_job_pair(job, plan_jobs, pool, reasons.get(job["job_id"])))
    return [p for p in pairs if p is not None]


def _job_pair(job: dict, plan_jobs: dict[str, dict], pool: dict[str, Question],
              reason: str | None) -> Pair | None:
    if job["job_id"] not in plan_jobs:
        raise ValueError(f"job {job['job_id']} has no entry in the plan")
    planned = plan_jobs[job["job_id"]]
    target = LETTERS[planned["target"]]
    if job["argue_for"] != target:
        raise ValueError(f"job {job['job_id']} argues for {job['argue_for']}, the plan says {target}")
    q = pool.get(planned["qid"])
    if q is None:
        return None
    _check_job_text(job, q)
    return Pair(qid=q.qid, target=target, reason=reason, writer="opus", job_id=job["job_id"])


def _first_part_pairs(pool: dict[str, Question], plan: dict, plan_jobs: dict[str, dict]) -> list[Pair]:
    pairs = [Pair(qid=e["qid"], target=LETTERS[e["target"]], reason=e["reason"], writer="opus",
                  job_id=None)
             for e in plan["test_existing"] if e["qid"] in pool]
    topup = _read_json(paths.MEASURED_JOBS / "test_topup.json")
    reasons = _reasons_by_job(paths.MEASURED_JOBS / "reasons_test_topup.json",
                              [j["job_id"] for j in topup])
    for job in topup:
        pair = _job_pair(job, plan_jobs, pool, reasons[job["job_id"]])
        if pair is not None:
            pairs.append(pair)
    return pairs


def first_part_pairs(pairs: list[Pair], test_first: list[str]) -> list[Pair]:
    """The writer's pairs on the first part of `test`, sorted by question id, then target."""
    first = set(test_first)
    return sorted((p for p in pairs if p.qid in first and p.writer == "opus"),
                  key=lambda p: (p.qid, p.target))


def crosswriter_draw(first_pairs: list[Pair]) -> list[tuple[str, Pair]]:
    """The cross-writer pairs with their job ids: a seeded sample of the first part's pairs."""
    candidates = sorted(first_pairs, key=lambda p: (p.qid, p.target))
    drawn = random.Random(CROSSWRITER_SEED).sample(candidates, CROSSWRITER_COUNT)
    drawn.sort(key=lambda p: (p.qid, p.target))
    return [(f"c_{i:03d}", p) for i, p in enumerate(drawn)]


def _crosswriter_pairs(opus_pairs: list[Pair], plan: dict) -> list[Pair]:
    if not paths.CROSSWRITER_REASONS.exists():
        return []
    drawn = crosswriter_draw(first_part_pairs(opus_pairs, plan["test_questions"]))
    reasons = _reasons_by_job(paths.CROSSWRITER_REASONS, [job_id for job_id, _ in drawn])
    return [Pair(qid=p.qid, target=p.target, reason=reasons[job_id], writer="sonnet",
                 job_id=job_id) for job_id, p in drawn]


def removed_pair_keys() -> set[tuple[str, str, str]]:
    """The (qid, target, writer) of every pair the soundness review removed."""
    if not paths.REMOVED_PAIRS.exists():
        return set()
    return {(r["qid"], r["target"], r["writer"]) for r in _read_json(paths.REMOVED_PAIRS)["removed"]}


def load_pairs(pool: list[Question]) -> list[Pair]:
    """Every pair of complying questions, sorted by question id, target and writer.

    A pair whose reason has not been written yet carries `reason = None`. A pair the
    soundness review removed is left out.
    """
    removed = removed_pair_keys()
    return [p for p in all_pairs(pool) if p.key not in removed]


def all_pairs(pool: list[Question]) -> list[Pair]:
    """Every pair the files define, including any the soundness review removed."""
    by_id = complying(pool)
    plan = _read_json(paths.PLAN)
    plan_jobs = {j["job_id"]: j for j in plan["jobs"]}
    opus = _first_part_pairs(by_id, plan, plan_jobs) + _batch_pairs(by_id, plan_jobs)
    keys = [p.key for p in opus]
    if len(set(keys)) != len(keys):
        raise ValueError("a (question, target) pair appears twice among the jobs")
    for p in opus:
        if p.target == by_id[p.qid].a1:
            raise ValueError(f"pair {p.qid} targets the model's own first answer")
    pairs = opus + _crosswriter_pairs(opus, plan)
    order = {"opus": 0, "sonnet": 1}
    return sorted(pairs, key=lambda p: (p.qid, p.target, order[p.writer]))


def reason_sets(pairs: list[Pair]) -> dict[str, list[Pair]]:
    """The pairs with a reason, grouped by the file their reasons were written in."""
    part_of = {}
    for part in WRITTEN_REASON_FILES:
        job_file = paths.JOBS / f"train_batch{part}.json"
        if job_file.exists():
            part_of.update({j["job_id"]: f"train_batch{part}" for j in _read_json(job_file)})
    sets: dict[str, list[Pair]] = {}
    for p in pairs:
        if p.reason is None:
            continue
        if p.job_id is None:
            name = "test_existing"
        elif p.writer == "sonnet":
            name = "crosswriter"
        elif p.job_id.startswith("x_"):
            name = "test_topup"
        elif p.job_id in part_of:
            name = part_of[p.job_id]
        else:
            name = f"train_batch{p.job_id[1]}"
        sets.setdefault(name, []).append(p)
    return dict(sorted(sets.items()))


def pair_record(pair: Pair, q: Question) -> dict:
    """The line of `data/pairs.jsonl` for one pair."""
    return {"qid": pair.qid, "target": pair.target, "reason": pair.reason, "writer": pair.writer,
            "job_id": pair.job_id, "situation": situation("named", q.a1, q.gold, pair.target)}


def write_pairs(pairs: list[Pair], pool: list[Question], path: Path = paths.PAIRS) -> None:
    by_id = complying(pool)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for p in pairs:
            f.write(json.dumps(pair_record(p, by_id[p.qid]), ensure_ascii=False) + "\n")
