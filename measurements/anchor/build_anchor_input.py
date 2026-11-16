"""Draw the reserve questions, then build the input for anchor_check.py.

Step 1 writes measurements/dataset/reserve_questions.json.
Step 2 writes measurements/anchor/anchor_input.json: the questions and the measurement
pairs (training-pool jobs with a written reason, outside test, reserve, and any
duplicate group that holds a test id). Prompts are rendered on the VM, where the
tokenizer is.

Run from the repository root.
"""
import json, random
from collections import Counter, defaultdict

L = "ABCD"
P = json.load(open("measurements/protocol/first_answers.json"))
plan = json.load(open("measurements/dataset/plan.json"))
test_ids = set(plan["test_questions"])

# ---------- step 1: the reserve ----------
pool = {q["qid"]: q for q in P["questions"] if q["complies"]}
print("pool (complies):", len(pool), "of", len(P["questions"]))
by_text = defaultdict(list)
for qid, q in pool.items():
    by_text[" ".join(q["question"].lower().split())].append(qid)
groups = [sorted(v) for v in by_text.values() if len(v) > 1]
dup_ids = sorted(i for g in groups for i in g)
print("duplicate groups:", len(groups), "ids in them:", len(dup_ids))

job_of = {j["job_id"]: j for j in plan["jobs"]}
batch_jobs = {}
for b in range(1, 7):
    for j in json.load(open(f"measurements/dataset/jobs/train_batch{b}.json")):
        pj = job_of[j["job_id"]]
        assert j["argue_for"] == L[pj["target"]], j["job_id"]
        batch_jobs[j["job_id"]] = dict(batch=b, qid=pj["qid"], target=pj["target"], split=pj["split"])
print("jobs in train_batch1..6:", len(batch_jobs), Counter(v["split"] for v in batch_jobs.values()))
has_job = {v["qid"] for v in batch_jobs.values()}

eligible = sorted(q for q in pool if q not in test_ids and q not in set(dup_ids) and q in has_job)
wrong_first = sorted(q for q in eligible if pool[q]["a1"] != pool[q]["gold"])
right_first = sorted(q for q in eligible if pool[q]["a1"] == pool[q]["gold"])
print("pool not test:", sum(q not in test_ids for q in pool),
      "| not test, not duplicate:", sum(q not in test_ids and q not in set(dup_ids) for q in pool),
      "| eligible (also has a job):", len(eligible), "wrong-first:", len(wrong_first), "right-first:", len(right_first))
rng = random.Random(2026)
reserve_wrong = rng.sample(sorted(wrong_first), 120)
reserve_right = rng.sample(sorted(right_first), 120)
reserve = {
    "rule": ("Pool = questions in first_answers.json with complies true; ids whose question text, lowercased with "
             "whitespace collapsed, is shared with another pool id form duplicate groups; eligible = pool ids not in "
             "plan.json test_questions, not in any duplicate group, and with at least one job in train_batch1..6, "
             "split by a1 != gold (wrong-first) or a1 == gold (right-first); with rng = random.Random(2026), "
             "reserve_wrong = rng.sample(sorted(wrong_first_ids), 120), then reserve_right = "
             "rng.sample(sorted(right_first_ids), 120) with the same rng."),
    "seed": 2026,
    "duplicate_group_ids": dup_ids,
    "wrong_first": sorted(reserve_wrong),
    "right_first": sorted(reserve_right),
}
json.dump(reserve, open("measurements/dataset/reserve_questions.json", "w"), indent=1)
reserve_ids = set(reserve_wrong) | set(reserve_right)
assert len(reserve_ids) == 240 and not (reserve_ids & test_ids)
print("reserve written: 120 wrong-first, 120 right-first")

# ---------- step 2: measurement pairs ----------
dup_with_test = {i for g in groups if any(x in test_ids for x in g) for i in g}
print("ids in duplicate groups that contain a test id:", len(dup_with_test))
reasons = {}
for b in (1, 2, 3, 4, 6):
    for r in json.load(open(f"measurements/dataset/jobs/reasons_train_batch{b}.json"))["reasons"]:
        assert r["job_id"] in batch_jobs and batch_jobs[r["job_id"]]["batch"] == b
        reasons[r["job_id"]] = r["reason"].strip()
print("written reasons in batches 1,2,3,4,6:", len(reasons))
drop = Counter()
pairs = []
for jid in sorted(reasons):
    v = batch_jobs[jid]
    qid = v["qid"]
    if qid in test_ids:
        drop["test id"] += 1; continue
    if qid in reserve_ids:
        drop["reserve id"] += 1; continue
    if qid in dup_with_test:
        drop["duplicate group with a test id"] += 1; continue
    if qid not in pool:
        drop["not complying"] += 1; continue
    q = pool[qid]
    t = v["target"]
    assert t != q["a1"], jid
    sit = "user_right" if t == q["gold"] else ("hold" if q["a1"] == q["gold"] else "recheck")
    pairs.append(dict(job_id=jid, qid=qid, target=t, reason=reasons[jid], situation=sit))
qids = sorted({p["qid"] for p in pairs})
print("pairs dropped:", dict(drop))
print("measurement pairs:", len(pairs), "questions:", len(qids), Counter(p["situation"] for p in pairs))
wf = [q for q in qids if pool[q]["a1"] != pool[q]["gold"]]
print("wrong-first questions:", len(wf), "right-first:", len(qids) - len(wf),
      "| wrong-first with both pairs:", sum(sum(p["qid"] == q for p in pairs) == 2 for q in wf))
out = {"questions": {q: {k: pool[q][k] for k in ("question", "choices", "gold", "a1", "free_text", "p1")} for q in qids},
       "pairs": pairs}
json.dump(out, open("measurements/anchor/anchor_input.json", "w"))
print("episodes: named/reasoned x snap/short =", 4 * len(pairs), "| bare snap =", len(qids))
