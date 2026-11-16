"""Build the input for reply_format_compare.py from the test questions and their reasons."""
import json

P = json.load(open("measurements/protocol/first_answers.json"))
qs = {q["qid"]: q for q in P["questions"]}
plan = json.load(open("measurements/dataset/plan.json"))
top = {r["job_id"]: r["reason"].strip()
       for r in json.load(open("measurements/dataset/jobs/reasons_test_topup.json"))["reasons"]}
pairs, used = [], set()
for e in plan["test_existing"]:
    pairs.append(dict(qid=e["qid"], target=e["target"], reason=e["reason"]))
    used.add(e["qid"])
for j in plan["jobs"]:
    if j["split"] == "test" and j["job_id"] in top:
        pairs.append(dict(qid=j["qid"], target=j["target"], reason=top[j["job_id"]]))
        used.add(j["qid"])
out = {"questions": {q: {k: qs[q][k] for k in ("question", "choices", "gold", "a1", "free_text")} for q in used},
       "pairs": pairs}
json.dump(out, open("measurements/protocol/reply_format_input.json", "w"))
valid = sum(p["target"] == qs[p["qid"]]["gold"] for p in pairs)
print("test questions:", len(used), "pairs:", len(pairs), "right reasons:", valid, "wrong reasons:", len(pairs) - valid)
