"""Checking a curator's answers against its input, and keeping the questions they confirm.

A question is kept only where the curator's answer equals the key and it records no problem.
"""
from check2.data.expansion import files, paths
from check2.data.paths import LETTERS
from check2.data.pool import normalise

PROBLEMS = ("none", "no_correct_option", "more_than_one_correct", "unclear_question")
CONFIDENCES = ("high", "medium", "low")
# Written for a question the earlier college mathematics curation does not cover.
NOT_ANSWERED = "not_answered"


def check_curation(questions: list[dict], curation: dict) -> None:
    """Refuse a curation file that does not answer exactly `questions`, in their order."""
    if curation.get("answered_by") != "sonnet":
        raise ValueError(f"answered_by is {curation.get('answered_by')!r}, not 'sonnet'")
    got = [a["qid"] for a in curation["answers"]]
    want = [q["qid"] for q in questions]
    if got != want:
        extra, missing = sorted(set(got) - set(want)), sorted(set(want) - set(got))
        raise ValueError(f"curation ids differ from the input: {len(got)} answers for {len(want)} "
                         f"questions, {len(extra)} unknown, {len(missing)} missing, order "
                         f"{'differs' if not extra and not missing else 'not compared'}")
    for a in curation["answers"]:
        if a["problem"] == NOT_ANSWERED and a["answer"] is None:
            continue
        if a["answer"] not in LETTERS or len(a["answer"]) != 1:
            raise ValueError(f"{a['qid']}: answer {a['answer']!r} is not a letter A to D")
        if a["problem"] not in PROBLEMS:
            raise ValueError(f"{a['qid']}: problem {a['problem']!r} is not one of {PROBLEMS}")
        if a["confidence"] not in CONFIDENCES:
            raise ValueError(f"{a['qid']}: confidence {a['confidence']!r} is not one of {CONFIDENCES}")


def apply_curation(questions: list[dict], curation: dict) -> tuple[list[dict], dict]:
    """The questions the curator confirms, in input order, and the counts of the check."""
    check_curation(questions, curation)
    answers = {a["qid"]: a for a in curation["answers"]}
    kept, agree = [], 0
    problems = dict.fromkeys(PROBLEMS + (NOT_ANSWERED,), 0)
    for q in questions:
        a = answers[q["qid"]]
        problems[a["problem"]] += 1
        agree += a["answer"] == q["key"]
        if a["answer"] == q["key"] and a["problem"] == "none":
            kept.append(q)
    counts = {"curated": len(questions), "answer_equals_key": agree,
              "problems": {k: v for k, v in problems.items() if v or k != NOT_ANSWERED},
              "kept_by_curator": len(kept)}
    return kept, counts


def earlier_curation(questions: list[dict]) -> dict:
    """The earlier Sonnet answers to college mathematics, keyed by this set's question ids.

    The earlier answers are keyed by another file's ids, so they are joined by normalised
    question text, and an answer is used only where that file's four options equal these in
    order, since its letter names a position. A question with no such answer gets an entry
    with no answer, which the keeping rule never keeps.
    """
    earlier = {e["qid"]: e for e in files.read_jsonl(paths.EARLIER_CURATED_QUESTIONS)}
    by_text: dict[str, dict] = {}
    for a in files.read_json(paths.EARLIER_CURATION)["answers"]:
        e = earlier[a["qid"]]
        by_text.setdefault(normalise(e["question"]), {**a, "choices": [c.strip() for c in e["choices"]]})
    answers = []
    for q in questions:
        a = by_text.get(normalise(q["question"]))
        if a is not None and a["choices"] == q["choices"]:
            answers.append({"qid": q["qid"], "answer": a["answer"], "problem": a["problem"],
                            "confidence": a["confidence"], "note": a.get("note", "")})
        else:
            answers.append({"qid": q["qid"], "answer": None, "problem": NOT_ANSWERED,
                            "confidence": None, "note": ""})
    return {"answered_by": "sonnet", "answers": answers}
