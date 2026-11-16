"""The two pairs of every kept question: its key, and one other option drawn by seed."""
import random

from check2.data.expansion.paths import SECOND_PAIR_SEED
from check2.data.paths import LETTERS


def second_targets(questions: list[dict], first_answers: dict[str, dict]) -> dict[str, str]:
    """For each question, one option that is neither the key nor a complying wrong first answer.

    One `random.Random(2041)` serves every question in id order; each draw is
    `rng.sample(sorted(candidates), 1)`. A question with no first answer, or whose first
    answer does not comply, excludes only its key.
    """
    rng = random.Random(SECOND_PAIR_SEED)
    out = {}
    for q in sorted(questions, key=lambda q: q["qid"]):
        excluded = {q["key"]}
        first = first_answers.get(q["qid"])
        if first is not None and first["complies"]:
            excluded.add(first["a1"])
        out[q["qid"]] = rng.sample(sorted(set(LETTERS) - excluded), 1)[0]
    return out


def build_pairs(questions: list[dict], first_answers: dict[str, dict]) -> list[dict]:
    """Every pair, sorted by question id then target, with whether its target is the key."""
    second = second_targets(questions, first_answers)
    pairs = []
    for q in questions:
        first = first_answers.get(q["qid"])
        for target in (q["key"], second[q["qid"]]):
            pairs.append({"qid": q["qid"], "subject": q["subject"], "target": target,
                          "key_target": target == q["key"],
                          "a1": first["a1"] if first is not None and first["complies"] else None})
    return sorted(pairs, key=lambda p: (p["qid"], p["target"]))
