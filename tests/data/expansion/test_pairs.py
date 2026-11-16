"""The second pair: never the key, never a complying wrong first answer, one seeded draw."""
import random

from check2.data.expansion.pairs import build_pairs, second_targets


def _q(n, key):
    return {"qid": f"expansion/s/{n}", "subject": "s", "question": f"Q{n}?", "choices": ["w", "x", "y", "z"],
            "key": key}


def test_the_second_target_is_never_the_key_or_a_complying_wrong_first_answer():
    qs = [_q(n, "ABCD"[n % 4]) for n in range(200)]
    first = {q["qid"]: {"qid": q["qid"], "a1": "ABCD"[(n + 1) % 4], "complies": n % 3 != 0}
             for n, q in enumerate(qs)}
    second = second_targets(qs, first)
    for q in qs:
        assert second[q["qid"]] != q["key"]
        if first[q["qid"]]["complies"]:
            assert second[q["qid"]] != first[q["qid"]]["a1"]


def test_a_non_complying_or_missing_first_answer_excludes_only_the_key():
    qs = [_q(n, "A") for n in range(300)]
    first = {q["qid"]: {"qid": q["qid"], "a1": "B", "complies": False} for q in qs[:150]}
    second = second_targets(qs, first)
    assert set(second.values()) == {"B", "C", "D"}


def test_the_draw_follows_sorted_ids_with_one_generator():
    qs = [_q(10, "A"), _q(2, "B")]
    first = {"expansion/s/2": {"qid": "expansion/s/2", "a1": "C", "complies": True}}
    rng = random.Random(2041)
    want_10 = rng.sample(["B", "C", "D"], 1)[0]
    want_2 = rng.sample(["A", "D"], 1)[0]
    assert second_targets(qs, first) == {"expansion/s/10": want_10, "expansion/s/2": want_2}


def test_each_question_has_its_key_pair_and_one_other():
    qs = [_q(0, "C")]
    pairs = build_pairs(qs, {})
    assert [p["key_target"] for p in pairs].count(True) == 1
    assert len(pairs) == 2 and {p["target"] for p in pairs} >= {"C"}
    assert [p["target"] for p in pairs] == sorted(p["target"] for p in pairs)
