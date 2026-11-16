"""The seeded option order, the key moved with its option, and options that name others."""
import random

from check2.data.expansion.options import order_options, refers_to_options


def _q(n, key, choices):
    return {"qid": f"expansion/s/{n}", "subject": "s", "question": f"Q{n}?", "choices": list(choices), "key": key}


def test_options_that_refer_to_other_options():
    assert refers_to_options(["1", "2", "3", "All of the above"])
    assert refers_to_options(["1", "2", "none of these", "3"])
    assert refers_to_options(["1", "2", "Both A and C", "3"])
    assert refers_to_options(["1", "2", "(A) or (B)", "3"])
    assert refers_to_options(["1", "2", "A, B and D", "3"])
    assert refers_to_options(["1", "2", "Option B is wrong", "3"])
    assert not refers_to_options(["Paris", "Lyon", "Nice", "Lille"])
    assert not refers_to_options(["(i) and (ii) only", "(iii) only", "(iv) only", "(i) only"])
    assert not refers_to_options(["A dog", "a cat", "Vitamin C", "Type A blood"])


def test_the_key_follows_its_option_text():
    qs = [_q(n, "ABCD"[n % 4], [f"opt{n}-{i}" for i in range(4)]) for n in range(40)]
    before = {q["qid"]: q["choices"]["ABCD".index(q["key"])] for q in qs}
    after = order_options(qs)
    for q in after:
        assert q["choices"]["ABCD".index(q["key"])] == before[q["qid"]]
        assert sorted(q["choices"]) == sorted(next(x for x in qs if x["qid"] == q["qid"])["choices"])
    assert sum(q["choices"] != next(x for x in qs if x["qid"] == q["qid"])["choices"] for q in after) > 20


def test_the_order_is_one_seeded_draw_over_sorted_ids_skipping_marked_questions():
    qs = [_q(2, "A", ["a", "b", "c", "d"]), _q(1, "B", ["e", "f", "g", "all of the above"]),
          _q(10, "C", ["h", "i", "j", "k"])]
    after = order_options(qs)
    assert [q["qid"] for q in after] == ["expansion/s/1", "expansion/s/10", "expansion/s/2"]
    assert after[0]["choices"] == ["e", "f", "g", "all of the above"]
    assert after[0]["key"] == "B" and after[0]["options_reordered"] is False
    rng = random.Random(2040)
    first, second = rng.sample(range(4), 4), rng.sample(range(4), 4)
    assert after[1]["choices"] == [["h", "i", "j", "k"][i] for i in first]
    assert after[2]["choices"] == [["a", "b", "c", "d"][i] for i in second]
    assert after[1]["options_reordered"] is True
