"""The learning curve's nested subsets and its verdict on made-up outputs."""
import random

from check2.reader.curve import FRACTIONS, nested_subsets, summarise


def test_subsets_are_nested_seeded_prefixes_of_round_q_n():
    units = [(f"u/{i}",) for i in range(10)] + [("g/1", "g/2")]
    subsets = nested_subsets(units)
    assert [len(s) for s in subsets] == [round(q * 11) for q in FRACTIONS] == [3, 6, 8, 11]
    for small, large in zip(subsets, subsets[1:]):
        assert large[:len(small)] == small
    order = random.Random(2032).sample(sorted(units), len(units))
    assert subsets[-1] == order
    assert nested_subsets(list(reversed(units))) == subsets


def _train(n=40):
    episodes = []
    for q in range(n):
        for fmt in ("snap", "short"):
            for target, situation in (("A", "user_right"), ("C", "recheck")):
                episodes.append({"episode_id": f"q/{q}:reasoned:{fmt}:{target}", "qid": f"q/{q}",
                                 "form": "reasoned", "format": fmt, "a1": "B", "gold": "A",
                                 "situation": situation})
    return episodes


def _p(train, strength, seed):
    rng = random.Random(seed)
    return {e["episode_id"]: [rng.random() + (strength if e["situation"] == "user_right" else 0), 0, 0]
            for e in train}


def test_a_curve_still_rising_at_the_last_step_is_short_of_data():
    train = _train()
    rising = [_p(train, s, 7) for s in (0.0, 0.2, 0.4, 1.0)]
    result = summarise(rising, train)
    assert [s["fraction"] for s in result["sizes"]] == list(FRACTIONS)
    assert result["steps"][-1]["interval"][0] > 0 and result["short_of_data"]


def test_a_flat_curve_is_not_short_of_data():
    train = _train()
    flat = [_p(train, 0.5, 7)] * 4
    result = summarise(flat, train)
    assert result["steps"][-1]["difference"] == 0 and not result["short_of_data"]
