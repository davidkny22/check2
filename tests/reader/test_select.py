"""The read-position rule and the variant choice on hand-built score tables."""
from check2.reader.select import best_position, choose_position, choose_variant


def test_earliest_position_within_one_standard_error_of_the_best():
    scores = {0: 0.70, 1: 0.78, 2: 0.81, 3: 0.84, 4: 0.86, 5: 0.85, 6: 0.80, 7: 0.79}
    assert best_position(scores) == 4
    # best 0.86, one standard error 0.03: 0.83 and above qualify; tail[3] is the earliest
    assert choose_position(scores, 0.03) == 3
    # a wider error admits tail[1]
    assert choose_position(scores, 0.09) == 1
    # a zero error leaves the best itself
    assert choose_position(scores, 0.0) == 4


def test_a_score_exactly_one_standard_error_below_the_best_qualifies():
    assert choose_position({0: 0.5, 1: 0.75, 2: 1.0}, 0.25) == 1


def test_ties_go_to_the_earliest():
    assert best_position({0: 0.6, 1: 0.9, 2: 0.9}) == 1
    assert choose_variant({"six_layers": 0.8, "linear_probe": 0.8, "attention_only": 0.7}) == "linear_probe"
    assert choose_variant({"six_layers": 0.8, "deltanet_only": 0.85}) == "deltanet_only"
