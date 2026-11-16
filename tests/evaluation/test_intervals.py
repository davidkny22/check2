"""Intervals resampled by question id, against brute-force loops written in the tests."""

import math
import random

import numpy as np
import pytest

from check2.evaluation.intervals import (
    Interval,
    interval,
    paired_difference,
    resamples,
    standard_error,
)


def mean_value(records):
    return sum(r["v"] for r in records) / len(records)


def by_qid(records):
    grouped = {}
    for r in records:
        grouped.setdefault(r["qid"], []).append(r)
    return grouped


def clustered(question_values, per_question):
    """Each question has `per_question` identical episodes carrying the question's value."""
    return by_qid(
        [{"qid": q, "v": v, "k": k} for q, v in question_values.items() for k in range(per_question)]
    )


def spec_draws(ids, n=2000, seed=2033):
    """The draws as the specification words them: one generator, `choices` of the sorted ids."""
    rng = random.Random(seed)
    ordered = sorted(ids)
    return [rng.choices(ordered, k=len(ordered)) for _ in range(n)]


def test_resamples_are_the_specified_draws():
    ids = ["q3", "q1", "q2", "q10", "q7"]
    assert list(resamples(ids)) == spec_draws(ids)
    assert list(resamples(ids, n=5, seed=11)) == spec_draws(ids, n=5, seed=11)


def test_resamples_count_length_and_membership():
    ids = [f"q{i}" for i in range(30)]
    draws = list(resamples(ids, n=50))
    assert len(draws) == 50
    assert all(len(d) == 30 and set(d) <= set(ids) for d in draws)
    # Drawn with replacement: across 50 draws some id appears twice in a draw.
    assert any(len(set(d)) < 30 for d in draws)


def test_resamples_do_not_depend_on_the_order_of_the_ids_given():
    ids = [f"q{i}" for i in range(12)]
    assert list(resamples(ids, n=20)) == list(resamples(list(reversed(ids)), n=20))


def test_different_seeds_give_different_draws():
    ids = [f"q{i}" for i in range(12)]
    assert list(resamples(ids, n=5, seed=1)) != list(resamples(ids, n=5, seed=2))


def test_the_statistic_sees_every_record_of_each_drawn_question_as_often_as_it_was_drawn():
    records = by_qid(
        [{"qid": "a", "v": 1}, {"qid": "a", "v": 2}, {"qid": "b", "v": 3}, {"qid": "c", "v": 4},
         {"qid": "c", "v": 5}, {"qid": "c", "v": 6}]
    )
    seen = []

    def spy(recs):
        seen.append([(r["qid"], r["v"]) for r in recs])
        return float(len(recs))

    interval(spy, records, n=300)
    expected = []
    for draw in spec_draws(list(records), n=300):
        batch = []
        for qid in draw:
            batch.extend((r["qid"], r["v"]) for r in records[qid])
        expected.append(batch)
    # The first call is the estimate over all records, in sorted question order.
    assert seen[0] == [("a", 1), ("a", 2), ("b", 3), ("c", 4), ("c", 5), ("c", 6)]
    assert seen[1:] == expected


def test_the_same_seed_gives_the_same_interval_twice():
    records = clustered({f"q{i}": (i * 7) % 5 / 4 for i in range(20)}, 3)
    first = interval(mean_value, records, n=500, seed=5)
    second = interval(mean_value, records, n=500, seed=5)
    assert tuple(first) == tuple(second)
    assert first.skipped == second.skipped == 0
    other = interval(mean_value, records, n=500, seed=6)
    assert tuple(other) != tuple(first)


def test_interval_unpacks_as_estimate_low_high_and_carries_the_skip_count():
    records = clustered({"q1": 0.0, "q2": 1.0, "q3": 1.0}, 2)
    result = interval(mean_value, records, n=200)
    estimate, low, high = result
    assert isinstance(result, Interval)
    assert (estimate, low, high) == (result.estimate, result.low, result.high)
    assert estimate == pytest.approx(2 / 3)
    assert low <= estimate <= high
    assert result.skipped == 0


def test_interval_is_the_percentile_interval_of_the_resampled_statistics():
    values = {f"q{i}": float(i % 4) for i in range(25)}
    records = clustered(values, 2)
    stats = []
    for draw in spec_draws(list(records), n=400, seed=9):
        stats.append(np.mean([values[q] for q in draw]))
    low, high = np.percentile(stats, [2.5, 97.5])
    _, got_low, got_high = interval(mean_value, records, n=400, seed=9)
    assert got_low == pytest.approx(low, abs=1e-12)
    assert got_high == pytest.approx(high, abs=1e-12)
    _, got_low, got_high = interval(mean_value, records, n=400, seed=9, level=0.8)
    low, high = np.percentile(stats, [10, 90])
    assert got_low == pytest.approx(low, abs=1e-12)
    assert got_high == pytest.approx(high, abs=1e-12)


def test_resampling_question_ids_is_wider_than_resampling_episodes_by_the_computed_factor():
    # Ten questions, five identical episodes each. The statistic is a mean, so its
    # resampling variance is the population variance of what is drawn over the count
    # drawn: question values over 10 when ids are drawn, over 50 when episodes are drawn.
    # The widths of the two intervals therefore stand as the square root of five.
    question_values = {f"q{i}": v for i, v in enumerate([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])}
    per_question = 5
    records = clustered(question_values, per_question)
    flat = [r for group in records.values() for r in group]
    values = np.array([question_values[q] for q in sorted(question_values)])
    question_se = math.sqrt(values.var() / len(values))
    episode_se = math.sqrt(values.var() / (len(values) * per_question))
    assert question_se / episode_se == pytest.approx(math.sqrt(per_question))

    n = 2000
    _, low, high = interval(mean_value, records, n=n)
    # The episodes-resampled interval, by a brute-force loop over episodes.
    rng = random.Random(2033)
    stats = [np.mean([r["v"] for r in rng.choices(flat, k=len(flat))]) for _ in range(n)]
    episode_low, episode_high = np.percentile(stats, [2.5, 97.5])

    factor = (high - low) / (episode_high - episode_low)
    assert factor == pytest.approx(math.sqrt(per_question), rel=0.1)
    assert standard_error(mean_value, records, n=n) == pytest.approx(question_se, rel=0.06)
    assert np.std(stats, ddof=1) == pytest.approx(episode_se, rel=0.06)


def test_standard_error_is_the_standard_deviation_over_the_resamples():
    values = {f"q{i}": float((i * 3) % 7) for i in range(15)}
    records = clustered(values, 2)
    stats = [np.mean([values[q] for q in draw]) for draw in spec_draws(list(records), n=300, seed=4)]
    assert standard_error(mean_value, records, n=300, seed=4) == pytest.approx(np.std(stats, ddof=1), abs=1e-12)


def test_standard_error_with_fewer_than_two_defined_resamples_is_refused():
    records = clustered({"q1": 1.0, "q2": 2.0}, 1)
    with pytest.raises(ValueError):
        standard_error(lambda recs: None, records, n=10)


def test_resamples_where_the_statistic_is_undefined_are_skipped_and_counted():
    # Only question q1 carries records of kind x. The statistic is the mean over kind x,
    # undefined on any resample that does not draw q1.
    records = by_qid(
        [{"qid": "q1", "kind": "x", "v": 1.0}, {"qid": "q1", "kind": "x", "v": 0.0},
         {"qid": "q2", "kind": "y", "v": 5.0}, {"qid": "q3", "kind": "y", "v": 7.0}]
    )

    def mean_of_x(recs):
        xs = [r["v"] for r in recs if r["kind"] == "x"]
        return sum(xs) / len(xs) if xs else None

    draws = spec_draws(list(records), n=500, seed=3)
    undefined = sum("q1" not in draw for draw in draws)
    defined = [mean_of_x([r for q in draw for r in records[q]]) for draw in draws if "q1" in draw]
    estimate, low, high = result = interval(mean_of_x, records, n=500, seed=3)
    assert 0 < undefined < 500
    assert result.skipped == undefined
    assert estimate == 0.5
    expected_low, expected_high = np.percentile(defined, [2.5, 97.5])
    assert low == pytest.approx(expected_low, abs=1e-12)
    assert high == pytest.approx(expected_high, abs=1e-12)


def test_a_statistic_undefined_everywhere_gives_no_bounds():
    records = clustered({"q1": 1.0, "q2": 2.0}, 1)
    result = interval(lambda recs: None, records, n=25)
    assert tuple(result) == (None, None, None)
    assert result.skipped == 25


def test_paired_difference_of_a_run_with_itself_is_exactly_zero_with_zero_width():
    records = clustered({f"q{i}": (i * 5) % 3 / 2 for i in range(18)}, 3)
    result = paired_difference(mean_value, records, records, n=300)
    assert tuple(result) == (0.0, 0.0, 0.0)
    assert result.skipped == 0


def test_paired_difference_draws_the_same_ids_for_both_runs():
    run_a = by_qid([{"qid": f"q{i}", "run": "a", "v": float(i)} for i in range(9)])
    run_b = by_qid([{"qid": f"q{i}", "run": "b", "v": float(2 * i)} for i in range(9)])
    calls = []

    def spy(recs):
        calls.append((recs[0]["run"], [r["qid"] for r in recs]))
        return 0.0

    paired_difference(spy, run_a, run_b, n=100)
    # Two calls for the estimate, then for each resample the first run, then the second.
    assert calls[0][0] == "a" and calls[1][0] == "b"
    resampled = calls[2:]
    draws = spec_draws(list(run_a), n=100)
    assert len(resampled) == 200
    for i, draw in enumerate(draws):
        run_first, ids_first = resampled[2 * i]
        run_second, ids_second = resampled[2 * i + 1]
        assert (run_first, run_second) == ("a", "b")
        assert ids_first == ids_second == draw


def test_paired_difference_matches_a_brute_force_loop():
    run_a = clustered({f"q{i}": (i % 4) / 3 for i in range(14)}, 2)
    run_b = clustered({f"q{i}": ((i * 3) % 5) / 4 for i in range(14)}, 2)
    diffs = []
    for draw in spec_draws(list(run_a), n=400, seed=8):
        diffs.append(
            np.mean([run_a[q][0]["v"] for q in draw]) - np.mean([run_b[q][0]["v"] for q in draw])
        )
    low, high = np.percentile(diffs, [2.5, 97.5])
    estimate, got_low, got_high = paired_difference(mean_value, run_a, run_b, n=400, seed=8)
    assert estimate == pytest.approx(
        np.mean([run_a[q][0]["v"] for q in run_a]) - np.mean([run_b[q][0]["v"] for q in run_b])
    )
    assert got_low == pytest.approx(low, abs=1e-12)
    assert got_high == pytest.approx(high, abs=1e-12)


def test_pairing_removes_the_variation_the_two_runs_share():
    # The second run is the first plus 0.1 on every question. Drawn together, the
    # difference is 0.1 in every resample; drawn apart it would vary.
    base = {f"q{i}": float(i % 6) for i in range(30)}
    run_a = clustered({q: v + 0.1 for q, v in base.items()}, 2)
    run_b = clustered(base, 2)
    estimate, low, high = paired_difference(mean_value, run_a, run_b, n=300)
    assert estimate == pytest.approx(0.1)
    assert low == pytest.approx(0.1)
    assert high == pytest.approx(0.1)


def test_paired_difference_skips_resamples_where_either_run_is_undefined():
    run_a = by_qid([{"qid": "q1", "kind": "x", "v": 1.0}, {"qid": "q2", "kind": "y", "v": 0.0},
                    {"qid": "q3", "kind": "y", "v": 0.0}])
    run_b = by_qid([{"qid": "q1", "kind": "y", "v": 1.0}, {"qid": "q2", "kind": "x", "v": 0.5},
                    {"qid": "q3", "kind": "y", "v": 0.0}])

    def mean_of_x(recs):
        xs = [r["v"] for r in recs if r["kind"] == "x"]
        return sum(xs) / len(xs) if xs else None

    draws = spec_draws(list(run_a), n=300, seed=2)
    undefined = sum(not ("q1" in d and "q2" in d) for d in draws)
    result = paired_difference(mean_of_x, run_a, run_b, n=300, seed=2)
    assert 0 < undefined < 300
    assert result.skipped == undefined
    # Run b has no x records among all of its records except q2, run a only q1: both defined overall.
    assert result.estimate == pytest.approx(1.0 - 0.5)


def test_paired_difference_refuses_runs_with_different_questions():
    run_a = clustered({"q1": 1.0, "q2": 2.0}, 1)
    run_b = clustered({"q1": 1.0, "q3": 2.0}, 1)
    with pytest.raises(ValueError):
        paired_difference(mean_value, run_a, run_b, n=10)


def test_level_must_lie_between_zero_and_one():
    records = clustered({"q1": 1.0, "q2": 2.0}, 1)
    with pytest.raises(ValueError):
        interval(mean_value, records, n=10, level=1.0)
