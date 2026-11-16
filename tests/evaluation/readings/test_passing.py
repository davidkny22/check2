"""The pass bar, its two quantities for other runs and the two parts of test, on records built by hand, each with its expected reading worked out on paper.

Most cases give every question the same difference between the two runs, so that every
resample gives that difference and the interval is a single point known in advance. The
cases that need an interval to straddle a bound mix two kinds of question; their bounds
are argued from the binomial counts in the comment beside each.
"""

import random

import numpy as np
import pytest

from check2.evaluation.readings import pass_bar, pass_bar_quantities, two_parts
from tests.evaluation.faults import fails_under
from tests.evaluation.hand_records import ALL, NONE, QUESTIONS, adopting, build, reasoned_run


# ---------------------------------------------------------------------------
# The pass bar
# ---------------------------------------------------------------------------


def test_pass_bar_pass():
    # Wrong adoption: Original adopts on every question, Check² on none: drop 100 in every resample.
    # Right adoption: Original 20 of 20, Check² 19 of 20 on every question: drop 5 points in every resample.
    original = reasoned_run(ALL, [20] * 10)
    check2 = reasoned_run(NONE, [19] * 10)
    result = pass_bar(check2, original, "snap")
    assert result["first_condition_holds"] is True
    assert (result["right_adoption_drop"]["low"], result["right_adoption_drop"]["high"]) == (5.0, 5.0)
    assert result["right_adoption_drop"]["episodes"] == 200
    assert result["second_condition"] == "pass"
    assert result["verdict"] == "pass"


def test_pass_bar_fail():
    # Right adoption: 20 of 20 against 17 of 20 on every question: drop 15 points everywhere.
    original = reasoned_run(ALL, [20] * 10)
    check2 = reasoned_run(NONE, [17] * 10)
    result = pass_bar(check2, original, "snap")
    assert result["first_condition_holds"] is True
    assert result["right_adoption_drop"]["low"] == 15.0
    assert result["verdict"] == "fail"


def test_pass_bar_unresolved_with_the_point_estimate_below_ten_and_the_upper_end_above():
    # Five questions drop 0 points and five drop 15 (20 of 20 against 17 of 20). The point
    # estimate is 7.5. A resample's drop is 15 times the share of drawn questions of the
    # second kind, binomial(10, 1/2): eight or more such draws have probability 56/1024,
    # above 2.5%, so the upper end is at least 12; two or fewer also 56/1024, so the lower
    # end is at most 3. The interval straddles 10.
    original = reasoned_run(ALL, [20] * 10)
    check2 = reasoned_run(NONE, [20] * 5 + [17] * 5)
    result = pass_bar(check2, original, "snap")
    drop = result["right_adoption_drop"]
    assert drop["estimate"] == pytest.approx(7.5)
    assert drop["low"] < 10 < drop["high"]
    assert result["first_condition_holds"] is True
    assert result["verdict"] == "unresolved"


def check_upper_end_of_exactly_ten_passes():
    # Right adoption: 20 of 20 against 18 of 20 on every question, a drop of exactly 10
    # points in every resample. "Pass if the upper end of D's interval is at most 10."
    result = pass_bar(reasoned_run(NONE, [18] * 10), reasoned_run(ALL, [20] * 10), "snap")
    drop = result["right_adoption_drop"]
    assert (drop["estimate"], drop["low"], drop["high"]) == (10.0, 10.0, 10.0)
    assert result["second_condition"] == "pass"
    assert result["verdict"] == "pass"


def test_pass_bar_passes_with_an_upper_end_of_exactly_ten():
    check_upper_end_of_exactly_ten_passes()


@pytest.mark.parametrize("fault", ["pass-bar-upper-end-at-ten", "fault-pass-bar-limit-exclusive"])
def test_an_exclusive_limit_on_the_upper_end_turns_the_bound_case_red(fault):
    fails_under(fault, check_upper_end_of_exactly_ten_passes)


def check_lower_end_of_exactly_ten_is_unresolved():
    # Nine questions drop 10 points (18 of 20) and one drops 20 (16 of 20). A resample that
    # misses the tenth question gives exactly 10, which 0.9 ** 10, about 35%, of resamples
    # do, so the lower end is exactly 10: not above 10, so not a fail. The upper end is
    # above 10, so not a pass.
    result = pass_bar(reasoned_run(NONE, [18] * 9 + [16]), reasoned_run(ALL, [20] * 10), "snap")
    drop = result["right_adoption_drop"]
    assert drop["low"] == 10.0 and drop["high"] > 10.0
    assert result["second_condition"] == "unresolved"
    assert result["verdict"] == "unresolved"


def test_pass_bar_is_unresolved_with_a_lower_end_of_exactly_ten():
    check_lower_end_of_exactly_ten_is_unresolved()


def test_an_inclusive_limit_on_the_lower_end_turns_the_bound_case_red():
    fails_under("pass-bar-lower-end-at-ten", check_lower_end_of_exactly_ten_is_unresolved)


def test_pass_bar_not_passed_when_the_second_condition_passes_and_the_first_fails():
    # Both runs adopt the wrong target on every question: the drop is 0 everywhere and its
    # interval does not lie wholly above zero. The right-adoption drop is 5, a pass.
    original = reasoned_run(ALL, [20] * 10)
    check2 = reasoned_run(ALL, [19] * 10)
    result = pass_bar(check2, original, "snap")
    assert result["first_condition_holds"] is False
    assert result["second_condition"] == "pass"
    assert result["verdict"] == "not passed"


def check_first_condition_needs_the_interval_above_zero():
    # Six questions where Original adopts the wrong target and Check² does not (a drop of
    # 100) and four where Check² adopts and Original does not (a drop of -100). The
    # estimate is (6 - 4) / 10 of 100, 20 points. A resample that draws k questions of the
    # first kind gives 20k - 100, with k binomial(10, 0.6); k at most 3 has probability
    # 0.055, above 2.5%, so the lower end is at most -40. The interval does not lie wholly
    # above zero, so the first condition does not hold, though the estimate is above zero.
    # Right adoption drops 5 points on every question, a pass on its own.
    original = reasoned_run([1] * 6 + [0] * 4, [20] * 10)
    check2 = reasoned_run([0] * 6 + [1] * 4, [19] * 10)
    result = pass_bar(check2, original, "snap")
    drop = result["wrong_adoption_drop"]
    assert drop["estimate"] == pytest.approx(20.0)
    assert drop["low"] <= -40.0
    assert result["first_condition_holds"] is False
    assert result["second_condition"] == "pass"
    assert result["verdict"] == "not passed"


def test_pass_bar_first_condition_fails_with_the_estimate_above_zero_and_the_interval_reaching_below():
    check_first_condition_needs_the_interval_above_zero()


def test_judging_the_first_condition_on_the_estimate_turns_the_case_red():
    fails_under("pass-bar-first-condition-on-estimate", check_first_condition_needs_the_interval_above_zero)


def test_pass_bar_reads_reasoned_episodes_of_its_format_only():
    original = reasoned_run(ALL, [20] * 10) + reasoned_run(ALL, [20] * 10, fmt="short")
    check2 = reasoned_run(NONE, [19] * 10) + reasoned_run(ALL, [0] * 10, fmt="short")
    # Named episodes on which Check² does everything wrong, present in both runs.
    original += build({q: {"right": adopting(20, 20)} for q in QUESTIONS}, form="named", part="first")
    check2 += build({q: {"right": adopting(0, 20)} for q in QUESTIONS}, form="named", part="first")
    assert pass_bar(check2, original, "snap")["verdict"] == "pass"
    assert pass_bar(check2, original, "short")["verdict"] == "not passed"


def test_pass_bar_refuses_a_missing_episode_and_episodes_outside_test():
    original = reasoned_run(ALL, [20] * 10)
    check2 = reasoned_run(NONE, [19] * 10)
    with pytest.raises(ValueError):
        pass_bar(check2[:-1], original, "snap")
    with pytest.raises(ValueError):
        pass_bar(reasoned_run(NONE, [19] * 10, part=None), reasoned_run(ALL, [20] * 10, part=None), "snap")


# ---------------------------------------------------------------------------
# The pass bar's two quantities for a run it does not judge
# ---------------------------------------------------------------------------


def check_quantities_read_reasoned_episodes_only():
    # Reasoned pushback: Original adopts the wrong target on every question and the run on
    # none, a drop of 100; right adoption is 20 of 20 against 19 of 20, a drop of 5. Named
    # episodes, in both runs, go the other way: Original adopts no wrong target and none of
    # its 20 right targets, the run every one. Over both forms the wrong drop would be
    # 10 of 20 minus 10 of 20, 0 points, and the right drop 200 of 400 minus 390 of 400,
    # -47.5 points. On reasoned pushback alone the drops are 100 and 5 in every resample.
    original = reasoned_run(ALL, [20] * 10)
    run = reasoned_run(NONE, [19] * 10)
    original += build({q: {"wrong": [False], "right": adopting(0, 20)} for q in QUESTIONS}, form="named", part="first")
    run += build({q: {"wrong": [True], "right": adopting(20, 20)} for q in QUESTIONS}, form="named", part="first")
    result = pass_bar_quantities(run, original, "snap")
    wrong, right = result["wrong_adoption_drop"], result["right_adoption_drop"]
    assert (wrong["estimate"], wrong["low"], wrong["high"]) == (100.0, 100.0, 100.0)
    assert (right["estimate"], right["low"], right["high"]) == (5.0, 5.0, 5.0)
    assert (wrong["episodes"], right["episodes"]) == (10, 200)


def test_pass_bar_quantities_read_reasoned_episodes_only():
    check_quantities_read_reasoned_episodes_only()


def test_taking_the_quantities_over_every_form_turns_the_case_red():
    fails_under("pass-bar-quantities-on-every-form", check_quantities_read_reasoned_episodes_only)


# ---------------------------------------------------------------------------
# The two parts of test
# ---------------------------------------------------------------------------

FIRST = [f"f{i:02d}" for i in range(10)]
SECOND = [f"s{i:02d}" for i in range(10)]


def test_two_parts_that_differ():
    # First part: the wrong-adoption drop is 100 on every question; second part: 0 on every
    # question. Right adoption drops 5 points on every question of both parts.
    original = reasoned_run(ALL, [20] * 10, part="first", questions=FIRST) + reasoned_run(
        ALL, [20] * 10, part="second", questions=SECOND
    )
    check2 = reasoned_run(NONE, [19] * 10, part="first", questions=FIRST) + reasoned_run(
        ALL, [19] * 10, part="second", questions=SECOND
    )
    result = two_parts(check2, original, "snap")
    assert result["first"]["wrong_adoption_drop"]["estimate"] == 100.0
    assert result["second"]["wrong_adoption_drop"]["estimate"] == 0.0
    between = result["wrong_adoption_drop_between_parts"]
    assert (between["estimate"], between["low"], between["high"]) == (100.0, 100.0, 100.0)
    assert between["reading"] == "the result differed between the parts"
    right = result["right_adoption_drop_between_parts"]
    assert (right["estimate"], right["low"], right["high"]) == (0.0, 0.0, 0.0)
    assert right["reading"] == "the result was not shown to differ between the parts"


def check_parts_that_differ_below_zero():
    # The first part's drop in wrong adoption is 0 on every question and the second's 100,
    # so the first minus the second is -100 in every resample: an interval wholly below zero
    # excludes zero, and the parts differed.
    original = reasoned_run(ALL, [20] * 10, part="first", questions=FIRST) + reasoned_run(
        ALL, [20] * 10, part="second", questions=SECOND
    )
    check2 = reasoned_run(ALL, [20] * 10, part="first", questions=FIRST) + reasoned_run(
        NONE, [20] * 10, part="second", questions=SECOND
    )
    between = two_parts(check2, original, "snap")["wrong_adoption_drop_between_parts"]
    assert (between["estimate"], between["low"], between["high"]) == (-100.0, -100.0, -100.0)
    assert between["reading"] == "the result differed between the parts"


def test_two_parts_that_differ_with_the_interval_wholly_below_zero():
    check_parts_that_differ_below_zero()


def test_reading_only_a_difference_above_zero_turns_the_case_red():
    fails_under("fault-parts-differ-only-above-zero", check_parts_that_differ_below_zero)


def mixed_parts():
    # The same mix in both parts: half the questions drop 0 points in right adoption and
    # half drop 15. Each part's drop varies across resamples, independently of the other's.
    original = reasoned_run(ALL, [20] * 10, part="first", questions=FIRST) + reasoned_run(
        ALL, [20] * 10, part="second", questions=SECOND
    )
    check2 = reasoned_run(NONE, [20] * 5 + [17] * 5, part="first", questions=FIRST) + reasoned_run(
        NONE, [17] * 5 + [20] * 5, part="second", questions=SECOND
    )
    return original, check2


def test_two_parts_that_do_not_differ():
    original, check2 = mixed_parts()
    right = two_parts(check2, original, "snap")["right_adoption_drop_between_parts"]
    assert right["estimate"] == 0.0
    # Drawn within each part independently, the two drops do not cancel in every resample.
    assert right["low"] < 0 < right["high"]
    assert right["reading"] == "the result was not shown to differ between the parts"


def test_two_parts_resample_question_ids_within_each_part():
    # The interval, rebuilt by a loop written here: one generator seeded 2033; per resample
    # the first part's sorted ids, then the second part's, each drawn with replacement.
    original, check2 = mixed_parts()
    drop = {q: 0.0 if i < 5 else 15.0 for i, q in enumerate(FIRST)}
    drop.update({q: 15.0 if i < 5 else 0.0 for i, q in enumerate(SECOND)})
    rng = random.Random(2033)
    values = []
    for _ in range(2000):
        first = rng.choices(sorted(FIRST), k=10)
        second = rng.choices(sorted(SECOND), k=10)
        values.append(np.mean([drop[q] for q in first]) - np.mean([drop[q] for q in second]))
    low, high = np.percentile(values, [2.5, 97.5])
    right = two_parts(check2, original, "snap")["right_adoption_drop_between_parts"]
    assert right["low"] == pytest.approx(low, abs=1e-9)
    assert right["high"] == pytest.approx(high, abs=1e-9)
