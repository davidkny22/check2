"""The causal readings and snap against short, on records built by hand, each with its expected reading worked out on paper.

Most cases give every question the same difference between the two runs, so that every
resample gives that difference and the interval is a single point known in advance. The
cases that need an interval to straddle a bound mix two kinds of question; their bounds
are argued from the binomial counts in the comment beside each.
"""

import pytest

from check2.evaluation.readings import (
    adapter_without_signal,
    channel_against_adapter,
    content,
    presence,
    snap_against_short,
)
from tests.evaluation.faults import fails_under
from tests.evaluation.hand_records import ALL, NONE, QUESTIONS, build, reasoned_run


# ---------------------------------------------------------------------------
# Presence of the write
# ---------------------------------------------------------------------------


def test_presence_with_an_interval_wholly_above_zero():
    # Off adopts the wrong target on every question, Check² on none: Off minus Check² is 100 everywhere.
    off = reasoned_run(ALL, [20] * 10, part=None)
    check2 = reasoned_run(NONE, [20] * 10, part=None)
    result = presence(off, check2, "snap")
    assert result["wrong_adoption_off_minus_check2"]["low"] == 100.0
    assert result["holds"] is True
    assert result["reading"] == "the behaviour depends on the write being present"


def test_presence_with_an_interval_that_includes_zero():
    # Off adopts on the first five questions and Check² on the last five: the estimate is 0
    # and a resample's difference is 100 times (first-kind share minus second-kind share),
    # which takes both signs.
    off = reasoned_run([1] * 5 + [0] * 5, [20] * 10, part=None)
    check2 = reasoned_run([0] * 5 + [1] * 5, [20] * 10, part=None)
    result = presence(off, check2, "snap")
    d = result["wrong_adoption_off_minus_check2"]
    assert d["estimate"] == 0.0
    assert d["low"] < 0 < d["high"]
    assert result["holds"] is False
    assert result["reading"] == "not shown to depend on the write being present"


def test_presence_with_a_point_interval_at_zero_does_not_hold():
    run = reasoned_run(ALL, [20] * 10, part=None)
    assert presence(run, run, "snap")["holds"] is False


def check_presence_reads_reasoned_episodes_only():
    # Reasoned pushback: Off adopts the wrong target on every question and Check² on none,
    # 100 points in every resample, so the write matters. Named episodes, in both runs, go
    # the other way: Check² adopts on every question and Off on none. Over both forms Off
    # and Check² would each adopt 10 of 20, a difference of 0 that does not hold.
    off = reasoned_run(ALL, [20] * 10, part=None) + build({q: {"wrong": [False]} for q in QUESTIONS}, form="named")
    check2 = reasoned_run(NONE, [20] * 10, part=None) + build({q: {"wrong": [True]} for q in QUESTIONS}, form="named")
    result = presence(off, check2, "snap")
    d = result["wrong_adoption_off_minus_check2"]
    assert (d["estimate"], d["low"], d["high"], d["episodes"]) == (100.0, 100.0, 100.0, 10)
    assert result["holds"] is True
    assert result["reading"] == "the behaviour depends on the write being present"


def test_presence_reads_reasoned_episodes_only():
    check_presence_reads_reasoned_episodes_only()


def test_taking_presence_over_named_and_reasoned_turns_the_case_red():
    fails_under("presence-on-named-and-reasoned", check_presence_reads_reasoned_episodes_only)


# ---------------------------------------------------------------------------
# Content of the signal
# ---------------------------------------------------------------------------


def swap_pair(check2_right, swap_right, check2_wrong, swap_wrong):
    """Named and reasoned episodes: per question two user_right and two hold episodes.

    Check² also holds recheck episodes; Swap holds none, since they are left out at scoring.
    """
    check2, swap = [], []
    for form in ("named", "reasoned"):
        check2 += build(
            {q: {"right": [check2_right] * 2, "wrong": [check2_wrong] * 2, "recheck": [True, False]} for q in QUESTIONS},
            form=form,
        )
        swap += build({q: {"right": [swap_right] * 2, "wrong": [swap_wrong] * 2} for q in QUESTIONS}, form=form)
    return swap, check2


def presence_reading(holds):
    return {"format": "snap", "holds": holds}


def test_content_both_hold():
    # Right adoption on user_right: 100 against 0, a fall of 100 everywhere. Wrong adoption on hold: 0 against 100.
    swap, check2 = swap_pair(check2_right=True, swap_right=False, check2_wrong=False, swap_wrong=True)
    result = content(swap, check2, "snap")
    assert result["right_adoption_user_right_swap_minus_check2"]["high"] == -100.0
    assert result["wrong_adoption_hold_swap_minus_check2"]["low"] == 100.0
    assert result["reading"] == "the behaviour follows the content of the signal"


def test_content_neither_holds_without_presence():
    swap, check2 = swap_pair(check2_right=True, swap_right=True, check2_wrong=False, swap_wrong=False)
    result = content(swap, check2, "snap", presence_reading=presence_reading(False))
    assert result["reading"] == "the content of the signal was not shown to matter"


def test_content_neither_holds_with_presence():
    swap, check2 = swap_pair(check2_right=True, swap_right=True, check2_wrong=False, swap_wrong=False)
    result = content(swap, check2, "snap", presence_reading=presence_reading(True))
    assert result["reading"] == "the content of the signal was not shown to matter; presence only"


def test_content_neither_holds_needs_the_presence_reading():
    swap, check2 = swap_pair(check2_right=True, swap_right=True, check2_wrong=False, swap_wrong=False)
    with pytest.raises(ValueError):
        content(swap, check2, "snap")


def test_content_one_holds():
    # Right adoption falls by 100; wrong adoption is 0 in both.
    swap, check2 = swap_pair(check2_right=True, swap_right=False, check2_wrong=False, swap_wrong=False)
    result = content(swap, check2, "snap")
    assert result["right_adoption_falls"] is True and result["wrong_adoption_rises"] is False
    assert result["reading"] == "right adoption on user_right episodes falls; wrong adoption on hold episodes does not rise"
    # Wrong adoption rises by 100; right adoption is 100 in both.
    swap, check2 = swap_pair(check2_right=True, swap_right=True, check2_wrong=False, swap_wrong=True)
    assert content(swap, check2, "snap")["reading"] == (
        "wrong adoption on hold episodes rises; right adoption on user_right episodes does not fall"
    )


def check_content_reads_named_and_reasoned_episodes():
    # Per question two named and two reasoned episodes of each situation. User_right: Check²
    # adopts all four and Swap none, a fall of 100 everywhere. Hold: in named episodes Swap
    # adopts both and Check² neither; in reasoned episodes Check² adopts one and Swap
    # neither. Over both forms Swap adopts 2 of 4 and Check² 1 of 4 on every question, a
    # rise of 25 points in every resample, so both conditions hold. Reasoned episodes alone
    # would give 0 of 2 against 1 of 2, a fall of 50, and wrong adoption would not rise.
    check2 = build({q: {"right": [True, True], "wrong": [False, False], "recheck": [True]} for q in QUESTIONS},
                   form="named")
    check2 += build({q: {"right": [True, True], "wrong": [True, False], "recheck": [True]} for q in QUESTIONS},
                    form="reasoned")
    swap = build({q: {"right": [False, False], "wrong": [True, True]} for q in QUESTIONS}, form="named")
    swap += build({q: {"right": [False, False], "wrong": [False, False]} for q in QUESTIONS}, form="reasoned")
    result = content(swap, check2, "snap")
    wrong = result["wrong_adoption_hold_swap_minus_check2"]
    assert (wrong["estimate"], wrong["low"], wrong["high"], wrong["episodes"]) == (25.0, 25.0, 25.0, 40)
    assert result["right_adoption_user_right_swap_minus_check2"]["high"] == -100.0
    assert result["reading"] == "the behaviour follows the content of the signal"


def test_content_reads_named_and_reasoned_episodes():
    check_content_reads_named_and_reasoned_episodes()


def test_taking_the_wrong_adoption_half_of_content_over_reasoned_alone_turns_the_case_red():
    fails_under("content-on-reasoned-only", check_content_reads_named_and_reasoned_episodes)


# ---------------------------------------------------------------------------
# Channel against adapter
# ---------------------------------------------------------------------------


def test_channel_adds():
    # Check² adopts no wrong target, Adapter only every one: -100 everywhere. Right adoption equal.
    check2 = reasoned_run(NONE, [20] * 10)
    adapter = reasoned_run(ALL, [20] * 10)
    result = channel_against_adapter(check2, adapter, "snap")
    assert result["reading"] == "the channel adds to the adapter"
    assert result["note"] is None


def test_channel_adds_through_right_adoption():
    # Right adoption higher by 5 points everywhere; wrong adoption equal.
    check2 = reasoned_run(NONE, [20] * 10)
    adapter = reasoned_run(NONE, [19] * 10)
    assert channel_against_adapter(check2, adapter, "snap")["reading"] == "the channel adds to the adapter"


def test_adapter_did_better():
    check2 = reasoned_run(ALL, [20] * 10)
    adapter = reasoned_run(NONE, [20] * 10)
    assert channel_against_adapter(check2, adapter, "snap")["reading"] == "the adapter alone did better"


def check_adapter_did_better_through_right_adoption():
    # Right adoption is 5 points lower in Check² on every question; wrong adoption is equal.
    # The difference in right adoption favours the adapter with its interval wholly below
    # zero, and the other's interval, the point 0, does not lie wholly on Check²'s side.
    result = channel_against_adapter(reasoned_run(NONE, [19] * 10), reasoned_run(NONE, [20] * 10), "snap")
    right = result["right_adoption_check2_minus_adapter_only"]
    wrong = result["wrong_adoption_check2_minus_adapter_only"]
    assert (right["low"], right["high"]) == (-5.0, -5.0) and (wrong["low"], wrong["high"]) == (0.0, 0.0)
    assert result["reading"] == "the adapter alone did better"


def test_adapter_did_better_through_right_adoption_with_wrong_adoption_equal():
    check_adapter_did_better_through_right_adoption()


def test_reading_the_adapter_better_only_through_wrong_adoption_turns_the_case_red():
    fails_under("fault-channel-adapter-better-through-wrong-only", check_adapter_did_better_through_right_adoption)


def test_channel_and_adapter_matched():
    run = reasoned_run([1] * 5 + [0] * 5, [20] * 5 + [17] * 5)
    result = channel_against_adapter(run, run, "snap")
    assert result["reading"] == "the two matched within their intervals"
    assert result["note"] == (
        "What the channel gives that an adapter does not: an exact off position, a dial, "
        "and a signal that can be logged and overwritten."
    )


def test_channel_and_adapter_trade():
    # Check² is lower in wrong adoption by 100 (favours Check²) and lower in right adoption
    # by 5 points (favours the adapter), each everywhere.
    check2 = reasoned_run(NONE, [19] * 10)
    adapter = reasoned_run(ALL, [20] * 10)
    assert channel_against_adapter(check2, adapter, "snap")["reading"] == "they trade one rate for the other"


def check_channel_reads_reasoned_episodes_only():
    # Reasoned pushback: Check² adopts no wrong target and Adapter only every one, -100
    # points in every resample, favouring Check²; right adoption is 20 of 20 in both, 0.
    # The channel adds. Named episodes, two hold episodes per question in both runs, go the
    # other way: Check² adopts both and Adapter only neither. Over both forms Check² would
    # adopt 2 of 3 and Adapter only 1 of 3 on every question, a difference wholly above
    # zero, and the adapter alone would have done better.
    check2 = reasoned_run(NONE, [20] * 10) + build({q: {"wrong": [True, True]} for q in QUESTIONS}, form="named",
                                                   part="first")
    adapter = reasoned_run(ALL, [20] * 10) + build({q: {"wrong": [False, False]} for q in QUESTIONS}, form="named",
                                                   part="first")
    result = channel_against_adapter(check2, adapter, "snap")
    wrong = result["wrong_adoption_check2_minus_adapter_only"]
    right = result["right_adoption_check2_minus_adapter_only"]
    assert (wrong["estimate"], wrong["low"], wrong["high"], wrong["episodes"]) == (-100.0, -100.0, -100.0, 10)
    assert (right["low"], right["high"]) == (0.0, 0.0)
    assert result["reading"] == "the channel adds to the adapter"


def test_channel_against_adapter_reads_reasoned_episodes_only():
    check_channel_reads_reasoned_episodes_only()


def test_taking_the_channel_reading_over_named_and_reasoned_turns_the_case_red():
    fails_under("channel-on-named-and-reasoned", check_channel_reads_reasoned_episodes_only)


# ---------------------------------------------------------------------------
# The adapter without the signal, and snap against short
# ---------------------------------------------------------------------------


def test_adapter_without_signal_gives_both_episode_sets_and_no_verdict():
    original = reasoned_run(NONE, [20] * 10, part=None) + build(
        {q: {"wrong": [False]} for q in QUESTIONS}, form="named"
    )
    # Off adopts every wrong target in reasoned episodes and none in named ones.
    off = reasoned_run(ALL, [20] * 10, part=None) + build({q: {"wrong": [False]} for q in QUESTIONS}, form="named")
    adapter = reasoned_run(NONE, [19] * 10, part=None) + build({q: {"wrong": [False]} for q in QUESTIONS}, form="named")
    result = adapter_without_signal(off, original, adapter, "snap")
    # Reasoned alone: 10 of 10 against 0. Pooled with named: 10 of 20 against 0.
    assert result["reasoned"]["off_minus_original"]["wrong_adoption"]["estimate"] == 100.0
    assert result["named_and_reasoned"]["off_minus_original"]["wrong_adoption"]["estimate"] == 50.0
    assert result["reasoned"]["adapter_only_minus_original"]["right_adoption"]["estimate"] == -5.0
    assert "reading" not in result


def test_snap_against_short_pairs_by_question_target_and_form():
    check2 = build({q: {"right": [True], "wrong": [False]} for q in QUESTIONS}, fmt="snap", form="reasoned")
    check2 += build({q: {"right": [True], "wrong": [True]} for q in QUESTIONS}, fmt="snap", form="named")
    original = build({q: {"right": [True], "wrong": [True]} for q in QUESTIONS}, fmt="short", form="reasoned")
    original += build({q: {"right": [False], "wrong": [True]} for q in QUESTIONS}, fmt="short", form="named")
    result = snap_against_short(check2, original)
    assert result["reasoned"]["wrong_adoption"]["estimate"] == -100.0
    assert result["reasoned"]["right_adoption"]["estimate"] == 0.0
    assert result["named"]["right_adoption"]["estimate"] == 100.0
    assert result["named"]["wrong_adoption"]["estimate"] == 0.0
    assert "reading" not in result
