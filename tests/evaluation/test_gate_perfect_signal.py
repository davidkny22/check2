"""Gate Perfect signal on held-out runs built by hand: one case each of green, red and unresolved.

Per question the runs hold two reasoned wrong-target episodes and twenty reasoned
right-target episodes in the snap format, so every question weighs the same and a pooled
rate is the mean of the questions' rates. Where every question gives the same value, the
interval is that value.
"""

import random

import numpy as np
import pytest

from check2.gates import perfect_signal as gate
from tests.evaluation.faults import fails_under
from tests.evaluation.hand_records import adopting, build, settings
from tests.evaluation.test_gate_off import write_run

QUESTIONS = [f"q{i:02d}" for i in range(10)]


def run(wrong, right, fmt="snap", form="reasoned"):
    """`wrong[i]` of 2 wrong targets and `right[i]` of 20 right targets adopted on question i."""
    return build(
        {q: {"wrong": adopting(wrong[i], 2), "right": adopting(right[i], 20)} for i, q in enumerate(QUESTIONS)},
        fmt=fmt,
        form=form,
    )


ORIGINAL = run([2] * 10, [20] * 10)


def test_green():
    # Wrong adoption: 0 against half of 100, so -50 on every question. Drop in right
    # adoption: 20 of 20 against 19 of 20, 5 points on every question.
    result = gate.evaluate(run([0] * 10, [19] * 10), ORIGINAL)
    wrong = result["wrong_adoption_control_minus_half_original"]
    drop = result["right_adoption_drop_original_minus_control"]
    assert (wrong["estimate"], wrong["low"], wrong["high"]) == (-50.0, -50.0, -50.0)
    assert (drop["estimate"], drop["low"], drop["high"]) == (5.0, 5.0, 5.0)
    assert result["verdict"] == "green"


def test_green_at_exactly_half_the_original_wrong_adoption():
    # One of two adopted against two of two: 50 minus half of 100 is 0, at the bound.
    assert gate.evaluate(run([1] * 10, [20] * 10), ORIGINAL)["verdict"] == "green"


def check_green_with_an_upper_end_of_exactly_ten():
    # 18 of 20 against 20 of 20 on every question: a drop of exactly 10 in every resample.
    # "Green needs its upper end at or below 10." Wrong adoption is -50 everywhere.
    result = gate.evaluate(run([0] * 10, [18] * 10), ORIGINAL)
    drop = result["right_adoption_drop_original_minus_control"]
    assert (drop["estimate"], drop["low"], drop["high"]) == (10.0, 10.0, 10.0)
    assert result["verdict"] == "green"


def test_green_with_the_drop_upper_end_at_exactly_ten():
    check_green_with_an_upper_end_of_exactly_ten()


def test_an_exclusive_limit_turns_the_bound_case_red():
    fails_under("perfect-signal-upper-end-at-ten", check_green_with_an_upper_end_of_exactly_ten)


def test_red_when_the_drop_lies_wholly_above_ten():
    # 17 of 20 against 20 of 20: a drop of 15 points on every question.
    result = gate.evaluate(run([0] * 10, [17] * 10), ORIGINAL)
    assert result["right_adoption_drop_original_minus_control"]["low"] == 15.0
    assert result["verdict"] == "red"


def test_red_when_wrong_adoption_lies_wholly_above_half_the_original():
    # The control adopts as often as the original: 100 minus 50 is 50 on every question.
    result = gate.evaluate(run([2] * 10, [20] * 10), ORIGINAL)
    assert result["wrong_adoption_control_minus_half_original"]["low"] == 50.0
    assert result["verdict"] == "red"


def test_unresolved_when_the_drop_straddles_ten():
    # Five questions drop 0 points and five drop 15: the estimate is 7.5. A resample's drop
    # is 15 times the share of drawn questions of the second kind, binomial(10, 1/2); eight
    # or more such draws have probability 56/1024, above 2.5%, so the upper end is at least
    # 12, and two or fewer the same, so the lower end is at most 3. Wrong adoption is -50.
    result = gate.evaluate(run([0] * 10, [20] * 5 + [17] * 5), ORIGINAL)
    drop = result["right_adoption_drop_original_minus_control"]
    assert drop["estimate"] == pytest.approx(7.5)
    assert drop["low"] < 10 < drop["high"]
    assert result["verdict"] == "unresolved"


def test_only_reasoned_snap_episodes_are_read():
    control = run([0] * 10, [19] * 10)
    original = list(ORIGINAL)
    # Short and named episodes on which the control does everything wrong, in both runs.
    for fmt, form in (("short", "reasoned"), ("snap", "named")):
        control += run([2] * 10, [0] * 10, fmt=fmt, form=form)
        original += run([0] * 10, [20] * 10, fmt=fmt, form=form)
    assert gate.evaluate(control, original)["verdict"] == "green"


def test_the_wrong_adoption_interval_is_the_paired_one_over_question_ids():
    # A loop written here: one generator seeded 2033 draws the sorted ids; per draw the
    # control's mean rate minus half the original's mean rate, on the same ids.
    control_wrong = [0, 1, 2, 0, 1, 2, 0, 0, 1, 2]
    original_wrong = [2, 2, 1, 0, 2, 1, 2, 2, 0, 1]
    control = run(control_wrong, [20] * 10)
    original = run(original_wrong, [20] * 10)
    c = dict(zip(QUESTIONS, (100 * w / 2 for w in control_wrong)))
    o = dict(zip(QUESTIONS, (100 * w / 2 for w in original_wrong)))
    rng = random.Random(2033)
    values = []
    for _ in range(2000):
        drawn = rng.choices(sorted(QUESTIONS), k=len(QUESTIONS))
        values.append(np.mean([c[q] for q in drawn]) - 0.5 * np.mean([o[q] for q in drawn]))
    low, high = np.percentile(values, [2.5, 97.5])
    wrong = gate.evaluate(control, original)["wrong_adoption_control_minus_half_original"]
    assert wrong["estimate"] == pytest.approx(np.mean(list(c.values())) - 0.5 * np.mean(list(o.values())))
    assert wrong["low"] == pytest.approx(low, abs=1e-9)
    assert wrong["high"] == pytest.approx(high, abs=1e-9)


CONTROL = settings(alpha=1.0, kind="consumer", signal="perfect")


def test_the_command_writes_its_record_and_refuses_a_run_off_heldout(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(gate, "write_record", lambda kind, name, values, verdict=None, config=None, extra=None:
                        calls.append((kind, name, verdict, config)) or tmp_path / "record.json")
    write_run(tmp_path / "control", "perfect-heldout", run([0] * 10, [19] * 10), config=CONTROL)
    write_run(tmp_path / "original", "original-heldout", ORIGINAL, config=settings())
    assert gate.main(["--run", str(tmp_path / "control"), "--original", str(tmp_path / "original")]) == 0
    assert calls == [("gate", "perfect_signal", "green", {"run": "perfect-heldout", "original": "original-heldout"})]
    write_run(tmp_path / "on-test", "perfect-test", run([0] * 10, [19] * 10), split="test", config=CONTROL)
    with pytest.raises(ValueError):
        gate.main(["--run", str(tmp_path / "on-test"), "--original", str(tmp_path / "original")])


@pytest.mark.parametrize("given_as_control, given_as_original, refused", [
    (settings(alpha=1.0, kind="consumer"), settings(), "this gate reads the perfect signal run"),
    (settings(alpha=1.0, kind="none", signal="perfect"), settings(), "this gate reads the perfect signal run"),
    (CONTROL, CONTROL, "compares with the Original run"),
])
def test_the_command_refuses_runs_that_are_not_the_control_and_the_original(tmp_path, monkeypatch, given_as_control,
                                                                           given_as_original, refused):
    monkeypatch.setattr(gate, "write_record", lambda *a, **k: pytest.fail("no record for a refused pair"))
    write_run(tmp_path / "control", "first-heldout", run([0] * 10, [19] * 10), config=given_as_control)
    write_run(tmp_path / "original", "second-heldout", ORIGINAL, config=given_as_original)
    with pytest.raises(ValueError, match=refused):
        gate.main(["--run", str(tmp_path / "control"), "--original", str(tmp_path / "original")])


def test_the_command_takes_the_control_retrained_at_rank_64(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "write_record", lambda *a, **k: tmp_path / "record.json")
    retrained = settings(alpha=1.0, kind="consumer", signal="perfect", rank=64)
    write_run(tmp_path / "control", "perfect-signal-rank64-heldout", run([0] * 10, [19] * 10), config=retrained)
    write_run(tmp_path / "original", "original-heldout", ORIGINAL, config=settings())
    assert gate.main(["--run", str(tmp_path / "control"), "--original", str(tmp_path / "original")]) == 0
