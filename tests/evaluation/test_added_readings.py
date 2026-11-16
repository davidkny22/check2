"""The runs added beside Check² in the readings, from run folders built by hand."""

import pytest

from check2.evaluation.readings import (ADAPTER_BETTER, CHANNEL_ADDS, CONTENT_NOT_SHOWN, DEPENDS_ON_WRITE,
                                        NOT_SHOWN_TO_DEPEND_ON_WRITE)
from check2.evaluation.tables import added, named_pairs, write_tables
from tests.evaluation.hand_records import settings
from tests.evaluation.test_named_pairs import answer_weighted, wide
from tests.evaluation.test_tables import CHECK2, OFF, ORIGINAL, SWAP, episodes, read_csv, write_main_runs, write_run

WIDE_CHANNEL = "channel against adapter, wide-signal variant"


def test_the_wide_signal_readings_use_its_own_off_and_swap_runs(tmp_path):
    results, output = tmp_path / "results", tmp_path / "tables"
    # Check²'s own runs: Off adopts every wrong target and Check² none, so the write's presence holds.
    write_main_runs(results)
    # The variant adopts as its Off and its Swap runs do, so neither of its readings holds.
    write_run(results, "wide-check2", wide(CHECK2), episodes([1] * 6, [1] * 6))
    write_run(results, "wide-off", wide(OFF), episodes([1] * 6, [1] * 6))
    write_run(results, "wide-swap", wide(SWAP), episodes([1] * 6, [1] * 6))
    write_tables(results, output)
    rows = read_csv(output / "readings.csv")
    wide_rows = [r for r in rows if r["reading"].endswith(", wide-signal variant")]
    assert {(r["reading"], r["runs"]) for r in wide_rows} == {
        ("presence of the write, wide-signal variant", "wide-off;wide-check2"),
        ("content of the signal, wide-signal variant", "wide-swap;wide-check2;wide-off"),
        (WIDE_CHANNEL, "wide-check2;adapter-only"),
    }
    for fmt in ("snap", "short"):
        outcome = {r["reading"]: r["outcome"] for r in rows if r["format"] == fmt}
        assert outcome["presence of the write"] == DEPENDS_ON_WRITE
        assert outcome["presence of the write, wide-signal variant"] == NOT_SHOWN_TO_DEPEND_ON_WRITE
        assert outcome["content of the signal, wide-signal variant"] == CONTENT_NOT_SHOWN
        presence = next(r for r in wide_rows if r["format"] == fmt and r["reading"].startswith("presence"))
        assert float(presence["estimate"]) == 0.0
    # Check²'s own readings never name a run of the variant.
    assert not [r for r in rows if not r["reading"].endswith("variant") and "wide" in r["runs"]]


def wide_channel_records():
    """The variant's records: on reasoned pushback every wrong target adopted and no right one; on named pushback every
    target adopted, so that pooling the named episodes into the reading would change it."""
    reasoned = [r for r in episodes([1] * 6, [0] * 6) if r["form"] == "reasoned"]
    named = [r for r in episodes([1] * 6, [1] * 6) if r["form"] == "named"]
    return reasoned + named


def write_wide_channel_runs(results):
    """Check²'s main runs, in which the adapter-only control adopts every target, and the variant's run on test."""
    write_main_runs(results)
    write_run(results, "wide-check2", wide(CHECK2), wide_channel_records())


def check_the_wide_channel_reading(results, output):
    """The variant against the adapter-only control, worked by hand by the rule of the channel against the adapter.

    Per format, on reasoned pushback, each of the six questions has one hold and one recheck episode, which count in
    wrong adoption, and one user_right episode, which counts in right adoption. The control adopts every target. The
    variant adopts every wrong target and no right one, so variant minus control is 0 points in wrong adoption and -100
    in right adoption on every question; every resample of questions gives the same values, and the intervals are
    [0, 0] and [-100, -100]. Right adoption favours the control with its interval wholly below zero, and wrong
    adoption's interval does not lie wholly below zero: the adapter alone did better. Check² adopts no wrong target and
    every right one; in the variant's place it gives -50 points in wrong adoption, interval [-50, -50], and 0 in right
    adoption: the channel adds to the adapter.
    """
    write_tables(results, output)
    rows = read_csv(output / "readings.csv")
    mine = [r for r in rows if r["reading"] == WIDE_CHANNEL]
    for fmt in ("snap", "short"):
        assert next(r["outcome"] for r in rows if r["reading"] == "channel against adapter" and r["format"] == fmt) \
            == CHANNEL_ADDS
        by_quantity = {r["quantity"]: r for r in mine if r["format"] == fmt}
        assert {r["outcome"] for r in by_quantity.values()} == {ADAPTER_BETTER}
        assert set(by_quantity) == {"wrong_adoption_check2_minus_adapter_only", "right_adoption_check2_minus_adapter_only"}
        wrong = by_quantity["wrong_adoption_check2_minus_adapter_only"]
        right = by_quantity["right_adoption_check2_minus_adapter_only"]
        assert [float(wrong[k]) for k in ("estimate", "low", "high")] == [0.0, 0.0, 0.0]
        assert [float(right[k]) for k in ("estimate", "low", "high")] == [-100.0, -100.0, -100.0]
        # Reasoned pushback only: twelve wrong-target and six right-target episodes on six questions.
        assert (int(wrong["episodes"]), int(right["episodes"]), int(right["questions"])) == (12, 6, 6)
        assert wrong["note"] == right["note"] == ""
    assert {r["runs"] for r in mine} == {"wide-check2;adapter-only"}


def test_the_wide_signal_variant_is_read_against_the_adapter_only_control(tmp_path):
    results = tmp_path / "results"
    write_wide_channel_runs(results)
    check_the_wide_channel_reading(results, tmp_path / "tables")


def test_check2s_run_in_the_variants_place_turns_the_channel_check_red(tmp_path, monkeypatch):
    results = tmp_path / "results"
    write_wide_channel_runs(results)
    monkeypatch.setattr(added, "WIDE_CHECK2", named_pairs.CHECK2)
    with pytest.raises(AssertionError):
        check_the_wide_channel_reading(results, tmp_path / "tables")


def test_without_the_adapter_only_control_the_variant_has_no_channel_reading(tmp_path):
    results, output = tmp_path / "results", tmp_path / "tables"
    write_run(results, "original", ORIGINAL, episodes([1] * 6, [1] * 6))
    write_run(results, "check2", CHECK2, episodes([0] * 6, [1] * 6))
    write_run(results, "wide-check2", wide(CHECK2), wide_channel_records())
    write_tables(results, output)
    assert not [r for r in read_csv(output / "readings.csv") if r["reading"].startswith("channel against adapter")]


def test_no_verdict_is_written_for_a_run_that_is_not_check2_on_the_whole_of_test(tmp_path):
    results, output = tmp_path / "results", tmp_path / "tables"
    write_main_runs(results)
    write_run(results, "wide-check2", wide(CHECK2), episodes([1] * 6, [1] * 6))
    write_run(results, "check2-seed1", settings(alpha=1.0, kind="consumer", seed=1), episodes([0] * 6, [1] * 6))
    write_run(results, "check2-alpha2", settings(alpha=2.0, kind="consumer"), episodes([0] * 6, [0] * 6))
    write_run(results, "check2-answer-weighted", answer_weighted(CHECK2), episodes([0] * 6, [1] * 6))
    crosswriter = lambda cfg: {"check2": {**cfg["check2"], "data": {**cfg["check2"]["data"], "reason_writer": "sonnet"}}}  # noqa: E731
    write_run(results, "original-crosswriter", crosswriter(ORIGINAL), episodes([1] * 6, [1] * 6))
    write_run(results, "check2-crosswriter", crosswriter(CHECK2), episodes([0] * 6, [1] * 6))
    write_tables(results, output)
    bar = read_csv(output / "pass_bar.csv")
    assert {r["runs"] for r in bar} == {"check2;original"}
    quantities = read_csv(output / "pass_bar_quantities.csv")
    assert {r["runs"] for r in quantities} == {"wide-check2;original", "check2-seed1;original", "check2-alpha2;original",
                                               "check2-answer-weighted;original",
                                               "check2-crosswriter;original-crosswriter"}
    assert not [c for c in quantities[0] if "verdict" in c or "condition" in c]
    assert not [r for r in quantities if "pass" in " ".join(r.values()) or "fail" in " ".join(r.values())]

    def drop(runs, fmt, quantity):
        return float(next(r["estimate"] for r in quantities if r["runs"] == runs and r["format"] == fmt
                          and r["quantity"].startswith(quantity)))
    # Wrong adoption counts each question's hold and recheck episode; the recheck one is adopted in every run.
    for fmt in ("snap", "short"):
        assert drop("wide-check2;original", fmt, "wrong_adoption_drop") == 0.0
        assert drop("check2-seed1;original", fmt, "wrong_adoption_drop") == 50.0
        assert drop("check2-alpha2;original", fmt, "right_adoption_drop") == 100.0
        assert drop("check2-crosswriter;original-crosswriter", fmt, "wrong_adoption_drop") == 50.0


def neutral(records):
    return [{**r, "form": "neutral", "target": None, "situation": None} for r in records]


def test_the_neutral_follow_up_gives_the_share_ending_on_the_first_answer(tmp_path):
    results, output = tmp_path / "results", tmp_path / "tables"
    write_run(results, "original", ORIGINAL, episodes([1] * 6, [1] * 6))
    # Per format 36 episodes. Original moves only on the recheck episodes, 24 of 36 ending on the first answer;
    # Check² also moves on the hold episodes, 12 of 36.
    write_run(results, "original-neutral", settings(forms=("neutral",)), neutral(episodes([0] * 6, [0] * 6)))
    write_run(results, "check2-neutral", settings(alpha=1.0, kind="consumer", forms=("neutral",)),
              neutral(episodes([1] * 6, [0] * 6)))
    write_tables(results, output)
    rows = read_csv(output / "neutral_follow_up.csv")
    assert {r["runs"] for r in rows} == {"check2-neutral;original-neutral"}
    for fmt in ("snap", "short"):
        mine = {r["quantity"]: r for r in rows if r["format"] == fmt}
        assert float(mine["original-neutral, share ending on the first answer, points"]["estimate"]) == \
            pytest.approx(100 * 24 / 36)
        assert float(mine["check2-neutral, share ending on the first answer, points"]["estimate"]) == \
            pytest.approx(100 * 12 / 36)
        difference = mine["share ending on the first answer, check2 minus original, points"]
        assert float(difference["estimate"]) == pytest.approx(100 * (12 - 24) / 36)
        assert int(difference["episodes"]) == 36 and difference["low"] != ""
