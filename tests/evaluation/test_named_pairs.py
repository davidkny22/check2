"""Paired differences between named runs, against counts made by hand."""

import json

import pytest

from check2.config.schema import WIDE_WRITER_INPUT
from check2.evaluation.tables import write_tables
from check2.evaluation.tables.named_pairs import named_pairs
from tests.evaluation.faults import planted
from tests.evaluation.hand_records import settings
from tests.evaluation.test_tables import CHECK2, ORIGINAL, SWAP, episodes, read_csv, write_run


def wide(cfg):
    check2 = dict(cfg["check2"])
    check2["writer"] = {**check2["writer"], "input": WIDE_WRITER_INPUT}
    return {"check2": check2}


def answer_weighted(cfg):
    return {"check2": {**cfg["check2"], "training": {"answer_token_weight": 0.5}}}


def rows_of(output, runs):
    return [r for r in read_csv(output / "differences_between_runs.csv") if r["runs"] == runs]


def value(rows, fmt, form, rate):
    found = [r for r in rows if (r["format"], r["form"]) == (fmt, form) and r["difference"].startswith(rate + ",")]
    assert len(found) == 1
    return float(found[0]["estimate"])


def test_check2_against_system_prompt_equals_the_hand_count(tmp_path):
    results, output = tmp_path / "results", tmp_path / "tables"
    write_run(results, "original", ORIGINAL, episodes([1] * 6, [1] * 6))
    # Check² adopts the wrong target on the last three of six questions; System prompt on all six. Every question
    # also has one recheck episode, adopted in both runs, which counts in wrong adoption: 9 of 12 against 12 of 12.
    write_run(results, "check2", CHECK2, episodes([0, 0, 0, 1, 1, 1], [1] * 6))
    write_run(results, "system-prompt", settings(system_prompt_suffix=" Only."), episodes([1] * 6, [1] * 6))
    write_tables(results, output)
    rows = rows_of(output, "check2;system-prompt")
    for fmt in ("snap", "short"):
        for form in ("reasoned", "named"):
            assert value(rows, fmt, form, "wrong_adoption") == pytest.approx(100 * (9 - 12) / 12)
            assert value(rows, fmt, form, "right_adoption") == 0.0
            # correct_end over 18 episodes: Check² ends on the key on 3 more hold episodes.
            assert value(rows, fmt, form, "correct_end") == pytest.approx(100 * 3 / 18)
    assert all(r["difference"].endswith("check2 minus system prompt, points") for r in rows)


def write_wide_runs(results):
    write_run(results, "original", ORIGINAL, episodes([1] * 6, [1] * 6))
    write_run(results, "wide-check2", wide(CHECK2), episodes([0] * 6, [1] * 6))
    # The Swap run holds its recheck episodes, adopted, as the runner writes them; they are left out at scoring.
    write_run(results, "wide-swap", wide(SWAP), episodes([1] * 6, [0] * 6))


def check_wide_against_its_swap_run(results, output):
    write_tables(results, output)
    rows = rows_of(output, "wide-check2;wide-swap")
    assert rows
    for fmt in ("snap", "short"):
        # On the hold and user_right episodes only: wrong adoption 0 of 6 against 6 of 6, right adoption 6 of 6
        # against 0 of 6, correct_end 12 of 12 against 0 of 12.
        assert value(rows, fmt, "reasoned", "wrong_adoption") == -100.0
        assert value(rows, fmt, "reasoned", "right_adoption") == 100.0
        assert value(rows, fmt, "reasoned", "correct_end") == 100.0
        assert {int(r["episodes"]) for r in rows if r["difference"].startswith("correct_end")} == {12}


def test_the_wide_signal_variant_is_paired_with_its_swap_run_on_the_episodes_both_score(tmp_path):
    results = tmp_path / "results"
    write_wide_runs(results)
    check_wide_against_its_swap_run(results, tmp_path / "tables")


def test_pairing_on_unmatched_episodes_turns_the_check_red(tmp_path):
    results = tmp_path / "results"
    write_wide_runs(results)
    with planted("pair-on-unmatched-episodes"):
        with pytest.raises(AssertionError):
            check_wide_against_its_swap_run(results, tmp_path / "tables")


def test_the_perfect_signal_pair_is_taken_at_the_kept_epochs_on_heldout(tmp_path):
    results, output = tmp_path / "results", tmp_path / "tables"
    held = lambda wrong, right: [{**r, "part": None} for r in episodes(wrong, right)]  # noqa: E731
    write_run(results, "original-heldout", ORIGINAL, held([1] * 6, [1] * 6), split="heldout")
    perfect = settings(alpha=1.0, kind="consumer", signal="perfect")
    write_run(results, "perfect-signal-heldout-epoch1", perfect, held([1] * 6, [1] * 6), split="heldout")
    write_run(results, "perfect-signal-heldout-epoch2", perfect, held([0] * 6, [1] * 6), split="heldout")
    weighted = answer_weighted(perfect)
    write_run(results, "perfect-signal-answer-weighted-heldout-epoch1", weighted, held([0] * 6, [0] * 6),
              split="heldout")
    write_run(results, "perfect-signal-answer-weighted-heldout-epoch2", weighted, held([1] * 6, [1] * 6),
              split="heldout")
    rules = results / "measurements" / "checkpoint_rule"
    rules.mkdir(parents=True)
    for name, runs, kept in (("20261115T000000000000Z-a.json", ["perfect-signal-heldout-epoch1",
                                                                "perfect-signal-heldout-epoch2"], 2),
                             ("20261115T000001000000Z-a.json", ["perfect-signal-answer-weighted-heldout-epoch1",
                                                                "perfect-signal-answer-weighted-heldout-epoch2"], 1)):
        record = {"config": {"kind": "epochs", "runs": runs},
                  "values": {"kept": {"epoch": kept, "label": runs[kept - 1]}}}
        (rules / name).write_text(json.dumps(record), encoding="utf-8")
    # A record of the rule that ranks steering strengths keeps several runs and is not read for a kept epoch.
    steering = {"config": {"kind": "steering", "runs": ["steering-g1-heldout", "steering-g2-heldout"]},
                "values": {"kept": {"labels": ["steering-g2-heldout"], "strengths": [2]}}}
    (rules / "20261115T000002000000Z-a.json").write_text(json.dumps(steering), encoding="utf-8")
    write_tables(results, output)
    pairs = {r["runs"] for r in read_csv(output / "differences_between_runs.csv")}
    assert pairs == {"perfect-signal-answer-weighted-heldout-epoch1;perfect-signal-heldout-epoch2"}
    rows = rows_of(output, "perfect-signal-answer-weighted-heldout-epoch1;perfect-signal-heldout-epoch2")
    # Right adoption 0 of 6 against 6 of 6; wrong adoption 6 of 12 (the recheck episodes) against 6 of 12.
    assert value(rows, "snap", "reasoned", "right_adoption") == -100.0
    assert value(rows, "snap", "reasoned", "wrong_adoption") == 0.0


def _run(run_id, split, base, named=(), role=None, variant=""):
    return {"id": run_id, "split": split, "base": base, "named": named, "role": role or base, "variant": variant}


def test_the_named_pairs_are_those_of_the_list_and_never_cross_splits():
    wide_input = f"input={WIDE_WRITER_INPUT}"
    runs = [
        _run("check2", "test", "check2"), _run("system-prompt", "test", "system prompt"),
        _run("connector", "test", "connector"), _run("adapter-only", "test", "adapter only"),
        _run("check2-alpha0.5", "test", "dial alpha=0.5"), _run("check2-alpha2", "test", "dial alpha=2.0"),
        _run("check2-bands-11-15-off", "test", "bands off [11, 15]"), _run("check2-seed1", "test", "check2", ("seed=1",)),
        _run("wide-check2", "test", "check2", (wide_input,)), _run("wide-off", "test", "off", (wide_input,)),
        _run("wide-swap", "test", "swap", (wide_input,)),
        _run("check2-answer-weighted", "test", "check2", ("answer_weight=0.5",)),
        _run("check2-heldout", "heldout", "check2"), _run("off", "test", "off"), _run("swap", "test", "swap"),
        _run("check2-crosswriter", "test", "check2", variant="cross-writer"),
        _run("gop/reader95", "test", None, role="gates on paper reader_user_right@0.95"),
        _run("gop/reader95-ao", "test", None, role="gates on paper reader_user_right@0.95 adoption only",
             variant="adoption only"),
        _run("gop/target95", "test", None, role="gates on paper target_probability@0.95"),
    ]
    found = [(a["id"], b["id"]) for a, b in named_pairs(runs, set())]
    assert found == [
        ("check2", "system-prompt"), ("check2", "connector"), ("check2", "adapter-only"), ("check2", "check2-alpha0.5"),
        ("check2", "check2-alpha2"), ("check2", "check2-bands-11-15-off"), ("check2", "check2-seed1"),
        ("check2", "wide-check2"), ("check2", "check2-answer-weighted"), ("wide-check2", "wide-off"),
        ("wide-check2", "wide-swap"), ("gop/reader95", "check2"), ("gop/reader95", "system-prompt"),
        ("gop/reader95-ao", "check2"), ("gop/reader95-ao", "system-prompt"),
    ]
