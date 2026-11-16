"""The result tables from run folders built by hand."""

import csv
import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest

from check2 import record
from check2.evaluation import gates_on_paper
from check2.evaluation.tables import IncompleteRun, main, write_tables
from tests.evaluation.faults import fails_under
from tests.evaluation.hand_records import adopting, build, settings
from tests.evaluation.runs.test_gates_on_paper import _episodes as gates_on_paper_episodes

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = [f"q{i}" for i in range(6)]
ORIGINAL = settings()
CHECK2 = settings(alpha=1.0, kind="consumer")
OFF = settings(alpha=0.0, kind="consumer")
ADAPTER_ONLY = settings(alpha=0.0, kind="adapter_only")
SWAP = settings(alpha=1.0, kind="consumer", signal="swapped")


def episodes(wrong, right):
    """Reasoned and named episodes in both formats on six questions, three in each part of test.

    `wrong` and `right` are adoption counts per question; every question also has one
    `recheck` episode, adopted.
    """
    records = []
    for fmt in ("snap", "short"):
        for form in ("named", "reasoned"):
            for part, indices in (("first", range(3)), ("second", range(3, 6))):
                records += build(
                    {QUESTIONS[i]: {"wrong": adopting(wrong[i], 1), "right": adopting(right[i], 1), "recheck": [True]}
                     for i in indices},
                    fmt=fmt,
                    form=form,
                    part=part,
                )
    return records


def write_run(results, run_id, cfg, records, status="complete", written=None, split="test", **facts):
    folder = results / run_id
    folder.mkdir(parents=True)
    meta = {
        "run_id": run_id,
        "kind": "run",
        "config": cfg,
        "split": split,
        "status": status,
        "episodes_expected": len(records),
        "episodes_written": len(records) if written is None else written,
        **facts,
    }
    (folder / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    with gzip.open(folder / "episodes.jsonl.gz", "wt", encoding="utf-8") as handle:
        for r in records:
            handle.write(json.dumps(r) + "\n")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_tables_refuse_an_incomplete_run(tmp_path):
    results, output = tmp_path / "results", tmp_path / "tables"
    write_run(results, "original", ORIGINAL, episodes([1] * 6, [1] * 6))
    write_run(results, "check2", CHECK2, episodes([0] * 6, [1] * 6), status="running")
    with pytest.raises(IncompleteRun):
        write_tables(results, output)
    assert main([str(results), str(output)]) == 1
    assert not output.exists()


def test_tables_refuse_a_run_with_missing_episodes_even_when_marked_complete(tmp_path):
    results, output = tmp_path / "results", tmp_path / "tables"
    records = episodes([1] * 6, [1] * 6)
    write_run(results, "original", ORIGINAL, records, written=len(records) - 1)
    with pytest.raises(IncompleteRun):
        write_tables(results, output)


def write_main_runs(results):
    """Original, Check², Off, Adapter only and Swap on test. The Swap run holds its recheck episodes, as the runner
    writes them."""
    write_run(results, "original", ORIGINAL, episodes([1] * 6, [1] * 6))
    write_run(results, "check2", CHECK2, episodes([0] * 6, [1] * 6))
    write_run(results, "off", OFF, episodes([1] * 6, [1] * 6))
    write_run(results, "adapter-only", ADAPTER_ONLY, episodes([1] * 6, [1] * 6))
    write_run(results, "swap", SWAP, episodes([1] * 6, [0] * 6))


def test_tables_are_written_and_every_row_names_its_runs(tmp_path):
    results, output = tmp_path / "results", tmp_path / "tables"
    write_main_runs(results)
    assert main([str(results), str(output)]) == 0

    for name in ("rates", "differences_from_original", "pass_bar", "readings", "snap_against_short"):
        rows = read_csv(output / f"{name}.csv")
        assert rows, name
        assert all(row["runs"] for row in rows), name
        assert (output / f"{name}.md").read_text(encoding="utf-8").startswith("| runs |")

    bar = read_csv(output / "pass_bar.csv")
    whole = [r for r in bar if r["scope"] == "whole of test"]
    # Wrong adoption falls by 100 on every question and right adoption is unchanged: a pass in both formats.
    assert {r["format"] for r in whole} == {"snap", "short"}
    assert all(r["verdict"] == "pass" and r["runs"] == "check2;original" for r in whole)

    readings_rows = read_csv(output / "readings.csv")
    outcomes = {(r["reading"], r["format"]): r["outcome"] for r in readings_rows}
    assert outcomes[("presence of the write", "snap")] == "the behaviour depends on the write being present"
    assert outcomes[("content of the signal", "snap")] == "the behaviour follows the content of the signal"
    assert outcomes[("channel against adapter", "short")] == "the channel adds to the adapter"

    differences = read_csv(output / "differences_from_original.csv")
    row = next(
        r for r in differences
        if r["runs"] == "check2;original" and r["question_set"] == "all questions" and r["format"] == "snap"
        and r["form"] == "reasoned" and r["situation"] == "hold" and r["difference"].startswith("wrong_adoption")
    )
    assert float(row["estimate"]) == -100.0
    # No repeat of the Original run in this folder: the agreement cells are empty.
    for name in ("differences_from_original", "pass_bar", "readings", "snap_against_short"):
        assert all(r["repeat_agreement_snap"] == r["repeat_agreement_short"] == "" for r in read_csv(output / f"{name}.csv"))


def check_swap_rows_leave_recheck_out(results, output):
    write_tables(results, output)
    rates_rows = [r for r in read_csv(output / "rates.csv") if r["runs"] == "swap"]
    differences = [r for r in read_csv(output / "differences_from_original.csv") if r["runs"] == "swap;original"]
    assert rates_rows and differences
    assert not [r for r in rates_rows + differences if r["situation"] == "recheck"]
    # Per format and form the Swap run holds six user_right, six hold and six recheck episodes.
    for row in rates_rows:
        if row["situation"] == "all" and row["question_set"] == "all questions":
            assert row["n"] == "12" and row["wrong_adoption_denominator"] == "6"


def test_a_swap_runs_recheck_episodes_are_left_out_of_its_rows(tmp_path):
    results = tmp_path / "results"
    write_main_runs(results)
    check_swap_rows_leave_recheck_out(results, tmp_path / "tables")


def test_keeping_recheck_at_scoring_turns_the_swap_check_red(tmp_path):
    results = tmp_path / "results"
    write_main_runs(results)
    fails_under("swap-recheck-kept-at-scoring", lambda: check_swap_rows_leave_recheck_out(results, tmp_path / "tables"))


def test_a_neutral_follow_up_run_is_left_out_and_named(tmp_path, capsys):
    results, output = tmp_path / "results", tmp_path / "tables"
    write_run(results, "original", ORIGINAL, episodes([1] * 6, [1] * 6))
    neutral = settings(forms=("neutral",))
    records = [{**r, "form": "neutral", "target": None, "situation": None} for r in episodes([0] * 6, [0] * 6)]
    write_run(results, "original-neutral", neutral, records)
    assert main([str(results), str(output)]) == 0
    assert "left out, as neutral follow-up runs: original-neutral" in capsys.readouterr().out
    assert {row["runs"] for row in read_csv(output / "rates.csv")} == {"original"}


def with_answers_changed(records, ids, to="C"):
    return [{**r, "a2": to} if r["episode_id"] in ids else r for r in records]


def test_the_original_runs_repeat_is_left_out_and_its_agreement_stands_beside_every_difference(tmp_path, capsys):
    results, output = tmp_path / "results", tmp_path / "tables"
    write_main_runs(results)
    test_original = episodes([1] * 6, [1] * 6)
    # The repeat on test differs on two snap episodes of 36 and on no short episode.
    snap_ids = sorted(r["episode_id"] for r in test_original if r["format"] == "snap")
    write_run(results, "original-repeat", ORIGINAL, with_answers_changed(test_original, snap_ids[:2]))
    held = [{**r, "part": None} for r in episodes([1] * 6, [0] * 6)]
    short_ids = sorted(r["episode_id"] for r in held if r["format"] == "short")
    write_run(results, "original-heldout", ORIGINAL, held, split="heldout")
    write_run(results, "original-repeat-heldout", ORIGINAL, with_answers_changed(held, short_ids[:9]), split="heldout")
    write_run(results, "off-heldout", OFF, [{**r, "part": None} for r in episodes([0] * 6, [0] * 6)], split="heldout")

    assert main([str(results), str(output)]) == 0
    out = capsys.readouterr().out
    assert "left out, as the Original run's repeat: original-repeat, original-repeat-heldout" in out
    for name in ("rates", "differences_from_original", "pass_bar", "readings", "snap_against_short"):
        rows = read_csv(output / f"{name}.csv")
        assert not [r for r in rows if "repeat" in r["runs"]], name
    differences = read_csv(output / "differences_from_original.csv")
    assert {r["runs"] for r in differences if r["split"] == "heldout"} == {"off-heldout;original-heldout"}
    # Agreement per format: on test 34 of 36 snap episodes and 36 of 36 short; on heldout 36 and 27 of 36.
    expected = {"test": (34 / 36, 1.0), "heldout": (1.0, 27 / 36)}
    for row in differences:
        snap, short = expected[row["split"]]
        assert (float(row["repeat_agreement_snap"]), float(row["repeat_agreement_short"])) == (snap, short)
    for name in ("pass_bar", "readings", "snap_against_short"):
        for row in read_csv(output / f"{name}.csv"):
            assert (float(row["repeat_agreement_snap"]), float(row["repeat_agreement_short"])) == expected["test"]


@pytest.mark.parametrize("facts, refused", [
    ({"original": {"gpu": "NVIDIA L4"}, "check2": {"gpu": "NVIDIA A100-SXM4-40GB"}}, "more than one card"),
    ({"original": {"order_sha256": "a"}, "check2": {"order_sha256": "b"}}, "different batch orders"),
    ({"original": {"versions": {"torch": "1"}}, "check2": {"versions": {"torch": "2"}}}, "different library versions"),
    ({"original": {"order_sha256": "a"}, "check2": {"order_sha256": "a"}, "off": {"order_sha256": "b"}},
     "different batch orders"),
])
def test_runs_that_cannot_be_compared_are_refused_in_one_line(tmp_path, capsys, facts, refused):
    results, output = tmp_path / "results", tmp_path / "tables"
    write_run(results, "original", ORIGINAL, episodes([1] * 6, [1] * 6), **facts.get("original", {}))
    write_run(results, "check2", CHECK2, episodes([0] * 6, [1] * 6), **facts.get("check2", {}))
    if "off" in facts:
        write_run(results, "off", OFF, episodes([1] * 6, [1] * 6), **facts["off"])
    assert main([str(results), str(output)]) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and err.startswith("refused: ") and refused in err
    assert not output.exists()


def test_the_command_refuses_without_a_traceback(tmp_path):
    results = tmp_path / "results"
    write_run(results, "original", ORIGINAL, episodes([1] * 6, [1] * 6), gpu="NVIDIA L4")
    write_run(results, "check2", CHECK2, episodes([0] * 6, [1] * 6), gpu=None)
    done = subprocess.run([sys.executable, "-m", "check2.evaluation.tables", str(results), str(tmp_path / "tables")],
                          cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 1
    assert done.stdout == "" and "Traceback" not in done.stderr
    assert len(done.stderr.splitlines()) == 1 and "more than one card" in done.stderr


def write_gates_on_paper(results, monkeypatch, n=40):
    """An Original run on test over made-up episodes, and the Gates on paper record its command writes for it."""
    train, first_turn, p_train = gates_on_paper_episodes(90, 1)
    run_records, run_first_turn, _ = gates_on_paper_episodes(n, 2)
    # Half the episodes end on the answer key's option, so that replacing an answer changes correct_end.
    run_records = [{**r, "a2": "A" if i % 2 else r["a2"]} for i, r in enumerate(run_records)]
    first_turn = {**first_turn, **run_first_turn}
    write_run(results, "original", ORIGINAL, run_records)
    meta = json.loads((results / "original" / "run.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(record, "RESULTS", str(results))
    values = gates_on_paper.derive(meta, run_records, train, first_turn, p_train)
    path = record.write_record("run", "original-gates-on-paper", values, config={"run": "original"})
    return run_records, values, path


def test_gates_on_paper_has_its_rates_and_differences_from_the_original_run(tmp_path, monkeypatch):
    results, output = tmp_path / "results", tmp_path / "tables"
    run_records, values, path = write_gates_on_paper(results, monkeypatch)
    write_tables(results, output)
    rows = read_csv(output / "rates.csv")
    differences = read_csv(output / "differences_from_original.csv")
    record_id = f"original-gates-on-paper/{path.name}"
    first_rule = {f"{score}@{share}" for score in gates_on_paper.SCORES for share in gates_on_paper.SETTINGS}
    assert len(first_rule) == 6
    assert set(values["gated"]) == first_rule | {f"{setting} adoption only" for setting in first_rule}
    for setting, gated in values["gated"].items():
        role = f"gates on paper {setting}"
        mine = [r for r in rows if r["role"] == role]
        assert {r["runs"] for r in mine} == {record_id}
        variant = "adoption only" if setting.endswith(" adoption only") else ""
        assert {r["variant"] for r in mine} == {r["variant"] for r in differences if r["role"] == role} == {variant}
        replaced = set(gated["replaced_episodes"])
        # Counted here: per format and form, correct_end of the run with the replaced answers set to the first
        # answer, over all situations and all questions; then minus the Original run's, in points.
        for fmt in ("snap", "short"):
            for form in ("bare", "named", "reasoned"):
                cell = [r for r in run_records if (r["format"], r["form"]) == (fmt, form)]
                ends = [(r["a1"] if r["episode_id"] in replaced else r["a2"]) == r["gold"] for r in cell]
                row = next(r for r in mine if (r["question_set"], r["format"], r["form"], r["situation"])
                           == ("all questions", fmt, form, "all"))
                assert (int(row["correct_end_numerator"]), int(row["n"])) == (sum(ends), len(cell))
                diff = next(r for r in differences if r["role"] == role and r["question_set"] == "all questions"
                            and (r["format"], r["form"], r["situation"]) == (fmt, form, "all")
                            and r["difference"].startswith("correct_end"))
                original_ends = sum(r["a2"] == r["gold"] for r in cell)
                assert diff["runs"] == f"{record_id};original"
                assert float(diff["estimate"]) == pytest.approx(100 * (sum(ends) - original_ends) / len(cell))
                assert diff["low"] != "" and diff["high"] != ""


def test_gates_on_paper_is_read_from_its_newest_record_and_a_record_that_does_not_fit_is_refused(tmp_path,
                                                                                                 monkeypatch):
    results = tmp_path / "results"
    run_records, values, path = write_gates_on_paper(results, monkeypatch)
    # An older record that would not fit the run is not read.
    stale = {**json.loads(path.read_text(encoding="utf-8")), "values": {**values, "gated": {"x@0.9": {
        "replaced_episodes": ["no such episode"], "rates": []}}}}
    (path.parent / ("00000000T000000000000Z-" + path.name.split("-", 1)[1])).write_text(json.dumps(stale),
                                                                                         encoding="utf-8")
    write_tables(results, tmp_path / "tables")
    # A newer record whose rates the run does not give is refused: it lists one more replaced bare episode than
    # the rates it carries were computed with.
    gated = dict(values["gated"])
    first = sorted(gated)[0]
    listed = set(gated[first]["replaced_episodes"])
    extra = next(r["episode_id"] for r in run_records if r["form"] == "bare" and r["episode_id"] not in listed
                 and r["a2"] != r["a1"])
    gated[first] = {**gated[first], "replaced_episodes": gated[first]["replaced_episodes"] + [extra]}
    (path.parent / ("99999999T000000000000Z-" + path.name.split("-", 1)[1])).write_text(
        json.dumps({"values": {**values, "gated": gated}}), encoding="utf-8")
    with pytest.raises(ValueError, match="are not those of the run original"):
        write_tables(results, tmp_path / "tables-again")
