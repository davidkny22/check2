"""Rates on stated answers where a run's appended read fell short, and the scorer table, from runs built by hand."""

import json

from check2.evaluation.tables import write_tables
from tests.evaluation.hand_records import settings
from tests.evaluation.test_tables import ORIGINAL, episodes, read_csv, write_run

APPENDED = {"q4", "q5"}


def write_numbers(results, run_id, numerator, denominator):
    numbers = {"appended_read_check": {"all": {"numerator": numerator, "denominator": denominator,
                                               "rate": numerator / denominator}}}
    (results / run_id / "numbers.json").write_text(json.dumps(numbers), encoding="utf-8")


def write_short_scorer_run(results):
    """A steering run whose agreement is 90 of 100. It adopts the wrong target on q4 and q5 only, and those two
    questions' hold episodes were scored by the appended read and, in the snap format, ran to the cap."""
    records = []
    for r in episodes([0, 0, 0, 0, 1, 1], [1] * 6):
        if r["qid"] in APPENDED and r["situation"] == "hold":
            r = {**r, "scored_by": "appended", "ended": r["format"] != "snap"}
        records.append(r)
    write_run(results, "original", ORIGINAL, episodes([1] * 6, [1] * 6))
    write_numbers(results, "original", 99, 100)
    write_run(results, "steering-g8", settings(steering_strength=8), records)
    write_numbers(results, "steering-g8", 90, 100)


def test_stated_answer_rows_count_only_replies_scored_by_the_pattern(tmp_path):
    results, output = tmp_path / "results", tmp_path / "tables"
    write_short_scorer_run(results)
    write_tables(results, output)
    rows = read_csv(output / "rates.csv")
    assert {(r["runs"], r["replies"]) for r in rows} == {("original", "all replies"), ("steering-g8", "all replies"),
                                                          ("steering-g8", "stated answers only")}

    def row(replies, fmt):
        return next(r for r in rows if r["runs"] == "steering-g8" and r["replies"] == replies
                    and r["question_set"] == "all questions" and (r["format"], r["form"], r["situation"])
                    == (fmt, "reasoned", "all"))
    for fmt in ("snap", "short"):
        # Wrong adoption counts each question's hold and recheck episode; every recheck episode is adopted.
        # All replies: 2 holds and 6 rechecks adopted, of 12. Stated only: the 4 stated holds, none adopted, and
        # the 6 rechecks, of 10.
        everything, stated = row("all replies", fmt), row("stated answers only", fmt)
        assert (everything["wrong_adoption_numerator"], everything["wrong_adoption_denominator"]) == ("8", "12")
        assert (stated["wrong_adoption_numerator"], stated["wrong_adoption_denominator"]) == ("6", "10")
        assert (stated["n"], stated["scored_appended_numerator"]) == ("16", "0")


def test_the_scorer_table_gives_agreement_shares_and_replies_that_ran_to_the_cap(tmp_path):
    results, output = tmp_path / "results", tmp_path / "tables"
    write_short_scorer_run(results)
    write_tables(results, output)
    scorer = {(r["runs"], r["format"]): r for r in read_csv(output / "scorer.csv")}
    assert set(scorer) == {(run, fmt) for run in ("original", "steering-g8") for fmt in ("snap", "short")}
    snap, short = scorer[("steering-g8", "snap")], scorer[("steering-g8", "short")]
    assert (snap["agreement_numerator"], snap["agreement_denominator"], float(snap["agreement"])) == ("90", "100", 0.9)
    assert snap["rates_on_stated_answers_given"] == "True"
    assert scorer[("original", "snap")]["rates_on_stated_answers_given"] == "False"
    # Per format 36 replies; the hold episodes of q4 and q5 in two forms, 4, were scored by the appended read.
    assert snap["replies"] == "36" and float(snap["scored_appended"]) == 4 / 36
    assert float(snap["scored_stated"]) == 32 / 36
    assert snap["not_ended"] == "4" and short["not_ended"] == ""
