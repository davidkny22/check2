"""The appended-read sample, and the gates Repeatability and Scorer fallback over run folders."""

from __future__ import annotations

import random

import pytest

from check2.evaluation.runs import appended_check, output, run
from check2.gates import repeatability, scorer_fallback

from .conftest import config


def _record(i: int, fmt: str, situation: str | None, scored_by: str, a2: str = "A", appended: str = "A") -> dict:
    return {"episode_id": f"q{i:04d}:{fmt}:{situation}", "qid": f"q{i:04d}", "format": fmt, "situation": situation,
            "form": "named", "a1": "B", "gold": "A", "target": "A", "a2": a2, "scored_by": scored_by,
            "appended_read": appended, "ended": True}


def _records(counts: dict) -> list[dict]:
    """`counts` maps (format, situation) to (stated, appended) reply counts."""
    out, i = [], 0
    for (fmt, situation), (stated, appended) in counts.items():
        for kind, n in (("stated", stated), ("appended", appended)):
            for _ in range(n):
                out.append(_record(i, fmt, situation, kind))
                i += 1
    return out


def test_the_sample_is_200_per_situation_per_format_or_all_there_are():
    records = _records({("snap", "hold"): (250, 40), ("snap", "user_right"): (30, 5), ("short", "hold"): (200, 0),
                        ("short", "recheck"): (0, 12)})
    drawn = appended_check.sample(records)
    assert {k: len(v) for k, v in drawn.items()} == {("snap", "hold"): 200, ("snap", "user_right"): 30,
                                                      ("short", "hold"): 200}
    stated_hold = sorted(r["episode_id"] for r in records
                         if (r["format"], r["situation"], r["scored_by"]) == ("snap", "hold", "stated"))
    assert drawn[("snap", "hold")] == random.Random(2034).sample(stated_hold, 200)
    by_id = {r["episode_id"]: r for r in records}
    assert all(by_id[i]["scored_by"] == "stated" for ids in drawn.values() for i in ids)


def _with_disagreements(records: list[dict], n: int) -> list[dict]:
    drawn = [i for ids in appended_check.sample(records).values() for i in sorted(ids)][:n]
    return [{**r, "appended_read": "D"} if r["episode_id"] in drawn else r for r in records]


def test_scorer_fallback_holds_at_97_percent_and_fails_below():
    records = _records({("snap", "hold"): (100, 3), ("short", "hold"): (100, 3)})
    assert scorer_fallback.evaluate(_with_disagreements(records, 6))["verdict"] == "green"
    assert scorer_fallback.evaluate(_with_disagreements(records, 7))["verdict"] == "red"
    missing = [dict(r) for r in records]
    missing[0]["a2"] = None
    assert scorer_fallback.evaluate(missing)["verdict"] == "red"
    assert scorer_fallback.evaluate(_records({("snap", "hold"): (0, 5)}))["verdict"] == "red"


@pytest.fixture()
def two_originals(bench, reader, results, episode_folder):
    folders = [run(config(), name, bench.model, bench.tokenizer, bench.layout, reader, None,
                   episode_folder=episode_folder) for name in ("original-heldout", "original-repeat-heldout")]
    return folders


def test_repeatability_is_green_on_two_executions_and_red_when_one_answer_differs(two_originals):
    from check2.evaluation.tables import load_run

    assert repeatability.main(["--run", str(two_originals[0]), "--repeat", str(two_originals[1])]) == 0
    meta_a, a = load_run(two_originals[0])
    meta_b, b = load_run(two_originals[1])
    changed = [dict(r) for r in b]
    changed[3]["a2"] = "D" if changed[3]["a2"] != "D" else "C"
    result = repeatability.evaluate(meta_a, a, meta_b, changed)
    assert result["verdict"] == "red" and result["episodes_differing"] == 1
    assert result["agreement"]["numerator"] == len(a) - 1


def test_repeatability_refuses_runs_that_are_not_the_original(two_originals):
    from check2.evaluation.tables import load_run

    meta_a, a = load_run(two_originals[0])
    meta_b, b = load_run(two_originals[1])
    other = {**meta_b, "config": {"check2": {**meta_b["config"]["check2"],
                                             "adapter": {**meta_b["config"]["check2"]["adapter"], "kind": "consumer"}}}}
    with pytest.raises(ValueError, match="two Original runs"):
        repeatability.evaluate(meta_a, a, other, b)


def test_repeatability_refuses_one_run_given_twice(two_originals):
    with pytest.raises(ValueError, match="both folders hold the run original-heldout"):
        repeatability.main(["--run", str(two_originals[0]), "--repeat", str(two_originals[0])])


def test_scorer_fallback_runs_on_a_run_folder(two_originals, results):
    code = scorer_fallback.main(["--run", str(two_originals[0])])
    assert code in (0, 1)
    assert list((results / "gates" / "scorer_fallback").glob("*.json"))
    assert output.read_records(two_originals[0])


def test_gates_on_paper_and_the_reader_table_run_on_a_run_folder(two_originals, results):
    from check2.evaluation import gates_on_paper, reader_table

    assert gates_on_paper.main(["--run", str(two_originals[0])]) == 0
    derived = list((results / "original-heldout-gates-on-paper").glob("*.json"))
    assert len(derived) == 1
    assert reader_table.main(["--run", str(two_originals[0])]) == 0
    assert list((results / "measurements" / "reader_table").glob("*.json"))


def test_live_p_is_compared_with_saved_p_episode_by_episode():
    from check2.evaluation.runs import live_signal

    records = [{"episode_id": f"e{i}", "p": [0.2, 0.3, 0.5]} for i in range(4)]
    saved = {f"e{i}": [0.2, 0.3, 0.5 + (0.0009 if i == 1 else 0.0)] for i in range(4)}
    result = live_signal.compare(records, saved)
    assert result["agree"] and abs(result["max_abs_difference"] - 0.0009) < 1e-12
    saved["e2"] = [0.2, 0.3011, 0.4989]
    result = live_signal.compare(records, saved)
    assert not result["agree"] and result["beyond_tolerance"] == 1 and result["largest"][0]["episode_id"] == "e2"
    del saved["e3"]
    assert live_signal.compare(records, {k: v for k, v in saved.items() if k != "e2"})["without_saved_p"] == 2
