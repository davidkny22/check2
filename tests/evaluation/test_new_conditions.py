"""The reader and its gate on paper under new conditions, on hand-built records with hand-computed figures."""
import json

import pytest

from check2.evaluation import new_conditions

THRESHOLDS = {"reader_user_right": {
    "0.9": {"reasoned/snap": {"threshold": 0.5}, "reasoned/short": {"threshold": 0.5}},
    "0.95": {"reasoned/snap": {"threshold": 0.2}, "reasoned/short": {"threshold": 0.2}}}}


def _record(qid, fmt, target, gold, a1, a2, p_right):
    situation = "user_right" if target == gold else ("hold" if a1 == gold else "recheck")
    return {"episode_id": f"{qid}:reasoned:{fmt}:{target}", "qid": qid, "form": "reasoned", "format": fmt,
            "target": target, "gold": gold, "a1": a1, "a2": a2, "situation": situation,
            "p": [p_right, (1 - p_right) / 2, (1 - p_right) / 2]}


def _records():
    rows = []
    for fmt in ("snap", "short"):
        # Four wrong-first questions: the gold pair is adopted, the wrong pair adopted on two of them.
        for i, (adopt_wrong, p_gold, p_wrong) in enumerate([(True, 0.9, 0.3), (True, 0.6, 0.1),
                                                           (False, 0.7, 0.4), (False, 0.8, 0.05)]):
            q = f"s/{i}"
            rows.append(_record(q, fmt, "A", "A", "B", "A", p_gold))
            rows.append(_record(q, fmt, "C", "A", "B", "C" if adopt_wrong else "B", p_wrong))
        # One right-first question with a wrong target, adopted, with a low p.
        rows.append(_record("s/9", fmt, "D", "A", "A", "D", 0.1))
    return rows


def test_adoption_rates_by_hand():
    rates = new_conditions.adoption_rates(_records())
    # Wrong-target episodes per format: four recheck (two adopted) and one hold (adopted): 3 of 5.
    assert rates["snap"]["all"]["wrong_adoption"]["estimate"] == pytest.approx(3 / 5)
    assert rates["snap"]["all"]["wrong_adoption"]["episodes"] == 5
    assert rates["snap"]["all"]["right_adoption"]["estimate"] == pytest.approx(1.0)
    assert rates["short"]["wrong_first"]["wrong_adoption"]["estimate"] == pytest.approx(2 / 4)


def test_the_gate_replaces_answers_below_the_fitted_threshold_and_nothing_is_fitted_here(tmp_path):
    saved = tmp_path / "p.jsonl"
    saved.write_text("".join(json.dumps({"episode_id": f"{q}:{t}", "p": [x, 0.5 - x / 2, 0.5 - x / 2]}) + "\n"
                             for q, t, x in [(f"s/{i}", "A", 0.8) for i in range(4)] + [(f"s/{i}", "C", 0.2) for i in range(4)]
                             + [("s/9", "D", 0.3)]), encoding="utf-8")
    rows = new_conditions.one_set(_records(), THRESHOLDS, saved)
    gate = rows["reader_gate"]["0.9"]
    # At threshold 0.5 every wrong-target episode (p 0.3, 0.1, 0.4, 0.05, 0.1) is replaced by its first answer;
    # every gold-target episode passes (p 0.9, 0.6, 0.7, 0.8).
    assert gate["snap"]["all"]["wrong_adoption"]["estimate"] == 0
    assert gate["snap"]["all"]["right_adoption"]["estimate"] == pytest.approx(1.0)
    assert gate["replaced_episodes"] == 10
    # At threshold 0.2 only the episodes at 0.1 and 0.05 are replaced: of the 3 adoptions, s/1's and s/9's go.
    loose = rows["reader_gate"]["0.95"]
    assert loose["snap"]["all"]["wrong_adoption"]["estimate"] == pytest.approx(1 / 5)
    # The reader separates user_right (p 0.6 to 0.9) from recheck (p 0.05 to 0.4) perfectly.
    assert rows["contrast_questions"] == 4
    assert rows["reader_selection"]["estimate"] == pytest.approx(1.0)
    assert rows["reason_only"]["selection"]["estimate"] == pytest.approx(1.0)
    assert rows["original"]["snap"]["all"]["wrong_adoption"]["estimate"] == pytest.approx(3 / 5)


def test_only_reasoned_episodes_enter(tmp_path):
    saved = tmp_path / "p.jsonl"
    saved.write_text("", encoding="utf-8")
    bare = {**_records()[-1], "form": "bare", "episode_id": "s/9:bare:snap:-", "target": None,
            "situation": "hold", "a2": "B"}
    reasoned = [r for r in _records() if r["a1"] == r["gold"]]
    rows = new_conditions.one_set([bare, *reasoned], THRESHOLDS, saved)
    assert rows["episodes"] == len(reasoned) == 2
    assert rows["original"]["snap"]["all"]["wrong_adoption"]["estimate"] == pytest.approx(1.0)
    assert rows["reader_gate"]["0.9"]["replaced_episodes"] == 2
