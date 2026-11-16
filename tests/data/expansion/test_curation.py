"""Checking a curation file against its input and keeping the questions it confirms."""
import json

import pytest

from check2.data.expansion import curation as curation_module
from check2.data.expansion import paths
from check2.data.expansion.curation import apply_curation, check_curation, earlier_curation


def _q(n, key="A", question=None, choices=("w", "x", "y", "z")):
    return {"qid": f"expansion/s/{n}", "question": question or f"Question {n}?", "choices": list(choices), "key": key}


def _a(n, answer="A", problem="none"):
    return {"qid": f"expansion/s/{n}", "answer": answer, "problem": problem, "confidence": "high", "note": ""}


def test_kept_only_where_the_answer_equals_the_key_and_no_problem():
    qs = [_q(0, "A"), _q(1, "B"), _q(2, "C"), _q(3, "D")]
    cur = {"answered_by": "sonnet", "answers": [_a(0, "A"), _a(1, "C"), _a(2, "C", "more_than_one_correct"),
                                                _a(3, "D", "none")]}
    kept, counts = apply_curation(qs, cur)
    assert [q["qid"] for q in kept] == ["expansion/s/0", "expansion/s/3"]
    assert counts["answer_equals_key"] == 3
    assert counts["kept_by_curator"] == 2
    assert counts["problems"]["more_than_one_correct"] == 1


def test_a_curation_for_another_input_is_refused():
    qs = [_q(0), _q(1)]
    with pytest.raises(ValueError, match="differ"):
        check_curation(qs, {"answered_by": "sonnet", "answers": [_a(1), _a(0)]})
    with pytest.raises(ValueError, match="differ"):
        check_curation(qs, {"answered_by": "sonnet", "answers": [_a(0)]})
    with pytest.raises(ValueError, match="differ"):
        check_curation(qs, {"answered_by": "sonnet", "answers": [_a(0), _a(2)]})


def test_a_curation_by_another_model_or_with_a_bad_value_is_refused():
    qs = [_q(0)]
    with pytest.raises(ValueError, match="sonnet"):
        check_curation(qs, {"answered_by": "opus", "answers": [_a(0)]})
    with pytest.raises(ValueError, match="letter"):
        check_curation(qs, {"answered_by": "sonnet", "answers": [_a(0, "E")]})
    with pytest.raises(ValueError, match="problem"):
        check_curation(qs, {"answered_by": "sonnet", "answers": [_a(0, "A", "wrong_key")]})


def test_earlier_answers_join_by_text_only_where_the_options_match(tmp_path, monkeypatch):
    earlier_q = tmp_path / "questions.jsonl"
    earlier_q.write_text("".join(json.dumps(r) + "\n" for r in [
        {"qid": "old/0", "question": "What  is one?", "choices": ["w", "x", "y", "z"], "key": "B"},
        {"qid": "old/1", "question": "What is two?", "choices": ["x", "w", "y", "z"], "key": "A"},
    ]), encoding="utf-8")
    earlier_a = tmp_path / "answers.json"
    earlier_a.write_text(json.dumps({"answered_by": "sonnet", "answers": [
        {"qid": "old/0", "answer": "B", "problem": "none", "confidence": "high", "note": ""},
        {"qid": "old/1", "answer": "A", "problem": "none", "confidence": "high", "note": ""}]}), encoding="utf-8")
    monkeypatch.setattr(paths, "EARLIER_CURATED_QUESTIONS", earlier_q)
    monkeypatch.setattr(paths, "EARLIER_CURATION", earlier_a)
    qs = [_q(0, "B", "what is one?"), _q(1, "A", "What is two?"), _q(2, "A", "What is three?")]
    cur = earlier_curation(qs)
    assert [a["answer"] for a in cur["answers"]] == ["B", None, None]
    assert [a["problem"] for a in cur["answers"]] == ["none", curation_module.NOT_ANSWERED, curation_module.NOT_ANSWERED]
    kept, counts = apply_curation(qs, cur)
    assert [q["qid"] for q in kept] == ["expansion/s/0"]
    assert counts["problems"][curation_module.NOT_ANSWERED] == 2
