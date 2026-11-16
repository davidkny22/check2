"""Dropping seeds, pool questions and repeats from a subject's generated rows."""
import json

from check2.data.expansion.generated import parse_subject
from check2.data.expansion.repeats import drop_repeats
from check2.data.pool import normalise


def _row(question, options, key):
    return {"original_prompt": question + "\n\n" + "\n\n".join(options), "original_completion": key}


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_drop_repeats_keeps_the_first_and_extends_seen():
    seen = {normalise("Old question?")}
    qs = [{"question": "New  question?"}, {"question": "old question?"}, {"question": "new question?"}]
    kept, repeated = drop_repeats(qs, seen)
    assert kept == [qs[0]]
    assert repeated == [qs[1], qs[2]]
    assert normalise("New question?") in seen


def test_parse_subject_drops_seeds_repeats_and_bad_rows(tmp_path):
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write(first, [
        _row("Seed question?", ["w", "x", "y", "z"], "A"),
        _row("Fresh one?", ["w", "x", "y", "z"], "B"),
        {"original_prompt": "Run together w x y z", "original_completion": "A"},
        _row("Bad key?", ["w", "x", "y", "z"], "E"),
    ])
    _write(second, [
        _row("fresh  ONE?", ["p", "q", "r", "s"], "C"),
        _row("Fresh two?", ["p", "q", "r", "s"], "D"),
    ])
    kept, dropped, counts = parse_subject("algebra", [first, second], {normalise("Seed question?")})
    assert [q["qid"] for q in kept] == ["expansion/algebra/1", "expansion/algebra/5"]
    assert kept[1] == {"qid": "expansion/algebra/5", "subject": "algebra", "question": "Fresh two?",
                       "choices": ["p", "q", "r", "s"], "key": "D", "source": "b.jsonl", "row": 1}
    assert [(d["qid"], d["why"]) for d in dropped] == [
        ("expansion/algebra/0", "pool_or_seed"), ("expansion/algebra/2", "does_not_split"),
        ("expansion/algebra/3", "key_not_a_letter"), ("expansion/algebra/4", "repeat")]
    assert counts == {"rows": 6, "does_not_split": 1, "key_not_a_letter": 1, "usable": 4,
                      "dropped_pool_or_seed": 1, "dropped_repeat": 1, "for_curation": 2}
