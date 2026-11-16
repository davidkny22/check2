"""Small runs draw their episodes as a gate does; folds give each fit its episodes."""
import random

import pytest

from check2.reader.files import p_rows, small_draw
from check2.reader.folds import fold_of, fold_parts


def test_small_draw_takes_n_per_format_by_the_gate_draw():
    episodes = [{"episode_id": f"q/{i}:bare:{fmt}:-", "format": fmt, "qid": f"q/{i}"}
                for i in range(20) for fmt in ("snap", "short")]
    drawn = small_draw(episodes, 3)
    assert len(drawn) == 6
    for fmt in ("snap", "short"):
        ids = sorted(e["episode_id"] for e in episodes if e["format"] == fmt)
        assert sorted(e["episode_id"] for e in drawn if e["format"] == fmt) == sorted(random.Random(2035).sample(ids, 3))
    assert small_draw(episodes, None) is episodes


def test_fold_parts_split_by_question():
    episodes = [{"episode_id": f"{q}:{f}", "qid": q} for q in ("a", "b", "c") for f in ("x", "y")]
    folds = [["a"], ["b", "c"]]
    assert fold_of(folds) == {"a": 0, "b": 1, "c": 1}
    assert fold_parts(episodes, folds, 0) == (["b:x", "b:y", "c:x", "c:y"], ["a:x", "a:y"])
    with pytest.raises(ValueError):
        fold_parts(episodes + [{"episode_id": "d:x", "qid": "d"}], folds, 0)
    with pytest.raises(ValueError):
        fold_of([["a"], ["a"]])


def test_p_rows_are_sorted_records():
    assert p_rows({"b": [0.2, 0.3, 0.5], "a": [1, 0, 0]}) == [{"episode_id": "a", "p": [1.0, 0.0, 0.0]},
                                                             {"episode_id": "b", "p": [0.2, 0.3, 0.5]}]
