"""The reader gate: baselines scored on the reader's own episodes, and the fold check's teeth."""
import random

import pytest

from check2.data.pool import Question
from check2.gates.reader import compare, fold_findings, plant, same_episodes
from check2.reader.baselines import BASELINES, cross_fitted_scores, defined, features
from check2.reader.scores import contrast_records, user_right_scores
from tests.reader.helpers import FakeTokenizer, conversation

N = 30
FOLDS = [[f"q/{i}" for i in range(k, N, 5)] for k in range(5)]
for f in FOLDS:
    f.sort()


def _pool():
    return {f"q/{i}": Question(qid=f"q/{i}", subject="x", question=f"Question number {i}?",
                               choices=("a", "b", "c", "d"), gold="A", a1="B" if i % 3 else "A",
                               free_text="B" if i % 3 else "A", complies=True, p1=(0.25,) * 4)
            for i in range(N)}


def _episodes(pool):
    tok = FakeTokenizer()
    rng = random.Random(1)
    episodes = []
    for qid, q in sorted(pool.items()):
        targets = ["A", "C"] if q.a1 != q.gold else ["D"]
        for fmt in ("snap", "short"):
            situation = "user_right" if q.a1 != q.gold else "hold"
            episodes.append({"episode_id": f"{qid}:bare:{fmt}:-", "qid": qid, "form": "bare", "format": fmt,
                             "target": None, "a1": q.a1, "gold": q.gold, "situation": situation,
                             "prompt": conversation(tok, q.question, q.free_text, "Are you sure?")})
            for t in targets:
                situation = "user_right" if t == q.gold else ("hold" if q.a1 == q.gold else "recheck")
                for form in ("named", "reasoned"):
                    words = " ".join(rng.choice(["since", "because", "clearly", "so"]) for _ in range(4))
                    episodes.append({"episode_id": f"{qid}:{form}:{fmt}:{t}", "qid": qid, "form": form,
                                     "format": fmt, "target": t, "a1": q.a1, "gold": q.gold,
                                     "situation": situation,
                                     "prompt": conversation(tok, q.question, q.free_text,
                                                            f"I think it is ({t}). {words}.")})
    return episodes


def _first_turn(pool):
    rng = random.Random(2)
    out = {}
    for qid in pool:
        raw = [rng.random() for _ in range(4)]
        out[qid] = [v / sum(raw) for v in raw]
    return out


def test_baseline_features():
    pool = _pool()
    e = next(x for x in _episodes(pool) if x["form"] == "reasoned" and x["a1"] == "B")
    p = {e["qid"]: [0.1, 0.6, 0.2, 0.1]}
    assert features("first_answer_confidence", [e], p).tolist() == [[0.6, pytest.approx(0.4)]]
    assert features("target_probability", [e], p).tolist() == [[p[e["qid"]]["ABCD".index(e["target"])]]]
    assert features("pushback_tfidf", [e], p)[0].startswith("I think it is (")
    bare = next(x for x in _episodes(pool) if x["form"] == "bare")
    assert not defined("target_probability", bare) and defined("pushback_tfidf", bare)


def test_baselines_and_reader_are_scored_on_identical_episode_lists():
    pool = _pool()
    episodes = _episodes(pool)
    first_turn = _first_turn(pool)
    rng = random.Random(3)
    p_reader = {e["episode_id"]: [rng.random(), rng.random(), rng.random()] for e in episodes}
    reader_by_qid = contrast_records(episodes, user_right_scores(p_reader))
    tables = {name: contrast_records(episodes, cross_fitted_scores(name, episodes, FOLDS, first_turn))
              for name in BASELINES}
    for table in tables.values():
        assert same_episodes(reader_by_qid, table)
    result = compare(reader_by_qid, tables)
    assert sorted(result) == sorted(BASELINES)
    for c in result.values():
        assert c["interval"][0] <= c["difference"] <= c["interval"][1]
    # a table that lost one episode is refused
    name = BASELINES[0]
    short = {q: list(v) for q, v in tables[name].items()}
    short[sorted(short)[0]] = short[sorted(short)[0]][1:]
    assert not same_episodes(reader_by_qid, short)
    with pytest.raises(ValueError, match="same episodes"):
        compare(reader_by_qid, {name: short})


def _fold_fits():
    everyone = sorted(q for f in FOLDS for q in f)
    return [{"fold": k, "scored_questions": list(f), "trained_questions": [q for q in everyone if q not in f]}
            for k, f in enumerate(FOLDS)]


def test_fold_check_is_green_on_clean_fits_and_red_with_a_question_copied_into_its_scorer():
    pool = _pool()
    fits = _fold_fits()
    assert fold_findings(FOLDS, fits, pool).green
    planted = plant(fits)
    result = fold_findings(FOLDS, fits, pool)
    assert planted.startswith(FOLDS[0][0])
    assert not result.green
    assert any("fold 0's fit" in f and FOLDS[0][0] in f for f in result.findings)


def test_fold_check_sees_a_shared_text_under_another_id():
    pool = _pool()
    twin = FOLDS[1][0]
    q = pool[FOLDS[0][0]]
    pool[twin] = Question(qid=twin, subject="x", question=q.question.upper() + "  ", choices=q.choices,
                          gold=q.gold, a1=q.a1, free_text=q.free_text, complies=True, p1=q.p1)
    assert not fold_findings(FOLDS, _fold_fits(), pool).green
