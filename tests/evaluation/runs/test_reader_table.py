"""The reader's score table on recorded p, against AUROCs counted pair by pair."""

from __future__ import annotations

import json
import random

import pytest

from check2.data import paths
from check2.evaluation import reader_table

SITUATIONS = ("user_right", "hold", "recheck")


def _read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _pairwise_auroc(scores, positives):
    pos = [s for s, y in zip(scores, positives) if y]
    neg = [s for s, y in zip(scores, positives) if not y]
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def test_the_reader_table_matches_aurocs_counted_by_hand():
    rng = random.Random(7)
    heldout = _read(paths.EPISODES / "heldout.jsonl")
    train = rng.sample(_read(paths.EPISODES / "train.jsonl"), 900)
    first_turn = {r["qid"]: r["probabilities"] for r in _read(paths.DATA / "reader" / "first_turn.jsonl")}
    records = []
    for e in heldout:
        k = SITUATIONS.index(e["situation"])
        raw = [rng.random() + (0.8 if i == k else 0.0) for i in range(3)]
        total = sum(raw)
        records.append({"episode_id": e["episode_id"], "qid": e["qid"], "situation": e["situation"],
                        "p": [v / total for v in raw]})
    values = reader_table.table(records, {e["episode_id"]: e for e in heldout}, train, first_turn)
    for k, name in enumerate(SITUATIONS):
        expected = _pairwise_auroc([r["p"][k] for r in records], [r["situation"] == name for r in records])
        got = values["reader"]["one_vs_rest"][name]
        assert abs(got["estimate"] - expected) < 1e-12
        assert got["low"] <= got["estimate"] <= got["high"]
    by_id = {e["episode_id"]: e for e in heldout}
    per_format = []
    for fmt in ("snap", "short"):
        rows = [r for r in records if by_id[r["episode_id"]]["form"] == "reasoned"
                and by_id[r["episode_id"]]["format"] == fmt and by_id[r["episode_id"]]["a1"] != by_id[r["episode_id"]]["gold"]]
        per_format.append(_pairwise_auroc([r["p"][0] for r in rows], [r["situation"] == "user_right" for r in rows]))
    assert abs(values["reader"]["selection"]["estimate"] - sum(per_format) / 2) < 1e-12
    for name, b in values["baselines"].items():
        assert abs(b["reader_minus_baseline"]["estimate"]
                   - (values["reader"]["selection"]["estimate"] - b["selection"]["estimate"])) < 1e-12
    assert set(values["baselines"]) == {"first_answer_confidence", "target_probability", "pushback_tfidf"}


def _test_pairs_and_p(rng):
    """The reasoned test episodes, and a made-up reason-only p per pair that leans to the pair's situation."""
    episodes = [e for e in _read(paths.EPISODES / "test.jsonl") if e["form"] == "reasoned"]
    p = {}
    for e in episodes:
        key = f"{e['qid']}:{e['target']}"
        if key not in p:
            k = SITUATIONS.index(e["situation"])
            raw = [rng.random() + (0.5 if i == k else 0.0) for i in range(3)]
            p[key] = [v / sum(raw) for v in raw]
    return episodes, p


def test_the_reason_only_score_on_test_is_counted_over_pairs(tmp_path):
    episodes, p = _test_pairs_and_p(random.Random(3))
    saved = tmp_path / "p_reason_only_test.jsonl"
    saved.write_text("".join(json.dumps({"episode_id": k, "p": v}) + "\n" for k, v in sorted(p.items())),
                     encoding="utf-8")
    result = reader_table.reason_only_score(episodes, saved)
    # By hand: one entry per pair of a wrong-first question, user_right against recheck, by p[user_right].
    contrast = {}
    for e in episodes:
        if e["a1"] != e["gold"]:
            contrast[f"{e['qid']}:{e['target']}"] = e["situation"] == "user_right"
    expected = _pairwise_auroc([p[k][0] for k in sorted(contrast)], [contrast[k] for k in sorted(contrast)])
    assert result["exists"] and result["pairs"] == len(p)
    assert abs(result["selection"]["estimate"] - expected) < 1e-12
    assert result["selection"]["low"] <= result["selection"]["estimate"] <= result["selection"]["high"]
    del p[sorted(contrast)[0]]
    saved.write_text("".join(json.dumps({"episode_id": k, "p": v}) + "\n" for k, v in sorted(p.items())),
                     encoding="utf-8")
    with pytest.raises(ValueError, match="has no p for 1 pairs"):
        reader_table.reason_only_score(episodes, saved)


def test_without_the_saved_file_the_record_says_so(tmp_path):
    result = reader_table.reason_only_score([], tmp_path / "p_reason_only_test.jsonl")
    assert result["exists"] is False and "does not exist" in result["note"]
