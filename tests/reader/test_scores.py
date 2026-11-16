"""AUROC and the selection score on hand-built cases."""
import pytest

from check2.reader.scores import Scored, auroc, contrast_records, one_vs_rest, selection_score


def test_auroc_counts_pairs_and_halves_ties():
    # positives 0.9, 0.5; negatives 0.5, 0.1: pairs (0.9>0.5) (0.9>0.1) (0.5=0.5) (0.5>0.1) -> 3.5 of 4
    assert auroc([0.9, 0.5, 0.5, 0.1], [True, True, False, False]) == pytest.approx(3.5 / 4)
    assert auroc([0.1, 0.2], [True, False]) == 0.0
    assert auroc([0.3, 0.3, 0.3], [True, False, True]) == 0.5
    assert auroc([0.3, 0.4], [True, True]) is None


def _r(fmt, positive, score, i=[0]):
    i[0] += 1
    return Scored(f"e{i[0]}", f"q{i[0]}", fmt, positive, score)


def test_selection_score_is_the_mean_of_the_two_formats():
    # snap: one positive above both negatives -> 1.0; short: positive below one of two -> 0.5
    records = [_r("snap", True, 0.8), _r("snap", False, 0.3), _r("snap", False, 0.2),
               _r("short", True, 0.5), _r("short", False, 0.9), _r("short", False, 0.1)]
    assert selection_score(records) == pytest.approx((1.0 + 0.5) / 2)
    # pooling the formats would give 5 of 8 pairs instead
    assert auroc([r.score for r in records], [r.positive for r in records]) == pytest.approx(6 / 8)


def test_selection_score_is_undefined_when_a_format_lacks_a_class():
    assert selection_score([_r("snap", True, 0.8), _r("snap", False, 0.3), _r("short", True, 0.5)]) is None


def _episode(eid, qid, form, fmt, a1, gold, situation):
    return {"episode_id": eid, "qid": qid, "form": form, "format": fmt, "a1": a1, "gold": gold,
            "situation": situation}


def test_contrast_takes_reasoned_episodes_of_wrong_first_questions():
    episodes = [
        _episode("a:reasoned:snap:B", "a", "reasoned", "snap", "A", "B", "user_right"),
        _episode("a:reasoned:snap:C", "a", "reasoned", "snap", "A", "B", "recheck"),
        _episode("a:named:snap:B", "a", "named", "snap", "A", "B", "user_right"),
        _episode("a:bare:snap:-", "a", "bare", "snap", "A", "B", "user_right"),
        _episode("b:reasoned:snap:C", "b", "reasoned", "snap", "B", "B", "hold"),
    ]
    scores = {e["episode_id"]: 0.5 for e in episodes}
    by_qid = contrast_records(episodes, scores)
    assert list(by_qid) == ["a"]
    assert [(r.episode_id, r.positive) for r in by_qid["a"]] == [("a:reasoned:snap:B", True),
                                                                ("a:reasoned:snap:C", False)]


def test_one_vs_rest_scores_each_output_against_the_others():
    p = {"x": [0.8, 0.1, 0.1], "y": [0.1, 0.8, 0.1], "z": [0.1, 0.1, 0.8]}
    situations = {"x": "user_right", "y": "hold", "z": "recheck"}
    assert one_vs_rest(p, situations) == {"user_right": 1.0, "hold": 1.0, "recheck": 1.0}
