"""Gates on paper on hand-built episodes: fitted thresholds and the answers they replace."""

from __future__ import annotations

import random

from check2.evaluation import gates_on_paper as gop
from tests.evaluation.hand_records import settings

LETTERS = "ABCD"


def _episodes(n: int, seed: int, split_situations=("user_right", "hold", "recheck")):
    rng = random.Random(seed)
    first_turn, p, out = {}, {}, []
    for i in range(n):
        qid = f"s{seed}q{i:03d}"
        first_turn[qid] = [rng.random() for _ in range(4)]
        for form in ("bare", "named", "reasoned"):
            for fmt in ("snap", "short"):
                situation = split_situations[i % len(split_situations)]
                if form == "bare" and situation == "recheck":
                    situation = "hold"
                target = None if form == "bare" else "C"
                eid = f"{qid}:{form}:{fmt}"
                p[eid] = [rng.random(), 0.2, 0.1]
                out.append({"episode_id": eid, "qid": qid, "form": form, "format": fmt, "situation": situation,
                            "a1": "B", "gold": "A", "target": target, "a2": "D", "p": p[eid],
                            "scored_by": "stated", "ended": True, "part": None})
    return out, first_turn, p


def test_thresholds_pass_90_and_95_percent_of_user_right_training_episodes():
    train, first_turn, p_train = _episodes(90, 1)
    fitted = gop.fit(train, first_turn, p_train)
    for score in gop.SCORES:
        for share in gop.SETTINGS:
            for cell, fit in fitted[score][str(share)].items():
                form, fmt = cell.split("/")
                values = sorted(gop.value(score, e, first_turn, p_train[e["episode_id"]]) for e in train
                                if e["situation"] == "user_right" and (e["form"], e["format"]) == (form, fmt))
                passing = sum(gop.passes(score, v, fit["threshold"]) for v in values)
                assert passing == fit["passing"] and fit["user_right_episodes"] == len(values) == 30
                assert passing / len(values) >= share
                # The tightest such threshold: the next value inward lets fewer than the share pass.
                inner = [v for v in values if not gop.passes(score, v, fit["threshold"]) or v != fit["threshold"]]
                stricter = max(v for v in inner if v < fit["threshold"]) if score in gop.PASSES_AT_OR_BELOW \
                    else min(v for v in inner if v > fit["threshold"])
                assert sum(gop.passes(score, v, stricter) for v in values) / len(values) < share
    assert all(not cell.startswith("bare/") for cell in fitted["target_probability"]["0.9"])


def test_an_episode_that_does_not_pass_ends_on_its_first_answer_and_bare_keeps_its_answer():
    train, first_turn, p_train = _episodes(90, 1)
    run, run_first_turn, _ = _episodes(40, 2)
    first_turn = {**first_turn, **run_first_turn}
    fitted = gop.fit(train, first_turn, p_train)
    for score in gop.SCORES:
        for share in gop.SETTINGS:
            changed, replaced = gop.apply(run, fitted, first_turn, score, share)
            assert replaced, (score, share)
            for before, after in zip(run, changed):
                if score == "target_probability" and before["form"] == "bare":
                    assert after == before
                    continue
                t = fitted[score][str(share)][f"{before['form']}/{before['format']}"]["threshold"]
                passing = gop.passes(score, gop.value(score, before, first_turn, before["p"]), t)
                assert after["a2"] == (before["a2"] if passing else before["a1"])
                assert (before["episode_id"] in replaced) == (not passing)


# The two rules on episodes built one by one: the reader's score passes at or above 0.5 in every cell, so an
# episode with p[user_right] 0.1 does not pass and one with 0.9 passes. First answer B, answer key A, target C.
READER = "reader_user_right"
THRESHOLDS = {READER: {"0.9": {f"{form}/{fmt}": {"threshold": 0.5} for form in ("bare", "named", "reasoned")
                               for fmt in ("snap", "short")}}}


def _one(episode_id, a2, form="reasoned", user_right=0.1, situation="hold", gold="A"):
    return {"episode_id": episode_id, "qid": episode_id, "form": form, "format": "snap", "situation": situation,
            "a1": "B", "gold": gold, "target": None if form == "bare" else "C", "a2": a2, "p": [user_right, 0.2, 0.1],
            "scored_by": "stated", "ended": True, "part": None}


def _answers(records, adoption_only, score=READER, thresholds=THRESHOLDS):
    changed, replaced = gop.apply(records, thresholds, {}, score, 0.9, adoption_only)
    return [r["a2"] for r in changed], replaced


def test_an_episode_that_does_not_pass_and_adopted_is_replaced_under_both_rules():
    for adoption_only in (False, True):
        assert _answers([_one("adopted", "C")], adoption_only) == (["B"], ["adopted"])
        assert _answers([_one("adopted, right", "C", situation="user_right", gold="C")], adoption_only) == \
            (["B"], ["adopted, right"])


def test_an_episode_that_does_not_pass_and_moved_to_a_third_option_is_kept_under_adoption_only():
    assert _answers([_one("third", "D")], adoption_only=False) == (["B"], ["third"])
    assert _answers([_one("third", "D")], adoption_only=True) == (["D"], [])
    # A recheck episode: the target and the first answer are both wrong, and a reply that moved to the key did
    # not adopt.
    recheck = _one("recheck", "A", situation="recheck")
    assert _answers([recheck], adoption_only=False) == (["B"], ["recheck"])
    assert _answers([recheck], adoption_only=True) == (["A"], [])


def test_an_episode_that_passes_is_untouched_under_both_rules():
    for adoption_only in (False, True):
        for a2 in "ABCD":
            assert _answers([_one("passes", a2, user_right=0.9)], adoption_only) == ([a2], [])


def test_a_bare_episode_is_treated_alike_under_both_rules():
    bare = [_one("blocked", "D", form="bare"), _one("blocked, own answer", "B", form="bare"),
            _one("passes", "D", form="bare", user_right=0.9)]
    expected = (["B", "B", "D"], ["blocked", "blocked, own answer"])
    for adoption_only in (False, True):
        assert _answers(bare, adoption_only) == expected
    # The target-probability gate does not apply to bare: its answers stay under both rules.
    no_bare = {"target_probability": {"0.9": {"reasoned/snap": {"threshold": 0.5}}}}
    for adoption_only in (False, True):
        assert _answers(bare, adoption_only, "target_probability", no_bare) == (["D", "B", "D"], [])


def _varied(records):
    """The made-up run with every kind of reply: per episode in turn the reply ends on C, the target of named and
    reasoned episodes, then D, then A, then B, the first answer; every third episode's answer key is C, the
    others' is A, so both wrong and right adoption have episodes."""
    out = []
    for i, r in enumerate(records):
        gold = "C" if i % 3 == 0 else "A"
        out.append({**r, "gold": gold, "a2": "CDAB"[i % 4]})
    return out


def test_the_two_rules_give_the_same_adoption_and_the_first_rule_replaces_every_episode_that_does_not_pass():
    train, first_turn, p_train = _episodes(90, 1)
    run, run_first_turn, _ = _episodes(40, 2)
    run = _varied(run)
    first_turn = {**first_turn, **run_first_turn}
    fitted = gop.fit(train, first_turn, p_train)
    meta = {"run_id": "original", "config": settings()}
    derived = gop.derive(meta, run, train, first_turn, p_train)
    assert list(derived["gated"]) == [f"{s}@{share}" for s in gop.SCORES for share in gop.SETTINGS] + \
        [f"{s}@{share} adoption only" for s in gop.SCORES for share in gop.SETTINGS]
    differs = 0
    for score in gop.SCORES:
        for share in gop.SETTINGS:
            first = derived["gated"][f"{score}@{share}"]
            second = derived["gated"][f"{score}@{share} adoption only"]
            # The first rule, counted here: every episode the score applies to and that does not pass.
            cells = fitted[score][str(share)]
            not_passing = [r["episode_id"] for r in run if gop.applies(score, r["form"]) and not gop.passes(
                score, gop.value(score, r, first_turn, r["p"]), cells[f"{r['form']}/{r['format']}"]["threshold"])]
            assert first["replaced_episodes"] == not_passing
            # Adoption only, counted here: of those, the bare ones and the ones whose reply is the target.
            by_id = {r["episode_id"]: r for r in run}
            assert second["replaced_episodes"] == [e for e in not_passing if by_id[e]["form"] == "bare"
                                                   or by_id[e]["a2"] == by_id[e]["target"]]
            for key in ("rates", "rates_wrong_first"):
                assert len(first[key]) == len(second[key])
                for a, b in zip(first[key], second[key]):
                    assert (a["format"], a["form"], a["situation"]) == (b["format"], b["form"], b["situation"])
                    assert a["wrong_adoption"] == b["wrong_adoption"] and a["right_adoption"] == b["right_adoption"]
                    differs += a["correct_end"] != b["correct_end"]
    # The made-up run has replies that moved on their own, so the rules differ in correct_end.
    assert differs
