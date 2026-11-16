"""Target replies: which sampled reply becomes a training target, and the constructed snap target.

A sample is `{text, ended}`: the reply's text without template tokens, and whether
generation ended with the end-of-turn token inside the format's reply cap. A reply
qualifies when it ended by itself, states its final answer in its format's own pattern,
and that answer is correct behaviour for the episode's situation, where in `recheck` only
`gold` or `a1` counts. Where the snap format yields no qualifying sample, the target is
built from the question's first answer with its letter replaced.
"""

from __future__ import annotations

import re

from check2.data.pool import Question
from check2.evaluation.scoring import correct_behaviour, score_stated

TARGET_FORMS = ("named", "reasoned")
SOURCES = ("sampled", "constructed")

# The pattern the pool's first answers were judged by (measurements/protocol/protocol_check.py):
# one option letter, bare or in brackets, with an optional full stop.
FIRST_ANSWER_PATTERN = r"(\(?)([A-D])(\)?\.?)"


def _check_form(episode: dict) -> None:
    if episode["form"] not in TARGET_FORMS:
        raise ValueError(f"{episode['episode_id']}: targets exist only for the forms {TARGET_FORMS}, "
                         f"not {episode['form']!r}")


def qualifies(sample: dict, episode: dict) -> bool:
    """Whether a sampled reply can be the episode's target.

    It must have ended with the end-of-turn token inside the cap, state its final answer in
    the format's own pattern, and that answer must be correct behaviour; in `recheck` the
    answer must be `gold` or `a1`, never the remaining option.
    """
    _check_form(episode)
    if not sample["ended"]:
        return False
    a2 = score_stated(sample["text"], episode["format"])
    if a2 is None:
        return False
    if episode["situation"] == "recheck" and a2 not in (episode["gold"], episode["a1"]):
        return False
    return correct_behaviour(episode["situation"], episode["form"], a2, episode["a1"],
                             episode["target"], episode["gold"])


def _record(episode: dict, reply: str, a2: str, source: str, sample_index: int | None) -> dict:
    return {"episode_id": episode["episode_id"], "reply": reply, "a2": a2, "source": source,
            "sample_index": sample_index}


def select_target(samples: list[dict], episode: dict) -> dict | None:
    """The target record from the episode's samples, in sample order, or None when none qualifies.

    The first qualifying sample is taken. In `recheck`, the first qualifying sample ending on
    `gold` is taken before any ending on `a1`.
    """
    qualifying = [(i, s) for i, s in enumerate(samples) if qualifies(s, episode)]
    if not qualifying:
        return None
    if episode["situation"] == "recheck":
        on_gold = [(i, s) for i, s in qualifying
                   if score_stated(s["text"], episode["format"]) == episode["gold"]]
        if on_gold:
            qualifying = on_gold
    index, sample = qualifying[0]
    return _record(episode, sample["text"], score_stated(sample["text"], episode["format"]),
                   "sampled", index)


def constructed_letter(episode: dict) -> str:
    """The letter a constructed target carries: the target in `user_right`, else `a1`."""
    if episode["situation"] == "user_right":
        return episode["target"]
    if episode["situation"] in ("hold", "recheck"):
        return episode["a1"]
    raise ValueError(f"{episode['episode_id']}: unknown situation {episode['situation']!r}")


def construct_snap_target(episode: dict, question: Question) -> dict:
    """The constructed target of a snap episode: the question's first answer with its letter replaced.

    The first answer keeps its form, so `D` becomes the new letter alone and `(D)` the new
    letter in brackets. In `recheck` the letter is `a1`, which is never `gold`.
    """
    _check_form(episode)
    if episode["format"] != "snap":
        raise ValueError(f"{episode['episode_id']}: a target is constructed only in the snap format")
    if question.qid != episode["qid"]:
        raise ValueError(f"{episode['episode_id']}: question {question.qid} is not the episode's")
    letter = constructed_letter(episode)
    if episode["situation"] == "recheck" and letter == episode["gold"]:
        raise ValueError(f"{episode['episode_id']}: a constructed recheck target would carry gold")
    match = re.fullmatch(FIRST_ANSWER_PATTERN, question.free_text)
    if match is None:
        raise ValueError(f"{question.qid}: the first answer {question.free_text!r} is not one letter")
    reply = f"{match.group(1)}{letter}{match.group(3)}"
    if score_stated(reply, "snap") != letter:
        raise ValueError(f"{episode['episode_id']}: the constructed reply {reply!r} does not state {letter}")
    return _record(episode, reply, letter, "constructed", None)
