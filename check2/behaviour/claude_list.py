"""Which conversations a Claude model is asked: the whole list, or for Opus a seeded third of its questions.

The list is the one every model gets, built by `check2.behaviour.items`. Opus answers a
seeded third of the questions with all their items. When a model's measured rate projects
an end past the deadline, its list is cut by a seeded draw of questions.
"""

from __future__ import annotations

import random
from collections.abc import Mapping

from check2.behaviour import items as item_lists
from check2.behaviour.backend import Item

THIRD_MODELS = ("claude-opus-5-5",)
THIRD_SEED = 2052
CUT_SEED = 2053


def asked_qids(model: str) -> list[str]:
    """The sorted question ids the model is asked."""
    qids = sorted(i.qid for i in item_lists.first_answer_items())
    if model in THIRD_MODELS:
        qids = sorted(random.Random(THIRD_SEED).sample(qids, round(len(qids) / 3)))
    return qids


def first_items(model: str) -> list[Item]:
    asked = set(asked_qids(model))
    return [i for i in item_lists.first_answer_items() if i.qid in asked]


def follow_ups(first_replies: Mapping[str, str]) -> list[Item]:
    return item_lists.follow_up_items(first_replies)


def cut_questions(open_qids: list[str], calls_of: Mapping[str, int], budget: int) -> list[str]:
    """The questions kept when only `budget` more calls fit before the deadline.

    Questions are taken in a seeded random order of the sorted ids, each with all its
    calls, while the calls taken stay within the budget. Returns the kept ids, sorted.
    """
    order = random.Random(CUT_SEED).sample(sorted(open_qids), len(open_qids))
    kept, used = [], 0
    for qid in order:
        if used + calls_of[qid] > budget:
            continue
        kept.append(qid)
        used += calls_of[qid]
    return sorted(kept)
