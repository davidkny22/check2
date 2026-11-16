"""The conversations put to a model with nothing added, and the labels its own first answers give them.

A model first answers every question of the pool and of a seeded sample of the expanded set.
Each reasoned pushback and each neutral follow-up is then built on that model's own first
reply: the first assistant turn is the reply it gave. A question whose first reply is not
exactly one option letter gets no follow-up, and a pair whose target is the model's own
first answer is dropped. Right-first, wrong-first and each pair's situation follow from the
model's own first answer.

The turn texts are those of the reviewed episodes; the pairs of the `test` questions are
read from the episode files, so that the pushback is word for word the one the other runs
used.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from check2.behaviour.backend import Item
from check2.data import paths
from check2.data.episodes import FORMATS, SYSTEM, episode_id, first_user_turn, pushback
from check2.data.expansion import paths as expansion_paths
from check2.data.first_answers import MAX_NEW_TOKENS as FIRST_ANSWER_CAP
from check2.data.pairs import situation
from check2.data.pool import Question, load_pool
from check2.data.sampling import REPLY_CAP
from check2.evaluation.scoring import score_stated

EXPANDED_SAMPLE_SEED = 2050
EXPANDED_SAMPLE_SIZE = 1000

SOURCES = ("pool", "expanded")
KINDS = ("first", "reasoned", "neutral")

TEST_EPISODES = paths.EPISODES / "test.jsonl"
CROSSWRITER_EPISODES = paths.EPISODES / "crosswriter_test.jsonl"
NEUTRAL_EPISODES = paths.EPISODES / "neutral_test.jsonl"


@dataclass(frozen=True)
class AskedQuestion:
    """A question as the model sees it, with its key and where it comes from."""

    qid: str
    source: str
    question: str
    choices: tuple[str, str, str, str]
    gold: str


@dataclass(frozen=True)
class AskedPair:
    """A (question, target) pair with one written reason."""

    qid: str
    target: str
    reason: str
    writer: str


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def expanded_sample(kept_ids: list[str]) -> list[str]:
    """The sampled expanded questions: `random.Random(2050).sample` of the sorted kept ids, sorted."""
    return sorted(random.Random(EXPANDED_SAMPLE_SEED).sample(sorted(kept_ids), EXPANDED_SAMPLE_SIZE))


@lru_cache(maxsize=1)
def questions() -> dict[str, AskedQuestion]:
    """Every question that is asked: the whole pool, then the sampled expanded questions, by id."""
    asked: dict[str, AskedQuestion] = {}
    for q in sorted(load_pool(), key=lambda q: q.qid):
        asked[q.qid] = AskedQuestion(q.qid, "pool", q.question, q.choices, q.gold)
    kept = {r["qid"]: r for r in _read_jsonl(expansion_paths.QUESTIONS)}
    for qid in expanded_sample(list(kept)):
        r = kept[qid]
        asked[qid] = AskedQuestion(qid, "expanded", r["question"], tuple(r["choices"]), r["key"])
    return asked


@lru_cache(maxsize=1)
def pairs() -> tuple[AskedPair, ...]:
    """The pairs that get reasoned pushback, sorted by question, target and writer.

    For the `test` questions: every reasoned pair of the episode files, with the second
    reason of the cross-writer pairs as a pair of its own writer. For the sampled expanded
    questions: every pair of the expanded set's pairs file.
    """
    found: dict[tuple[str, str, str], AskedPair] = {}
    for path in (TEST_EPISODES, CROSSWRITER_EPISODES):
        for e in _read_jsonl(path):
            if e["form"] != "reasoned":
                continue
            found.setdefault((e["qid"], e["target"], e["writer"]),
                             AskedPair(e["qid"], e["target"], e["reason"], e["writer"]))
    sampled = {qid for qid, q in questions().items() if q.source == "expanded"}
    for r in _read_jsonl(expansion_paths.PAIRS):
        if r["qid"] in sampled:
            key = (r["qid"], r["target"], r["writer"])
            if key in found:
                raise ValueError(f"the pair {key} appears twice in the expanded set's pairs")
            found[key] = AskedPair(r["qid"], r["target"], r["reason"], r["writer"])
    writer_order = {"opus": 0, "sonnet": 1}
    return tuple(sorted(found.values(), key=lambda p: (p.qid, p.target, writer_order[p.writer])))


@lru_cache(maxsize=1)
def neutral_qids() -> tuple[str, ...]:
    """The `test` questions that get the neutral follow-up."""
    return tuple(sorted({e["qid"] for e in _read_jsonl(NEUTRAL_EPISODES)}))


def _question_for_turn(q: AskedQuestion) -> Question:
    return Question(qid=q.qid, subject="", question=q.question, choices=q.choices, gold=q.gold,
                    a1="", free_text="", complies=False, p1=(0.0, 0.0, 0.0, 0.0))


def _opening(q: AskedQuestion) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": first_user_turn(_question_for_turn(q))}]


def first_item_id(qid: str) -> str:
    return f"{qid}:first"


def first_answer(reply: str) -> str | None:
    """The option letter of a first reply that is exactly one letter by the snap pattern, else None."""
    return score_stated(reply, "snap")


def first_answer_items() -> list[Item]:
    """One item per asked question: the system turn and the first user turn, in question order."""
    return [Item(item_id=first_item_id(q.qid), qid=q.qid, source=q.source, kind="first", fmt=None,
                 target=None, writer=None, messages=tuple(_opening(q)),
                 max_new_tokens=FIRST_ANSWER_CAP)
            for q in questions().values()]


def _follow_up(q: AskedQuestion, first_reply: str, kind: str, fmt: str, target: str | None,
               reason: str | None, writer: str | None) -> Item:
    form = "reasoned" if kind == "reasoned" else "neutral"
    messages = _opening(q) + [{"role": "assistant", "content": first_reply.strip()},
                              {"role": "user", "content": pushback(form, fmt, target, reason)}]
    return Item(item_id=episode_id(q.qid, form, fmt, target, writer), qid=q.qid, source=q.source,
                kind=kind, fmt=fmt, target=target, writer=writer, messages=tuple(messages),
                max_new_tokens=REPLY_CAP[fmt])


def follow_up_items(first_replies: Mapping[str, str]) -> list[Item]:
    """The reasoned pushback and neutral follow-up items built on the model's own first replies.

    `first_replies` maps a question id to the text of the model's first reply. A question
    with no entry was not asked and gets nothing. A question whose reply is not exactly one
    option letter gets no follow-up. A pair whose target is the model's first answer is
    dropped. Items come in question order; within a question, pairs by target and writer,
    then the neutral follow-up, each in the snap format before the short.
    """
    asked = questions()
    pairs_of: dict[str, list[AskedPair]] = {}
    for p in pairs():
        pairs_of.setdefault(p.qid, []).append(p)
    neutral = set(neutral_qids())
    items: list[Item] = []
    for qid, q in asked.items():
        if qid not in first_replies:
            continue
        reply = first_replies[qid]
        a1 = first_answer(reply)
        if a1 is None:
            continue
        for p in pairs_of.get(qid, []):
            if p.target == a1:
                continue
            for fmt in FORMATS:
                items.append(_follow_up(q, reply, "reasoned", fmt, p.target, p.reason, p.writer))
        if qid in neutral:
            for fmt in FORMATS:
                items.append(_follow_up(q, reply, "neutral", fmt, None, None, None))
    return items


def labels(first_replies: Mapping[str, str]) -> dict[str, dict]:
    """What the model's own first replies make of each question and each reasoned pair.

    Returns two maps. `questions`, for every asked question with a first reply: its
    `source`, `gold`, the model's first answer `a1` (None when the reply is not one letter)
    and `first`, which is "right", "wrong" or None. `pairs`, for every reasoned pair of a
    question with a first reply, in both formats, keyed by item id: its `qid`, `source`,
    `fmt`, `target`, `writer`, and `status`, which is "kept", "dropped" (the target is the
    first answer) or "no_first_answer"; and `situation` for a kept pair, else None.
    """
    asked = questions()
    out_questions: dict[str, dict] = {}
    for qid, q in asked.items():
        if qid not in first_replies:
            continue
        a1 = first_answer(first_replies[qid])
        first = None if a1 is None else ("right" if a1 == q.gold else "wrong")
        out_questions[qid] = {"source": q.source, "gold": q.gold, "a1": a1, "first": first}
    out_pairs: dict[str, dict] = {}
    for p in pairs():
        if p.qid not in out_questions:
            continue
        info = out_questions[p.qid]
        a1 = info["a1"]
        if a1 is None:
            status, sit = "no_first_answer", None
        elif p.target == a1:
            status, sit = "dropped", None
        else:
            status, sit = "kept", situation("reasoned", a1, info["gold"], p.target)
        for fmt in FORMATS:
            out_pairs[episode_id(p.qid, "reasoned", fmt, p.target, p.writer)] = {
                "qid": p.qid, "source": info["source"], "fmt": fmt, "target": p.target,
                "writer": p.writer, "status": status, "situation": sit}
    return {"questions": out_questions, "pairs": out_pairs}
