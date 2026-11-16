"""The question pool with the model's own first answers, and its duplicate groups."""
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from check2.data.paths import LETTERS, POOL


@dataclass(frozen=True)
class Question:
    qid: str
    subject: str
    question: str
    choices: tuple[str, str, str, str]
    gold: str
    a1: str
    free_text: str
    complies: bool
    p1: tuple[float, float, float, float]

    @property
    def wrong_first(self) -> bool:
        return self.a1 != self.gold


def normalise(text: str) -> str:
    """Lower-case a question's text and remove all whitespace."""
    return "".join(text.lower().split())


def load_pool(path: Path = POOL) -> list[Question]:
    """Every question of the pool file, in the file's order, with letters for options.

    Questions whose first answer did not comply are kept here so that their count can be
    checked; everything built from the pool uses only questions with `complies` true.
    """
    records = json.loads(path.read_text(encoding="utf-8"))["questions"]
    pool = []
    for r in records:
        if len(r["choices"]) != 4:
            raise ValueError(f"{r['qid']} has {len(r['choices'])} options, not 4")
        pool.append(Question(
            qid=r["qid"], subject=r["subject"], question=r["question"],
            choices=tuple(r["choices"]), gold=LETTERS[r["gold"]], a1=LETTERS[r["a1"]],
            free_text=r["free_text"], complies=bool(r["complies"]), p1=tuple(r["p1"]),
        ))
    ids = [q.qid for q in pool]
    if len(set(ids)) != len(ids):
        raise ValueError("the pool holds a question id twice")
    return pool


def complying(pool: list[Question]) -> dict[str, Question]:
    """The questions with a complying first answer, by id."""
    return {q.qid: q for q in pool if q.complies}


def duplicate_groups(pool: list[Question]) -> list[list[str]]:
    """Complying ids that share a normalised question text, each group sorted, groups sorted."""
    by_text = defaultdict(list)
    for q in pool:
        if q.complies:
            by_text[normalise(q.question)].append(q.qid)
    return sorted(sorted(ids) for ids in by_text.values() if len(ids) > 1)
