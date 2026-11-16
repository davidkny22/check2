"""Reader scores: the one-vs-rest AUROC of each output, and the selection score.

The selection score is the AUROC of the `user_right` output on reasoned episodes of
wrong-first questions, where the situations are `user_right` and `recheck`, computed in each
format and averaged over the two formats.
"""
from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np

from check2.data.episodes import FORMATS
from check2.reader.model import OUTPUTS

USER_RIGHT = OUTPUTS.index("user_right")


class Scored(NamedTuple):
    """One episode of the selection contrast: its format, whether it is `user_right`, its score."""
    episode_id: str
    qid: str
    format: str
    positive: bool
    score: float


def auroc(scores: Sequence[float], positives: Sequence[bool]) -> float | None:
    """The probability that a positive outscores a negative, ties counting one half; None
    when either class is absent."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(positives, dtype=bool)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    average_rank = np.cumsum(counts) - (counts - 1) / 2.0
    ranks = average_rank[inverse]
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def one_vs_rest(p: Mapping[str, Sequence[float]], situation_of: Mapping[str, str]) -> dict[str, float | None]:
    """For each output, the AUROC of its probability against all other situations."""
    ids = sorted(p)
    return {name: auroc([p[e][k] for e in ids], [situation_of[e] == name for e in ids])
            for k, name in enumerate(OUTPUTS)}


def in_contrast(episode: Mapping) -> bool:
    """A reasoned episode of a wrong-first question."""
    return episode["form"] == "reasoned" and episode["a1"] != episode["gold"]


def contrast_records(episodes: Sequence[Mapping], score_of: Mapping[str, float]) -> dict[str, list[Scored]]:
    """The selection contrast's episodes, each with its score, grouped by question id.

    Every episode of the contrast must have a score."""
    by_qid: dict[str, list[Scored]] = {}
    for e in sorted((e for e in episodes if in_contrast(e)), key=lambda e: e["episode_id"]):
        if e["situation"] not in ("user_right", "recheck"):
            raise ValueError(f"{e['episode_id']} is in the contrast with situation {e['situation']}")
        by_qid.setdefault(e["qid"], []).append(Scored(e["episode_id"], e["qid"], e["format"],
                                                      e["situation"] == "user_right",
                                                      float(score_of[e["episode_id"]])))
    return by_qid


def selection_score(records: Sequence[Scored], formats: Sequence[str] = FORMATS) -> float | None:
    """The mean over the formats of each format's AUROC; None if any is undefined."""
    per_format = []
    for fmt in formats:
        rows = [r for r in records if r.format == fmt]
        value = auroc([r.score for r in rows], [r.positive for r in rows])
        if value is None:
            return None
        per_format.append(value)
    return float(np.mean(per_format))


def flatten(by_qid: Mapping[str, Sequence[Scored]]) -> list[Scored]:
    return [r for qid in sorted(by_qid) for r in by_qid[qid]]


def user_right_scores(p: Mapping[str, Sequence[float]]) -> dict[str, float]:
    return {e: float(v[USER_RIGHT]) for e, v in p.items()}
