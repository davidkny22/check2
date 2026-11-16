"""The five folds of `train` as the reader's cross-fitting uses them: for each fold, the episodes
a fit trains on (those of the other four folds) and the episodes it scores (its own)."""
from collections.abc import Mapping, Sequence


def fold_of(folds: Sequence[Sequence[str]]) -> dict[str, int]:
    """The fold of each question id."""
    out: dict[str, int] = {}
    for k, fold in enumerate(folds):
        for qid in fold:
            if qid in out:
                raise ValueError(f"{qid} is in folds {out[qid]} and {k}")
            out[qid] = k
    return out


def fold_parts(episodes: Sequence[Mapping], folds: Sequence[Sequence[str]],
               k: int) -> tuple[list[str], list[str]]:
    """The sorted ids of the episodes fold `k`'s fit trains on, and of those it scores."""
    where = fold_of(folds)
    missing = sorted({e["qid"] for e in episodes} - set(where))
    if missing:
        raise ValueError(f"episodes of questions in no fold: {missing[:5]}")
    trained = sorted(e["episode_id"] for e in episodes if where[e["qid"]] != k)
    scored = sorted(e["episode_id"] for e in episodes if where[e["qid"]] == k)
    return trained, scored
