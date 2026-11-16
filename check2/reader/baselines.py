"""The three baselines the reader is compared with, each a logistic regression on the situation.

- `first_answer_confidence`: the first answer's probability at the first turn and its margin
  over the runner-up option.
- `target_probability`: the probability the model gave the pushback's target at the first
  turn; defined only where the pushback names a target.
- `pushback_tfidf`: a TF-IDF classifier on the pushback turn's text alone.

Each fit takes the same training episodes as the reader's fit (those where its feature
exists) and the same label, and scores an episode by its probability of `user_right`. The
settings are those of the design measurement in `measurements/anchor/anchor_check.py`.
"""
from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from check2.data.paths import LETTERS
from check2.reader.folds import fold_parts
from check2.reader.window import last_user_content

BASELINES = ("first_answer_confidence", "target_probability", "pushback_tfidf")
FirstTurn = Mapping[str, Sequence[float]]


def defined(name: str, episode: Mapping) -> bool:
    """Whether the baseline's feature exists for the episode."""
    return name != "target_probability" or episode["target"] is not None


def features(name: str, episodes: Sequence[Mapping], first_turn: FirstTurn):
    """The baseline's inputs for the episodes: a numeric array, or the list of pushback texts."""
    if name == "pushback_tfidf":
        return [last_user_content(e["prompt"]) for e in episodes]
    rows = []
    for e in episodes:
        p = first_turn[e["qid"]]
        if name == "first_answer_confidence":
            a1 = LETTERS.index(e["a1"])
            runner_up = max(v for k, v in enumerate(p) if k != a1)
            rows.append([p[a1], p[a1] - runner_up])
        elif name == "target_probability":
            rows.append([p[LETTERS.index(e["target"])]])
        else:
            raise ValueError(f"unknown baseline {name!r}")
    return np.asarray(rows, dtype=np.float64)


def _model(name: str):
    if name == "pushback_tfidf":
        return make_pipeline(TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2)),
                             LogisticRegression(max_iter=2000, C=1.0))
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def fit_baseline(name: str, episodes: Sequence[Mapping], first_turn: FirstTurn):
    """The baseline fitted on the episodes where its feature exists, against their situations."""
    rows = [e for e in episodes if defined(name, e)]
    model = _model(name)
    model.fit(features(name, rows, first_turn), [e["situation"] for e in rows])
    return model


def baseline_scores(name: str, model, episodes: Sequence[Mapping], first_turn: FirstTurn) -> dict[str, float]:
    """Each episode's probability of `user_right` under a fitted baseline, where its feature exists."""
    rows = [e for e in episodes if defined(name, e)]
    if not rows:
        return {}
    column = list(model.classes_).index("user_right")
    probabilities = model.predict_proba(features(name, rows, first_turn))[:, column]
    return {e["episode_id"]: float(v) for e, v in zip(rows, probabilities)}


def cross_fitted_scores(name: str, episodes: Sequence[Mapping], folds: Sequence[Sequence[str]],
                        first_turn: FirstTurn) -> dict[str, float]:
    """Each `train` episode's score from the baseline fitted on the other four folds."""
    by_id = {e["episode_id"]: e for e in episodes}
    out: dict[str, float] = {}
    for k in range(len(folds)):
        trained, scored = fold_parts(episodes, folds, k)
        model = fit_baseline(name, [by_id[e] for e in trained], first_turn)
        out.update(baseline_scores(name, model, [by_id[e] for e in scored], first_turn))
    return out
