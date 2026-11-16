"""Whether a reason's wording alone, without its question, gives away a gold-target reason."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

STYLE_FOLDS = 5


def style_score(reasons: list[str], is_gold_target: list[bool], groups: list[str]) -> float:
    """The cross-validated AUROC of a word-and-bigram TF-IDF logistic regression at telling
    gold-target reasons from the others, with folds grouped by question."""
    y = np.array([int(v) for v in is_gold_target])
    groups_arr = np.array(groups)
    scores = np.zeros(len(y))
    for train, held in GroupKFold(STYLE_FOLDS).split(reasons, y, groups_arr):
        clf = make_pipeline(TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2)),
                            LogisticRegression(max_iter=2000, C=1.0))
        clf.fit([reasons[i] for i in train], y[train])
        scores[held] = clf.decision_function([reasons[i] for i in held])
    return float(roc_auc_score(y, scores))
