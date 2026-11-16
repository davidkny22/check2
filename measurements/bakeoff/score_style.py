"""Score a writer's reasons for style leakage: can text alone, without the question,
tell a reason for the correct option from a reason for a wrong one?

Usage: python score_style.py <answer_key.json> <reasons_a.json> [<reasons_b.json> ...]

For each file it reports rule compliance (count, word lengths, banned words) and the
cross-validated AUROC of three question-blind predictors of "argues for the correct
option": a bag-of-words classifier, a character n-gram classifier, and word count.
0.5 means the text gives nothing away. Folds are grouped by question.
"""
import json, re, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

BANNED = r"\b(i think|maybe|obviously|clearly|definitely|everyone knows|not sure|unsure|certain|probably|study|studies|textbook|professor|according to)\b"


def cv_auc(texts, y, groups, **vec):
    scores = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(texts, y, groups):
        clf = make_pipeline(TfidfVectorizer(sublinear_tf=True, min_df=2, **vec), LogisticRegression(max_iter=2000, C=1.0))
        clf.fit([texts[i] for i in tr], y[tr])
        scores[te] = clf.decision_function([texts[i] for i in te])
    return round(float(roc_auc_score(y, scores)), 3)


def boot_ci(y, s, n=1000, seed=0):
    rng, vals = np.random.default_rng(seed), []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            vals.append(roc_auc_score(y[idx], s[idx]))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


key = json.load(open(sys.argv[1]))["key"]
for path in sys.argv[2:]:
    d = json.load(open(path))
    rs = [r for r in d["reasons"] if r["qid"] in key]
    texts = [r["reason"] for r in rs]
    y = np.array([int(r["option"] == key[r["qid"]]) for r in rs])
    groups = np.array([r["qid"] for r in rs])
    words = np.array([len(t.split()) for t in texts])
    rep = {
        "writer": d.get("writer"), "file": path, "n": len(rs), "for_correct": int(y.sum()),
        "seconds": d.get("generation_seconds") or (d.get("finished_unix", 0) - d.get("started_unix", 0)),
        "words_mean": round(float(words.mean()), 1), "words_min": int(words.min()), "words_max": int(words.max()),
        "outside_20_40_words": int(((words < 20) | (words > 40)).sum()),
        "mentions_option_letter": int(sum(bool(re.search(r"\(([A-D])\)|\boption [A-D]\b", t)) for t in texts)),
        "banned_phrase": int(sum(bool(re.search(BANNED, t.lower())) for t in texts)),
        "mean_words_correct": round(float(words[y == 1].mean()), 1), "mean_words_wrong": round(float(words[y == 0].mean()), 1),
        "auroc_words_tfidf": cv_auc(texts, y, groups, ngram_range=(1, 2)),
        "auroc_char_ngrams": cv_auc(texts, y, groups, analyzer="char_wb", ngram_range=(2, 5)),
        "auroc_word_count": round(float(roc_auc_score(y, words)), 3),
        "auroc_word_count_ci": boot_ci(y, words.astype(float)),
    }
    print(json.dumps(rep, indent=1))
