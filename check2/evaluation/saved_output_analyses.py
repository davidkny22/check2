"""Three analyses of outputs the runs already saved. Nothing is generated and no model is loaded.

    python -m check2.evaluation.saved_output_analyses

1. **Plausible against weak.** The wrong-target reasoned `test` episodes are grouped by the
   soundness review's rating of their reason: plausible, weak, and nonsense, which is
   reported apart. For each group: the Original run's wrong adoption per format, as count,
   total and rate; and the reader's selection score with that group as the negatives, the
   AUROC of `p[user_right]` between the gold-target reasoned episodes of wrong-first
   questions and the group's wrong-target reasoned episodes of wrong-first questions,
   averaged over the two formats. Then the paired difference, plausible minus weak.
2. **The second writer.** The reader's selection score on the cross-writer run's episodes,
   from the `p` that run recorded, beside the score of the same pairs written by the first
   writer, from the Original run's `p`, and their paired difference. A cross-writer pair
   with no first-writer episode in the Original run is listed, with whether the soundness
   review removed the first writer's reason.
3. **Reader against reason-only reader.** On the reasoned `test` episodes of wrong-first
   questions, the paired difference in selection score, reader minus reason-only reader,
   from the Original run's `p` and the reason-only reader's saved `p` on the `test` pairs.

Every interval is from 2,000 resamples of question ids. A paired difference draws the same
question ids for both of its sides; where a question has episodes on one side only, it
brings none to the other.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from check2.data import paths
from check2.data.episodes import FORMATS
from check2.evaluation.intervals import interval, paired_difference
from check2.evaluation.rates import rate
from check2.evaluation.run_folders import load_run, require_one_card
from check2.reader import files as reader_files
from check2.reader.reason_only import FORMAT as REASON_ONLY_FORMAT
from check2.reader.reason_only import P_TEST as REASON_ONLY_P_TEST
from check2.reader.reason_only import pair_key
from check2.reader.scores import Scored, contrast_records, selection_score, user_right_scores
from check2.record import write_record

NAME = "saved_output_analyses"
ORIGINAL = paths.ROOT / "results" / "original"
CROSSWRITER = paths.ROOT / "results" / "original-crosswriter"
REVIEW_FILES = "review_test_*.json"
RATINGS = ("plausible", "weak", "nonsense")
COMPARED = ("plausible", "weak")


def load_ratings(folder: Path = paths.REVIEWS) -> dict[str, str]:
    """The review's rating of each `test` reason, by pair key `qid:target`; a gold-target reason's rating is empty."""
    ratings: dict[str, str] = {}
    for path in sorted(folder.glob(REVIEW_FILES)):
        for review in json.loads(path.read_text(encoding="utf-8"))["reviews"]:
            if review["job_id"] in ratings:
                raise ValueError(f"{path.name}: reason {review['job_id']} is reviewed twice")
            ratings[review["job_id"]] = review["plausible"]
    return ratings


def load_removed(path: Path = paths.REMOVED_PAIRS) -> set[str]:
    """The pair keys whose reason the soundness review removed before any run."""
    return {pair_key(r["qid"], r["target"]) for r in json.loads(path.read_text(encoding="utf-8"))["removed"]}


def _shown_interval(result) -> dict:
    return {"estimate": result.estimate, "low": result.low, "high": result.high, "skipped": result.skipped}


def _by_qid(rows: Sequence[Any], qid_of) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for row in rows:
        out.setdefault(qid_of(row), []).append(row)
    return out


def _over_questions(a: Mapping[str, list], b: Mapping[str, list]) -> tuple[dict[str, list], dict[str, list]]:
    """Both sides keyed by every question either holds, a question absent from one side bringing no records to it."""
    qids = sorted(set(a) | set(b))
    return {q: list(a.get(q, [])) for q in qids}, {q: list(b.get(q, [])) for q in qids}


def _adoption(rows: Sequence[Mapping[str, Any]]) -> float | None:
    if not rows:
        return None
    return sum(r["a2"] == r["target"] for r in rows) / len(rows)


def wrong_adoption(episodes: Sequence[Mapping[str, Any]]) -> dict:
    """Per format: the episodes whose final answer is the target, their total, the rate and its interval."""
    out = {}
    for fmt in FORMATS:
        rows = [e for e in episodes if e["format"] == fmt]
        counted = rate(sum(e["a2"] == e["target"] for e in rows), len(rows))
        out[fmt] = {**counted, "interval": _shown_interval(interval(_adoption, _by_qid(rows, lambda e: e["qid"])))}
    return out


def _with_rating(episodes: Sequence[Mapping[str, Any]], ratings: Mapping[str, str], rating: str) -> list:
    """The wrong-target episodes whose reason the review rated `rating`."""
    return [e for e in episodes
            if e["target"] != e["gold"] and ratings.get(pair_key(e["qid"], e["target"])) == rating]


def _selection(by_qid: Mapping[str, Sequence[Scored]]) -> dict:
    return _shown_interval(interval(selection_score, by_qid))


def plausible_against_weak(records: Sequence[Mapping[str, Any]], ratings: Mapping[str, str]) -> dict:
    """The first analysis on the Original run's records: adoption and selection score per rating, and their difference."""
    reasoned = [r for r in records if r["form"] == "reasoned"]
    wrong_target = [r for r in reasoned if r["target"] != r["gold"]]
    unrated = sorted({pair_key(r["qid"], r["target"]) for r in wrong_target
                      if ratings.get(pair_key(r["qid"], r["target"])) not in RATINGS})
    positives = [r for r in reasoned if r["situation"] == "user_right" and r["a1"] != r["gold"]]
    p = user_right_scores({r["episode_id"]: r["p"] for r in reasoned})
    groups, contrasts = {}, {}
    for rating in RATINGS:
        rated = _with_rating(wrong_target, ratings, rating)
        negatives = [r for r in rated if r["a1"] != r["gold"]]
        contrasts[rating] = contrast_records(positives + negatives, p)
        groups[rating] = {
            "pairs": len({pair_key(r["qid"], r["target"]) for r in rated}),
            "episodes": len(rated),
            "questions": len({r["qid"] for r in rated}),
            "wrong_adoption": wrong_adoption(rated),
            "selection": {"negatives": len(negatives), "positives": len(positives),
                          "contrast_questions": len(contrasts[rating]), **_selection(contrasts[rating])},
        }
    first, second = COMPARED
    a, b = _over_questions(contrasts[first], contrasts[second])
    return {
        "wrong_target_pairs": len({pair_key(r["qid"], r["target"]) for r in wrong_target}),
        "wrong_target_episodes": len(wrong_target),
        "unrated_pairs": {"count": len(unrated), "pairs": unrated},
        "groups": groups,
        f"selection_{first}_minus_{second}": {"questions": len(a),
                                              **_shown_interval(paired_difference(selection_score, a, b))},
    }


def _pair_format(r: Mapping[str, Any]) -> tuple[str, str, str]:
    return r["qid"], r["target"], r["format"]


def second_writer(crosswriter: Sequence[Mapping[str, Any]], original: Sequence[Mapping[str, Any]],
                  removed: set[str]) -> dict:
    """The second analysis: the cross-writer score, the first writer's score on the same pairs, and their difference."""
    first_writer = {_pair_format(r): r for r in original if r["form"] == "reasoned"}
    matched = [r for r in crosswriter if _pair_format(r) in first_writer]
    for r in matched:
        other = first_writer[_pair_format(r)]
        if (r["a1"], r["gold"], r["situation"]) != (other["a1"], other["gold"], other["situation"]):
            raise ValueError(f"{r['episode_id']} and {other['episode_id']} share question, target and format "
                             "but not first answer, key or situation")
    unmatched = sorted({pair_key(r["qid"], r["target"]) for r in crosswriter if _pair_format(r) not in first_writer})
    first_same = [first_writer[_pair_format(r)] for r in matched]
    every = contrast_records(crosswriter, user_right_scores({r["episode_id"]: r["p"] for r in crosswriter}))
    second = contrast_records(matched, user_right_scores({r["episode_id"]: r["p"] for r in matched}))
    first = contrast_records(first_same, user_right_scores({r["episode_id"]: r["p"] for r in first_same}))
    return {
        "crosswriter": {"pairs": len({pair_key(r["qid"], r["target"]) for r in crosswriter}),
                        "episodes": len(crosswriter), "contrast_questions": len(every), **_selection(every)},
        "unmatched_pairs": [{"pair": k, "removed_by_review": k in removed} for k in unmatched],
        "same_pairs": {
            "pairs": len({pair_key(r["qid"], r["target"]) for r in matched}),
            "episodes": len(matched),
            "contrast_questions": len(second),
            "crosswriter": _selection(second),
            "first_writer": _selection(first),
            "crosswriter_minus_first_writer": _shown_interval(paired_difference(selection_score, second, first)),
        },
    }


def _either_selection(rows: Sequence[Scored]) -> float | None:
    """The selection score over the reader's two formats, or over the reason-only reader's one."""
    if any(r.format == REASON_ONLY_FORMAT for r in rows):
        return selection_score(rows, (REASON_ONLY_FORMAT,))
    return selection_score(rows, FORMATS)


def reader_against_reason_only(records: Sequence[Mapping[str, Any]], reason_only_p: Mapping[str, Sequence[float]]) -> dict:
    """The third analysis: reader minus reason-only reader on the reasoned episodes of wrong-first questions.

    The reason-only reader's prompt holds the reason alone, so the two formats of a pair share
    one `p` and its score is taken over pairs, in its own one format.
    """
    reasoned = [r for r in records if r["form"] == "reasoned"]
    reader = contrast_records(reasoned, user_right_scores({r["episode_id"]: r["p"] for r in reasoned}))
    pairs = {}
    for r in reasoned:
        key = pair_key(r["qid"], r["target"])
        pairs[key] = {"episode_id": key, "qid": r["qid"], "form": "reasoned", "format": REASON_ONLY_FORMAT,
                      "a1": r["a1"], "gold": r["gold"], "situation": r["situation"]}
    in_contrast = [e for e in pairs.values() if e["a1"] != e["gold"]]
    missing = sorted(e["episode_id"] for e in in_contrast if e["episode_id"] not in reason_only_p)
    if missing:
        raise ValueError(f"the reason-only reader has no p for {len(missing)} pairs of the contrast, e.g. {missing[:3]}")
    reason_only = contrast_records(in_contrast, user_right_scores({e["episode_id"]: reason_only_p[e["episode_id"]]
                                                                   for e in in_contrast}))
    return {
        "contrast_questions": len(reader),
        "contrast_pairs": len(in_contrast),
        "reader": _shown_interval(interval(_either_selection, reader)),
        "reason_only": _shown_interval(interval(_either_selection, reason_only)),
        "reader_minus_reason_only": _shown_interval(paired_difference(_either_selection, reader, reason_only)),
    }


def analyses(original: Sequence[Mapping[str, Any]], crosswriter: Sequence[Mapping[str, Any]],
             ratings: Mapping[str, str], removed: set[str], reason_only_p: Mapping[str, Sequence[float]]) -> dict:
    """The three analyses' values, as the record holds them."""
    return {"plausible_against_weak": plausible_against_weak(original, ratings),
            "second_writer": second_writer(crosswriter, original, removed),
            "reader_against_reason_only": reader_against_reason_only(original, reason_only_p)}


def _shown(d: Mapping[str, Any]) -> str:
    def f(x):
        return "undefined" if x is None else f"{x:.4f}"
    return f"{f(d['estimate'])} [{f(d['low'])}, {f(d['high'])}]"


def _relative(path: Path) -> str:
    return path.relative_to(paths.ROOT).as_posix() if path.is_relative_to(paths.ROOT) else str(path)


def main(argv: Sequence[str] | None = None) -> int:
    argparse.ArgumentParser(prog=f"python -m check2.evaluation.{NAME}", description=__doc__.splitlines()[0]).parse_args(argv)
    meta_original, original = load_run(ORIGINAL)
    meta_crosswriter, crosswriter = load_run(CROSSWRITER)
    require_one_card([meta_original, meta_crosswriter])
    values = analyses(original, crosswriter, load_ratings(), load_removed(), reader_files.read_p(REASON_ONLY_P_TEST))
    sources = {"original": _relative(ORIGINAL), "crosswriter": _relative(CROSSWRITER),
               "reviews": f"{_relative(paths.REVIEWS)}/{REVIEW_FILES}", "removed": _relative(paths.REMOVED_PAIRS),
               "reason_only_p": _relative(REASON_ONLY_P_TEST)}
    runs = {meta["run_id"]: {k: meta.get(k) for k in ("commit", "order_sha256", "split", "gpu", "episodes_written")}
            for meta in (meta_original, meta_crosswriter)}
    path = write_record("measurement", NAME, values, config=sources, extra={"runs": runs})
    first = values["plausible_against_weak"]
    print(f"wrong-target pairs {first['wrong_target_pairs']}; unrated {first['unrated_pairs']['count']}")
    for rating, group in first["groups"].items():
        adoption = "; ".join(f"{fmt} {a['numerator']}/{a['denominator']} {_shown(a['interval'])}"
                             for fmt, a in group["wrong_adoption"].items())
        print(f"{rating}: {group['pairs']} pairs; wrong adoption {adoption}; "
              f"selection {_shown(group['selection'])} over {group['selection']['contrast_questions']} questions")
    print(f"selection, plausible minus weak {_shown(first['selection_plausible_minus_weak'])}")
    second = values["second_writer"]
    print(f"cross-writer selection {_shown(second['crosswriter'])} over {second['crosswriter']['contrast_questions']} "
          f"questions; unmatched pairs {[u['pair'] for u in second['unmatched_pairs']]}")
    same = second["same_pairs"]
    print(f"same {same['pairs']} pairs over {same['contrast_questions']} questions: cross-writer "
          f"{_shown(same['crosswriter'])}, first writer {_shown(same['first_writer'])}, "
          f"difference {_shown(same['crosswriter_minus_first_writer'])}")
    third = values["reader_against_reason_only"]
    print(f"reader {_shown(third['reader'])}; reason-only reader {_shown(third['reason_only'])}; "
          f"reader minus reason-only {_shown(third['reader_minus_reason_only'])}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
