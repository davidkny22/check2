"""Assemble the study's data from the pinned measurement files and print its count table.

    python -m check2.data.build [--tail-position K] [--from-saved-splits] [--require-complete]

Without `--from-saved-splits` the splits are drawn and saved; if a saved file already
exists the draw must reproduce it exactly. With it, the saved splits are read and nothing
is drawn. Without `--require-complete`, pairs whose reason is not written yet are counted
and left without reasoned episodes; with it, any such pair is an error.
"""
import argparse
import json
import sys
import time
from dataclasses import asdict

from check2.data import counts, paths
from check2.data.episodes import build_episodes, write_episodes
from check2.data.jobs import apply_reviews, crosswriter_jobs, review_jobs
from check2.data.pairs import all_pairs, load_pairs, reason_sets, write_pairs
from check2.data.pool import complying, load_pool
from check2.data.splits import build_splits, load_splits, save_splits
from check2.data.style import style_score


class CountError(RuntimeError):
    """A count of the pinned files differs from the specification's."""


def _print_block(title: str, rows: dict) -> None:
    print(title)
    for key, value in rows.items():
        print(f"  {key:<34} {value}")


def _splits(pool, pairs, from_saved: bool):
    if from_saved:
        return load_splits(), "read from data/splits.json"
    drawn = build_splits(pool, pairs)
    if paths.SPLITS.exists():
        saved = load_splits()
        if asdict(saved) != asdict(drawn):
            raise RuntimeError("the draw does not reproduce the saved splits; the splits are drawn once")
        return saved, "drawn; equal to the saved data/splits.json"
    save_splits(drawn)
    return drawn, "drawn and saved to data/splits.json"


def _style_scores(pool, pairs) -> dict:
    by_id = complying(pool)
    scores = {}
    for name, members in reason_sets(pairs).items():
        gold = [p.target == by_id[p.qid].gold for p in members]
        scores[name] = {"reasons": len(members), "gold_target": sum(gold),
                        "auroc": round(style_score([p.reason for p in members], gold,
                                                   [p.qid for p in members]), 4)}
    paths.STYLE_SCORES.parent.mkdir(parents=True, exist_ok=True)
    paths.STYLE_SCORES.write_text(json.dumps(scores, indent=1) + "\n", encoding="utf-8", newline="\n")
    return scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tail-position", type=int, default=0)
    parser.add_argument("--from-saved-splits", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    t0 = time.time()

    pool = load_pool()
    every_pair = all_pairs(pool)
    plan = json.loads(paths.PLAN.read_text(encoding="utf-8"))
    second = json.loads(paths.TEST_SECOND_PART.read_text(encoding="utf-8"))
    known = {**counts.pool_counts(pool),
             **counts.pair_counts(pool, every_pair, plan["test_questions"],
                                  second["wrong_first"] + second["right_first"])}
    _print_block("counts of the files as given (specification's number in brackets)",
                 {k: f"{v} [{counts.EXPECTED[k]}]" for k, v in known.items()})
    wrong = counts.mismatches(known)
    if wrong:
        raise CountError("; ".join(wrong))

    review_counts = apply_reviews() if any(paths.REVIEWS.glob("review_*.json")) else None
    pairs = load_pairs(pool)
    splits, how = _splits(pool, every_pair, args.from_saved_splits)
    missing = [p for p in pairs if p.reason is None]
    if args.require_complete and missing:
        raise RuntimeError(f"{len(missing)} pairs have no reason")
    write_pairs(pairs, pool)
    _print_block(f"splits ({how})", counts.split_counts(splits))
    removed = len(every_pair) - len(pairs)
    test = set(splits.test)
    removed_test = sum(p.qid in test and p.writer == "opus" for p in every_pair) - \
        sum(p.qid in test and p.writer == "opus" for p in pairs)
    _print_block("pairs (data/pairs.jsonl)", {
        "pairs": len(pairs), "by the writer": sum(p.writer == "opus" for p in pairs),
        "cross-writer": sum(p.writer == "sonnet" for p in pairs),
        "removed by the review": removed, "without a reason yet": len(missing),
        "without a reason yet, in test": sum(p.qid in test for p in missing),
        "without a reason, by batch": dict(sorted({j: sum(p.job_id[:3] == j for p in missing)
                                                   for j in {p.job_id[:3] for p in missing}}.items())),
    })

    episodes = build_episodes(pool, splits, pairs, args.tail_position)
    files = write_episodes(episodes, args.tail_position)
    _print_block("episode files (data/episodes/)", files)
    table = counts.episode_table(episodes)
    _print_block("episodes by split, form and format", dict(sorted(table["split_form_format"].items())))
    _print_block("episodes by split and situation", dict(sorted(table["split_situation"].items())))
    missing_test = sum(p.qid in test and p.writer == "opus" for p in missing)
    expected_test = counts.expected_test_episodes(removed_test, missing_test)
    if files["test.jsonl"] != expected_test:
        raise CountError(f"test episodes: {files['test.jsonl']}, expected {expected_test}")

    jobs = {"crosswriter.json": len(crosswriter_jobs(every_pair))}
    try:
        jobs.update(review_jobs(pool, every_pair, splits))
    except ValueError as err:
        jobs["review jobs"] = f"not made: {err}"
    _print_block("job files (data/jobs/)", jobs)
    if review_counts is not None:
        _print_block("soundness reviews applied", review_counts)
    scores = _style_scores(pool, every_pair)
    _print_block("style scores (data/reviews/style_scores.json)",
                 {k: f"auroc {v['auroc']} on {v['reasons']} reasons, {v['gold_target']} gold-target"
                  for k, v in scores.items()})
    print(f"episodes in all: {len(episodes)}; seconds: {time.time() - t0:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
