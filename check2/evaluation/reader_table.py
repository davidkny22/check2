"""The reader's score table on a run's recorded `p`: its three one-vs-rest scores, its selection score and each baseline's.

    python -m check2.evaluation.reader_table --run <run folder>

The run's records carry `p`, computed live in the run. The one-vs-rest AUROC of each output
is taken over all of the run's episodes; the selection score is the AUROC of `user_right` on
reasoned episodes of wrong-first questions, averaged over the two formats. Each baseline is a
logistic regression fitted on all of `train` and scored on the run's episodes, and its
selection score is reported beside the reader's with the paired difference, reader minus
baseline. Every score carries its 95% interval from 2,000 resamples of question ids.

On a `test` run the table also gives the reason-only reader's score on the selection
contrast, with its interval, from its saved `p` on the `test` pairs
(`data/reader/p_reason_only_test.jsonl`); where that file does not exist, the record says so.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from check2.data import paths
from check2.evaluation.intervals import interval, paired_difference
from check2.evaluation.tables import load_run
from check2.reader.baselines import BASELINES, baseline_scores, fit_baseline
from check2.reader import files as reader_files
from check2.reader.model import OUTPUTS
from check2.reader.reason_only import FORMAT as REASON_ONLY_FORMAT
from check2.reader.reason_only import P_TEST as REASON_ONLY_P_TEST
from check2.reader.reason_only import pair_key
from check2.reader.scores import auroc, contrast_records, selection_score, user_right_scores
from check2.record import write_record

FIRST_TURN_FILE = paths.DATA / "reader" / "first_turn.jsonl"
EPISODE_FILES = {"test": "test.jsonl", "heldout": "heldout.jsonl", "train": "train.jsonl"}


def _read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _interval(result) -> dict:
    return {"estimate": result.estimate, "low": result.low, "high": result.high, "skipped": result.skipped}


def _by_qid(rows: Sequence[Any], qid_of) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for row in rows:
        out.setdefault(qid_of(row), []).append(row)
    return out


def one_vs_rest_scores(records: Sequence[Mapping[str, Any]]) -> dict:
    """Each output's AUROC against all other situations, with its interval."""
    by_qid = _by_qid(records, lambda r: r["qid"])
    out = {}
    for k, name in enumerate(OUTPUTS):
        def statistic(rows, k=k, name=name):
            return auroc([r["p"][k] for r in rows], [r["situation"] == name for r in rows])
        out[name] = _interval(interval(statistic, by_qid))
    return out


def _selection(by_qid: Mapping[str, Sequence[Any]]) -> dict:
    return _interval(interval(selection_score, by_qid))


def reason_only_score(run_episodes: Sequence[Mapping[str, Any]], saved: Path) -> dict:
    """The reason-only reader's selection score with its interval, on the pairs of the run's reasoned episodes.

    Its prompt holds the reason alone, so the two formats of a pair share one `p` and the
    score is taken over pairs. Every pair of the contrast must have a saved `p`.
    """
    shown = saved.relative_to(paths.ROOT).as_posix() if saved.is_relative_to(paths.ROOT) else str(saved)
    if not saved.exists():
        return {"file": shown, "exists": False,
                "note": f"{shown} does not exist, so the reason-only reader's score on this run's pairs is not given"}
    p = reader_files.read_p(saved)
    pairs = {}
    for e in run_episodes:
        if e["form"] == "reasoned":
            key = pair_key(e["qid"], e["target"])
            pairs[key] = {"episode_id": key, "qid": e["qid"], "form": "reasoned", "format": REASON_ONLY_FORMAT,
                          "a1": e["a1"], "gold": e["gold"], "situation": e["situation"]}
    contrast_pairs = [e for e in pairs.values() if e["a1"] != e["gold"]]
    missing = sorted(e["episode_id"] for e in contrast_pairs if e["episode_id"] not in p)
    if missing:
        raise ValueError(f"{shown} has no p for {len(missing)} pairs of the contrast, e.g. {missing[:3]}")
    contrast = contrast_records(list(pairs.values()), user_right_scores(p))

    def score(rows):
        return selection_score(rows, (REASON_ONLY_FORMAT,))
    return {"file": shown, "exists": True, "pairs": len(pairs), "contrast_questions": len(contrast),
            "selection": _interval(interval(score, contrast))}


def table(records: Sequence[Mapping[str, Any]], episodes: Mapping[str, Mapping[str, Any]],
          train: Sequence[Mapping[str, Any]], first_turn: Mapping[str, Sequence[float]],
          reason_only: Path | None = None) -> dict:
    """The table's values: the reader's scores, each baseline's selection score and the paired differences.

    With `reason_only`, the file of the reason-only reader's saved `p`, its score stands beside the reader's.
    """
    run_episodes = [episodes[r["episode_id"]] for r in records]
    reader_contrast = contrast_records(run_episodes, user_right_scores({r["episode_id"]: r["p"] for r in records}))
    result = {"episodes": len(records), "contrast_questions": len(reader_contrast),
              "reader": {"one_vs_rest": one_vs_rest_scores(records), "selection": _selection(reader_contrast)},
              "baselines": {}}
    if reason_only is not None:
        result["reason_only"] = reason_only_score(run_episodes, reason_only)
    for name in BASELINES:
        model = fit_baseline(name, train, first_turn)
        contrast = contrast_records(run_episodes, baseline_scores(name, model, run_episodes, first_turn))
        result["baselines"][name] = {
            "fitted_on_train_episodes": sum(1 for e in train if name != "target_probability" or e["target"] is not None),
            "selection": _selection(contrast),
            "reader_minus_baseline": _interval(paired_difference(selection_score, reader_contrast, contrast)),
        }
    return result


def _shown(d: Mapping[str, Any]) -> str:
    def f(x):
        return "undefined" if x is None else f"{x:.4f}"
    return f"{f(d['estimate'])} [{f(d['low'])}, {f(d['high'])}]"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m check2.evaluation.reader_table", description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, type=Path, help="a run folder whose records carry live p")
    args = parser.parse_args(argv)
    meta, records = load_run(args.run)
    split = meta["split"]
    episodes = {e["episode_id"]: e for e in _read_jsonl(paths.EPISODES / EPISODE_FILES[split])}
    train = _read_jsonl(paths.EPISODES / EPISODE_FILES["train"])
    first_turn = {r["qid"]: r["probabilities"] for r in _read_jsonl(FIRST_TURN_FILE)}
    values = table(records, episodes, train, first_turn, REASON_ONLY_P_TEST if split == "test" else None)
    path = write_record("measurement", "reader_table", values, config={"run": meta["run_id"]},
                        extra={"run": {k: meta.get(k) for k in ("run_id", "commit", "order_sha256", "split", "gpu")}})
    reader = values["reader"]
    print("reader one-vs-rest: " + ", ".join(f"{k} {_shown(v)}" for k, v in reader["one_vs_rest"].items()))
    print(f"reader selection score {_shown(reader['selection'])}")
    if "reason_only" in values:
        found = values["reason_only"]
        print(f"reason-only reader selection score {_shown(found['selection'])}" if found["exists"] else found["note"])
    for name, b in values["baselines"].items():
        print(f"{name}: selection {_shown(b['selection'])}; reader minus baseline {_shown(b['reader_minus_baseline'])}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
