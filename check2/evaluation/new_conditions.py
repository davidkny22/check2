"""The frozen reader and its gate on paper on reworded pushback and on the expanded pairs.

    python -m check2.evaluation.new_conditions [--runs <results folder>]

A measurement added after the `test` results had been read. It reads the Original runs of
the three reworded base sentences and of the expanded pairs, and the Original run over
`test` as run, each made with the reader's `p` computed live. Per set, on reasoned episodes:

- the reader's selection score with its interval and the number of questions in the contrast;
- the reason-only reader's selection score on the same pairs. Its prompt holds the reason
  alone, so the reworded sets share the saved `p` of the `test` pairs; the expanded pairs
  have their own;
- per format, wrong adoption and right adoption of the original model, and of the reader's
  row of Gates on paper at its two settings. The thresholds are those Gates on paper fits on
  `train` with the cross-fitted `p`; nothing is fitted on any of these sets. An episode that
  does not pass has its final answer replaced by its first answer. The same rates are also
  given over wrong-first questions only;
- the run's check on the appended read.

Every interval is from 2,000 resamples of question ids. Nothing is generated.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from check2.data import paths
from check2.data.condition_episodes import read_jsonl
from check2.evaluation import gates_on_paper
from check2.evaluation.intervals import interval
from check2.evaluation.reader_table import reason_only_score
from check2.evaluation.run_folders import load_run
from check2.reader.reason_only import P_TEST as REASON_ONLY_P_TEST
from check2.reader.scores import contrast_records, selection_score, user_right_scores
from check2.record import write_record

READER_SCORE = "reader_user_right"
SETS = {"test as run": "original", "believe": "original-believe", "disagree": "original-disagree",
        "hmm": "original-hmm", "expanded": "original-expanded"}
REASON_ONLY_EXPANDED = paths.DATA / "expansion" / "p_reason_only.jsonl"
LABEL = "added after the test results had been read"
EXPANDED_NOTE = "the expanded set's first answers were generated on a 40 GB A100; every run here was made on the card its record names"
FORMAT_ORDER = ("snap", "short")


def _shown(result) -> dict:
    return {"estimate": result.estimate, "low": result.low, "high": result.high, "skipped": result.skipped}


def _by_qid(records: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    out: dict[str, list[Mapping[str, Any]]] = {}
    for r in records:
        out.setdefault(r["qid"], []).append(r)
    return out


def adoption(records: Sequence[Mapping[str, Any]], wrong_target: bool) -> Callable[[Sequence], float | None]:
    """The statistic: the share of episodes ending on the target, over wrong-target or gold-target episodes."""
    def statistic(rows):
        chosen = [r for r in rows if (r["target"] != r["gold"]) == wrong_target]
        return sum(r["a2"] == r["target"] for r in chosen) / len(chosen) if chosen else None
    return statistic


def adoption_rates(records: Sequence[Mapping[str, Any]]) -> dict:
    """Per format, wrong and right adoption with their intervals and counts, over all questions and wrong-first ones."""
    out = {}
    for fmt in FORMAT_ORDER:
        rows = [r for r in records if r["format"] == fmt]
        cell = {}
        for scope, kept in (("all", rows), ("wrong_first", [r for r in rows if r["a1"] != r["gold"]])):
            by_qid = _by_qid(kept)
            cell[scope] = {
                name: {**_shown(interval(adoption(kept, wrong), by_qid)),
                       "episodes": sum((r["target"] != r["gold"]) == wrong for r in kept)}
                for name, wrong in (("wrong_adoption", True), ("right_adoption", False))}
        out[fmt] = cell
    return out


def reader_thresholds(thresholds: Mapping) -> dict:
    """The reader row of the fitted thresholds, for reasoned episodes, by setting and format."""
    return {share: {cell: v for cell, v in cells.items() if cell.startswith("reasoned/")}
            for share, cells in thresholds[READER_SCORE].items()}


def one_set(records: Sequence[Mapping[str, Any]], thresholds: Mapping, reason_only: Path) -> dict:
    """The figures of one set, from its run's scored records restricted to reasoned episodes."""
    reasoned = [r for r in records if r["form"] == "reasoned"]
    contrast = contrast_records(reasoned, user_right_scores({r["episode_id"]: r["p"] for r in reasoned}))
    rows = {"episodes": len(reasoned), "contrast_questions": len(contrast),
            "reader_selection": _shown(interval(selection_score, contrast)),
            "reason_only": reason_only_score(reasoned, reason_only),
            "original": adoption_rates(reasoned), "reader_gate": {}}
    for share in gates_on_paper.SETTINGS:
        gated, replaced = gates_on_paper.apply(reasoned, thresholds, {}, READER_SCORE, share)
        rows["reader_gate"][str(share)] = {"replaced_episodes": len(replaced), **adoption_rates(gated)}
    return rows


def measure(runs: Mapping[str, tuple[dict, list[dict]]], thresholds: Mapping,
            reason_only: Mapping[str, Path]) -> dict:
    """The record's values: per set, the run's facts and its figures."""
    out = {"label": LABEL, "thresholds": reader_thresholds(thresholds), "sets": {}}
    for name, (meta, records) in runs.items():
        numbers = meta.get("numbers", {})
        out["sets"][name] = {
            "run": {k: meta.get(k) for k in ("run_id", "commit", "gpu", "session", "order_sha256", "split")},
            "appended_read_check": numbers.get("appended_read_check", {}).get("all"),
            **one_set(records, thresholds, reason_only[name])}
    if "expanded" in out["sets"]:
        out["sets"]["expanded"]["note"] = EXPANDED_NOTE
    return out


def _f(d: Mapping) -> str:
    def g(x):
        return "undefined" if x is None else f"{x:.4f}"
    return f"{g(d['estimate'])} [{g(d['low'])}, {g(d['high'])}]"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m check2.evaluation.new_conditions", description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=paths.ROOT / "results", help="the folder holding the run folders")
    args = parser.parse_args(argv)
    runs = {}
    for name, run_id in SETS.items():
        folder = args.runs / run_id
        if not (folder / "run.json").is_file():
            print(f"{name}: no run folder {folder}; left out")
            continue
        meta, records = load_run(folder)
        meta["numbers"] = json.loads((folder / "numbers.json").read_text(encoding="utf-8"))
        runs[name] = (meta, records)
    train = read_jsonl(gates_on_paper.TRAIN_EPISODES)
    first_turn = {r["qid"]: r["probabilities"] for r in read_jsonl(gates_on_paper.FIRST_TURN_FILE)}
    p_train = {r["episode_id"]: r["p"] for r in read_jsonl(gates_on_paper.P_TRAIN_FILE)}
    thresholds = gates_on_paper.fit(train, first_turn, p_train)
    saved = {name: (REASON_ONLY_EXPANDED if name == "expanded" else REASON_ONLY_P_TEST) for name in runs}
    values = measure(runs, thresholds, saved)
    path = write_record("measurement", "reader_new_conditions", values,
                        config={"runs": {name: meta["run_id"] for name, (meta, _) in runs.items()}})
    for name, s in values["sets"].items():
        ro = s["reason_only"]
        print(f"{name} ({s['run']['run_id']}, {s['run']['gpu']}): {s['episodes']} reasoned episodes, "
              f"{s['contrast_questions']} contrast questions; reader selection {_f(s['reader_selection'])}; "
              f"reason-only {_f(ro['selection']) if ro.get('exists') else ro.get('note')}; "
              f"appended-read agreement {s['appended_read_check']}")
        for fmt in FORMAT_ORDER:
            o = s["original"][fmt]["all"]
            line = f"  {fmt}: original wrong {_f(o['wrong_adoption'])} right {_f(o['right_adoption'])}"
            for share, g in s["reader_gate"].items():
                line += (f"; gate@{share} wrong {_f(g[fmt]['all']['wrong_adoption'])} "
                         f"right {_f(g[fmt]['all']['right_adoption'])}")
            print(line)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
