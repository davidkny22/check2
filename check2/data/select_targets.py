"""The training targets chosen from the sampled replies, and the count of how each episode ended up.

    python -m check2.data.select_targets [--folder DIR]

Reads `samples.jsonl.gz` from the folder, and `samples_resampled.jsonl.gz` when it exists,
whose samples replace the first file's for the episodes it holds. Every named and reasoned
training episode ends up in exactly one of four outcomes: `sampled` (a qualifying sample is
its target), `constructed` (snap format, no qualifying sample), `dropped` (short format, no
qualifying sample) or `unsampled` (no samples in the files). Writes `targets.jsonl`, in the
order of the episode file, and `report.json` with the outcomes counted per situation and
format.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from check2.data.episodes import FORMATS
from check2.data.pool import Question, complying, load_pool
from check2.data.sampling import RESAMPLED, SAMPLES, TARGETS_DIR, TRAIN_EPISODES, read_jsonl, \
    training_episodes, write_jsonl
from check2.data.targets import TARGET_FORMS, construct_snap_target, select_target
from check2.evaluation.scoring import SITUATIONS, score_stated

OUTCOMES = ("sampled", "constructed", "dropped", "unsampled")
TARGETS_NAME = "targets.jsonl"
REPORT_NAME = "report.json"


def samples_by_episode(folder: Path) -> tuple[dict[str, list[dict]], list[str]]:
    """Each episode's samples in sample order, and the files they came from."""
    by_episode: dict[str, list[dict]] = {}
    files = []
    for path in (folder / SAMPLES.name, folder / RESAMPLED.name):
        if not path.exists():
            continue
        files.append(path.name)
        for line in read_jsonl(path):
            by_episode[line["episode_id"]] = line["samples"]
    return by_episode, files


def outcome_of(episode: dict, samples: list[dict] | None, question: Question) -> tuple[str, dict | None]:
    """The episode's outcome and its target record, if it has one."""
    if samples is None:
        return "unsampled", None
    target = select_target(samples, episode)
    if target is not None:
        return "sampled", target
    if episode["format"] == "snap":
        return "constructed", construct_snap_target(episode, question)
    return "dropped", None


def select_all(episodes: list[dict], samples: dict[str, list[dict]],
               pool: dict[str, Question]) -> tuple[list[dict], dict[str, str]]:
    """The target records in episode order, and each episode's outcome."""
    targets, outcomes = [], {}
    for e in episodes:
        outcome, target = outcome_of(e, samples.get(e["episode_id"]), pool[e["qid"]])
        outcomes[e["episode_id"]] = outcome
        if target is not None:
            targets.append(target)
    return targets, outcomes


def _counts(rows: list[dict], outcomes: dict[str, str]) -> dict:
    counts = {o: sum(outcomes[e["episode_id"]] == o for e in rows) for o in OUTCOMES}
    return {"episodes": len(rows), **counts}


def build_report(episodes: list[dict], outcomes: dict[str, str], samples: dict[str, list[dict]],
                 files: list[str]) -> dict:
    """Outcomes per situation and format, the no-qualifying shares per form, and the replies' ends."""
    cells = []
    for fmt in FORMATS:
        for sit in SITUATIONS:
            rows = [e for e in episodes if e["format"] == fmt and e["situation"] == sit]
            counts = _counts(rows, outcomes)
            sampled_rows = counts["episodes"] - counts["unsampled"]
            counts["dropped_share"] = round(counts["dropped"] / sampled_rows, 4) if sampled_rows else None
            counts["samples_per_episode"] = sorted({len(samples[e["episode_id"]]) for e in rows
                                                    if e["episode_id"] in samples})
            cells.append({"situation": sit, "format": fmt, **counts})
    by_form = []
    for fmt in FORMATS:
        for form in TARGET_FORMS:
            for sit in SITUATIONS:
                rows = [e for e in episodes if (e["format"], e["form"], e["situation"]) == (fmt, form, sit)
                        and outcomes[e["episode_id"]] != "unsampled"]
                none = sum(outcomes[e["episode_id"]] in ("constructed", "dropped") for e in rows)
                by_form.append({"form": form, "situation": sit, "format": fmt, "episodes": len(rows),
                                "no_qualifying": none,
                                "no_qualifying_share": round(none / len(rows), 4) if rows else None})
    replies = {}
    for fmt in FORMATS:
        all_samples = [s for e in episodes if e["format"] == fmt for s in samples.get(e["episode_id"], [])]
        ended = sum(s["ended"] for s in all_samples)
        stated = sum(score_stated(s["text"], fmt) is not None for s in all_samples)
        n = len(all_samples)
        replies[fmt] = {"replies": n, "ended": ended, "ended_share": round(ended / n, 4) if n else None,
                        "stated": stated, "stated_share": round(stated / n, 4) if n else None}
    return {"episodes": len(episodes), **{o: sum(v == o for v in outcomes.values()) for o in OUTCOMES},
            "cells": cells, "by_form": by_form, "replies": replies, "sample_files": files}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folder", type=Path, default=TARGETS_DIR,
                        help="the folder that holds the samples and receives the targets and report")
    args = parser.parse_args(argv)
    episodes = training_episodes(TRAIN_EPISODES)
    samples, files = samples_by_episode(args.folder)
    if not files:
        raise SystemExit(f"no samples file in {args.folder}")
    unknown = sorted(set(samples) - {e["episode_id"] for e in episodes})
    if unknown:
        raise SystemExit(f"the samples name {len(unknown)} episodes that are not training targets, "
                         f"such as {unknown[:3]}")
    targets, outcomes = select_all(episodes, samples, complying(load_pool()))
    report = build_report(episodes, outcomes, samples, files)
    write_jsonl(args.folder / TARGETS_NAME, targets)
    (args.folder / REPORT_NAME).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8",
                                           newline="\n")
    print(f"{len(targets)} targets; " + ", ".join(f"{o} {report[o]}" for o in OUTCOMES))
    for cell in report["cells"]:
        print(f"  {cell['format']:5} {cell['situation']:10} " + " ".join(
            f"{k}={cell[k]}" for k in ("episodes", *OUTCOMES, "dropped_share")))
    for row in report["by_form"]:
        if row["format"] == "snap":
            print(f"  snap {row['form']:8} {row['situation']:10} no qualifying sample: "
                  f"{row['no_qualifying']}/{row['episodes']} ({row['no_qualifying_share']})")
    for fmt, r in report["replies"].items():
        print(f"  {fmt} replies: {json.dumps(r)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
