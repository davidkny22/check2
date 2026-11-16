"""Dropped-episode report: how many training episodes have a sampled, constructed or no target.

    python -m check2.gates.dropped_episodes [--folder DIR]

Reads `report.json` and `targets.jsonl` from the targets folder and recounts the outcomes
from the targets and the training episodes, so a report that disagrees with the files it
describes is caught. Red when, in the short format, a (situation, format) cell drops 15% or
more of its episodes; also red when an episode was never sampled, when a short-format
target was constructed or a snap episode dropped, or when the report and the recount
disagree. For each cell that drops too many it writes `resample_cells.json` beside the
report, the cells to sample again with more replies per episode.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from check2.data.episodes import FORMATS
from check2.data.sampling import TARGETS_DIR, TRAIN_EPISODES, read_jsonl, training_episodes
from check2.data.select_targets import REPORT_NAME, TARGETS_NAME
from check2.evaluation.scoring import SITUATIONS
from check2.record import write_record

# A short-format cell is red when its dropped share reaches this value.
DROPPED_LIMIT = 0.15
RESAMPLE_CELLS_NAME = "resample_cells.json"


def recount(episodes: list[dict], targets: list[dict]) -> dict[tuple[str, str], dict]:
    """Per (situation, format): episodes, targets by source, and episodes with no target."""
    source_of = {t["episode_id"]: t["source"] for t in targets}
    counts = {}
    for fmt in FORMATS:
        for sit in SITUATIONS:
            rows = [e for e in episodes if e["format"] == fmt and e["situation"] == sit]
            sources = [source_of.get(e["episode_id"]) for e in rows]
            counts[(sit, fmt)] = {"episodes": len(rows), "sampled": sources.count("sampled"),
                                  "constructed": sources.count("constructed"),
                                  "without_target": sources.count(None)}
    return counts


def judge(report: dict, episodes: list[dict], targets: list[dict]) -> tuple[list[str], list[dict], list[dict]]:
    """The findings, the cells that drop too many, and the per-cell table."""
    findings, red_cells, table = [], [], []
    episode_ids = {e["episode_id"] for e in episodes}
    strangers = sorted({t["episode_id"] for t in targets} - episode_ids)
    if strangers:
        findings.append(f"{len(strangers)} targets name no training episode, such as {strangers[:3]}")
    if len(targets) != len({t["episode_id"] for t in targets}):
        findings.append("an episode has more than one target")
    if report.get("unsampled", 0):
        findings.append(f"{report['unsampled']} episodes were never sampled")
    expected = recount(episodes, targets)
    reported = {(c["situation"], c["format"]): c for c in report["cells"]}
    if set(reported) != set(expected):
        findings.append(f"the report's cells {sorted(reported)} are not the cells {sorted(expected)}")
    for (sit, fmt), want in expected.items():
        got = reported.get((sit, fmt))
        if got is None:
            continue
        if (got["episodes"], got["sampled"], got["constructed"], got["dropped"] + got["unsampled"]) != \
                (want["episodes"], want["sampled"], want["constructed"], want["without_target"]):
            findings.append(f"{fmt} {sit}: the report says {got}, the files give {want}")
        if fmt == "short" and got["constructed"]:
            findings.append(f"short {sit}: {got['constructed']} targets were constructed")
        if fmt == "snap" and got["dropped"]:
            findings.append(f"snap {sit}: {got['dropped']} episodes were dropped")
        share = got["dropped"] / got["episodes"] if got["episodes"] else 0.0
        row = {"situation": sit, "format": fmt, "episodes": got["episodes"], "sampled": got["sampled"],
               "constructed": got["constructed"], "dropped": got["dropped"],
               "dropped_share": round(share, 4), "samples_per_episode": got.get("samples_per_episode")}
        table.append(row)
        if fmt == "short" and share >= DROPPED_LIMIT:
            red_cells.append({"situation": sit, "format": fmt})
            findings.append(f"short {sit}: {got['dropped']} of {got['episodes']} dropped ({share:.2%}), "
                            f"at or above {DROPPED_LIMIT:.0%}")
    return findings, red_cells, table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folder", type=Path, default=TARGETS_DIR,
                        help="the folder that holds the report and the targets")
    args = parser.parse_args(argv)
    report = json.loads((args.folder / REPORT_NAME).read_text(encoding="utf-8"))
    targets = read_jsonl(args.folder / TARGETS_NAME)
    episodes = training_episodes(TRAIN_EPISODES)
    findings, red_cells, table = judge(report, episodes, targets)
    if red_cells:
        (args.folder / RESAMPLE_CELLS_NAME).write_text(json.dumps(red_cells, indent=1) + "\n",
                                                       encoding="utf-8", newline="\n")
    verdict = "red" if findings else "green"
    values = {"cells": table, "findings": findings, "red_cells": red_cells,
              "by_form": report.get("by_form"), "replies": report.get("replies"),
              "sample_files": report.get("sample_files"),
              "totals": {k: report.get(k) for k in ("episodes", "sampled", "constructed", "dropped",
                                                     "unsampled")}}
    record = write_record("gate", "dropped_episodes", values, verdict=verdict,
                          config={"dropped_limit": DROPPED_LIMIT})
    for row in table:
        print(f"{row['format']:5} {row['situation']:10} episodes={row['episodes']} sampled={row['sampled']} "
              f"constructed={row['constructed']} dropped={row['dropped']} "
              f"dropped_share={row['dropped_share']} samples={row['samples_per_episode']}")
    for finding in findings:
        print(f"FINDING {finding}")
    if red_cells:
        print(f"cells to sample again: {args.folder / RESAMPLE_CELLS_NAME}")
    print(f"verdict: {verdict}  record: {record}")
    return 0 if verdict == "green" else 1


if __name__ == "__main__":
    sys.exit(main())
