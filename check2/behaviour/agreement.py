"""How often a behaviour run's final answers equal another run's, on the reasoned episodes both hold.

    python -m check2.behaviour.agreement <behaviour run folder> <run folder with episodes.jsonl.gz>

Episodes are matched by id. The behaviour run's final answer is the one its reply states; a
reply that states none counts as not agreeing. The other run's final answer is its `a2`.
The counts are also given over the episodes whose reply states an answer in both runs.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from check2.behaviour.replies import read_replies
from check2.evaluation.scoring import score_stated


def _other_episodes(folder: Path) -> dict[str, dict]:
    with gzip.open(folder / "episodes.jsonl.gz", "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {r["episode_id"]: r for r in rows if r["form"] == "reasoned"}


def agreement(behaviour_folder: Path, other_folder: Path) -> dict:
    """Per format and together: episodes both runs hold, and how many end on the same answer."""
    other = _other_episodes(other_folder)
    mine = {r["item_id"]: r for r in read_replies(behaviour_folder) if r["kind"] == "reasoned"}
    out: dict[str, dict] = {}
    for fmt in ("snap", "short", "all"):
        both = sorted(i for i in mine.keys() & other.keys() if fmt == "all" or mine[i]["fmt"] == fmt)
        stated = [i for i in both if score_stated(mine[i]["text"], mine[i]["fmt"]) is not None]
        agree = [i for i in stated if score_stated(mine[i]["text"], mine[i]["fmt"]) == other[i]["a2"]]
        stated_both = [i for i in stated if other[i]["scored_by"] == "stated"]
        out[fmt] = {"both_hold": len(both), "agree": len(agree),
                    "share": len(agree) / len(both) if both else None,
                    "stated_in_this_run": len(stated), "stated_in_both": len(stated_both),
                    "agree_where_both_stated": sum(i in set(agree) for i in stated_both)}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("behaviour_folder", type=Path)
    parser.add_argument("other_folder", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(agreement(args.behaviour_folder, args.other_folder), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
