"""A run's folder as the runner leaves it, and the checks that let two runs' numbers be compared.

A run folder holds `run.json` and, for a run that scored episodes, `episodes.jsonl.gz` with
one scored record per line. Every reported number comes from one card type, and a paired
difference compares two runs made in one batch order under one set of library versions;
runs that break either are refused, never compared.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

# The repeat of the Original run is told from it by its config file's name; the two files are equal.
REPEAT_CONFIG = "original-repeat"
REPEAT_RUN_IDS = (REPEAT_CONFIG, f"{REPEAT_CONFIG}-heldout")


class IncompleteRun(ValueError):
    """A run folder whose record is not complete."""


class NotComparable(ValueError):
    """Runs that were made on different cards, in different orders or under different versions."""


def load_run(folder: Path) -> tuple[dict, list[dict]]:
    """The run's record and its scored records; a run that is not complete is refused."""
    folder = Path(folder)
    meta = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    if meta.get("status") != "complete" or meta.get("episodes_written") != meta.get("episodes_expected"):
        raise IncompleteRun(
            f"run {meta.get('run_id', folder.name)} is not complete: status {meta.get('status')!r}, "
            f"{meta.get('episodes_written')} of {meta.get('episodes_expected')} episodes written"
        )
    episodes = folder / "episodes.jsonl.gz"
    records = []
    if episodes.exists():
        with gzip.open(episodes, "rt", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    return meta, records


def is_repeat(run_id: str) -> bool:
    """Whether the run is the Original run's repeat, on `test` or on `heldout`."""
    return run_id in REPEAT_RUN_IDS


def require_one_card(metas: Iterable[Mapping[str, Any]]) -> None:
    """Refuse runs whose records name more than one card."""
    cards: dict[Any, list[str]] = {}
    for meta in metas:
        cards.setdefault(meta.get("gpu"), []).append(meta.get("run_id"))
    if len(cards) > 1:
        shown = "; ".join(f"{card!r}: {', '.join(sorted(ids))}" for card, ids in sorted(cards.items(), key=str))
        raise NotComparable(f"the runs were made on more than one card ({shown}); runs from two cards are never "
                            "compared")


def require_comparable(a: Mapping[str, Any], b: Mapping[str, Any]) -> None:
    """Refuse two runs a difference pairs when their batch orders or their library versions differ."""
    for key, what in (("order_sha256", "batch orders"), ("versions", "library versions")):
        if a.get(key) != b.get(key):
            raise NotComparable(f"runs {a.get('id', a.get('run_id'))} and {b.get('id', b.get('run_id'))} were "
                                f"made with different {what} ({a.get(key)!r} against {b.get(key)!r}); they are "
                                "not compared")
