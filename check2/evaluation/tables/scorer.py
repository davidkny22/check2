"""How each run's replies were scored, and which runs' rates are also given on stated answers only.

A reply that states its final answer is scored by its format's pattern; any other is scored
by the appended read. Each run records the appended read's agreement with the stated answer
on its checked sample, in `numbers.json`. Where that agreement is below the least the
scorer accepts, the run's rates are also given computed on the replies that state a final
answer only, beside the full rates.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from check2.evaluation.rates import rate
from check2.evaluation.scoring import FORMATS

NUMBERS = "numbers.json"
SNAP = FORMATS[0]


def agreement(folder: Path) -> dict | None:
    """The run's appended-read agreement on its checked sample, over all of it, or None where none is recorded."""
    path = Path(folder) / NUMBERS
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("appended_read_check", {}).get("all")


def least_agreement() -> float:
    # Imported here: the run execution package reads run folders through the tables package.
    from check2.evaluation.runs.appended_check import LEAST_AGREEMENT
    return LEAST_AGREEMENT


def needs_stated_rows(run: Mapping[str, Any]) -> bool:
    """Whether the run's agreement is recorded and below the least the scorer accepts."""
    found = run.get("agreement")
    return found is not None and found["rate"] is not None and found["rate"] < least_agreement()


def scorer_rows(runs: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Per run and format: the run's agreement, the shares of replies scored each way, and, in the snap format,
    the count of replies whose end-of-turn token did not come inside the cap."""
    rows = []
    for run in runs:
        found = run.get("agreement") or {}
        for fmt in FORMATS:
            replies = [r for r in run["records"] if r["format"] == fmt]
            if not replies:
                continue
            stated = rate(sum(r["scored_by"] == "stated" for r in replies), len(replies))
            appended = rate(sum(r["scored_by"] == "appended" for r in replies), len(replies))
            rows.append({"runs": run["id"], "split": run["split"], "role": run["role"], "variant": run["variant"],
                         "agreement_numerator": found.get("numerator"), "agreement_denominator": found.get("denominator"),
                         "agreement": found.get("rate"), "least_agreement": least_agreement(),
                         "rates_on_stated_answers_given": needs_stated_rows(run), "format": fmt, "replies": len(replies),
                         "scored_stated": stated["rate"], "scored_appended": appended["rate"],
                         "not_ended": sum(not r["ended"] for r in replies) if fmt == SNAP else None})
    return rows
