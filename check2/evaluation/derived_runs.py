"""Two inputs of the tables that are not runs of their own: the Original run's repeat and Gates on paper.

The repeat gives, per format, the share of episodes whose final answer is the same in the
Original run and in its repeat. Gates on paper leaves a derived record under
`results/<Original run id>-gates-on-paper/`; each of its twelve gated settings, three scores
at two settings under each of two rules, is the Original run with the final answer replaced
by the first answer on the episodes the record lists, and is rebuilt here as a run over the
Original run's episodes. A setting of the second rule is named with ` adoption only` after
the score and setting, and its rebuilt run's variant says so.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from check2.evaluation.rates import rate, rates
from check2.evaluation.readings import paired_records, selection
from check2.evaluation.scoring import FORMATS

GATES_ON_PAPER_FOLDER_SUFFIX = "-gates-on-paper"
GATES_ON_PAPER_ROLE = "gates on paper"
ADOPTION_ONLY = "adoption only"


def repeat_agreement(original: Sequence[Mapping[str, Any]], repeat: Sequence[Mapping[str, Any]]) -> dict:
    """Per format, the share of episodes with the same final answer in the Original run and its repeat."""
    out = {}
    for fmt in FORMATS:
        pairs = paired_records(original, repeat, selection(fmt))
        out[fmt] = rate(sum(a["a2"] == b["a2"] for a, b in pairs), len(pairs))
    return out


def _newest_record(folder: Path) -> Path | None:
    """The newest record in the folder; record names begin with their UTC time, so they sort by time."""
    found = sorted(folder.glob("*.json"))
    return found[-1] if found else None


def gated_runs(results: Path, runs: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Each gated setting of the newest Gates on paper record of each Original run, rebuilt as a run.

    A rebuilt run carries the id of its record, a role naming the score and the setting, the
    Original run as its `reference`, and the Original run's split, card, order and versions.
    Its variant is the Original run's, followed by `adoption only` for a setting of that rule.
    A record that names a run the tables do not hold, an episode that run does not hold, or
    rates that the rebuilt records do not give, is refused.
    """
    by_id = {r["id"]: r for r in runs}
    out = []
    for folder in sorted(p for p in Path(results).iterdir()
                         if p.is_dir() and p.name.endswith(GATES_ON_PAPER_FOLDER_SUFFIX) and not (p / "run.json").exists()):
        path = _newest_record(folder)
        if path is None:
            continue
        values = json.loads(path.read_text(encoding="utf-8"))["values"]
        source = by_id.get(values["source_run"])
        if source is None or folder.name != f"{values['source_run']}{GATES_ON_PAPER_FOLDER_SUFFIX}":
            raise ValueError(f"{folder.name}/{path.name} names the run {values['source_run']!r}, which is not an "
                             "Original run the tables hold")
        episode_ids = {r["episode_id"] for r in source["records"]}
        for setting, gated in sorted(values["gated"].items()):
            replaced = set(gated["replaced_episodes"])
            if not replaced <= episode_ids:
                raise ValueError(f"{folder.name}/{path.name}: {setting} replaces answers of episodes the run "
                                 f"{source['id']} does not hold")
            records = [{**r, "a2": r["a1"]} if r["episode_id"] in replaced else r for r in source["records"]]
            if json.loads(json.dumps(rates(records))) != gated["rates"]:
                raise ValueError(f"{folder.name}/{path.name}: the rates of {setting} are not those of the run "
                                 f"{source['id']} with its listed answers replaced")
            rule = ADOPTION_ONLY if setting.endswith(f" {ADOPTION_ONLY}") else ""
            variant = ", ".join(part for part in (source["variant"], rule) if part)
            out.append({**{k: source[k] for k in ("split", "gpu", "order_sha256", "versions")},
                        "id": f"{folder.name}/{path.name}", "role": f"{GATES_ON_PAPER_ROLE} {setting}",
                        "variant": variant, "records": records, "reference": source["id"]})
    return out
