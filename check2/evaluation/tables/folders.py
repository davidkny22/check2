"""The run folders the tables read: each complete run with its role, the Original run's repeats and the runs left out.

A run folder that holds `not_of_record.json`, `{"reason": "<one sentence>"}`, is not of
record: it is not read at all, it takes no part in the check that every run was made on one
card, and the tables list it with its reason.
"""

from __future__ import annotations

import json
from pathlib import Path

from check2.evaluation.roles import NEUTRAL_FOLLOW_UP, base_role, named_settings, run_role
from check2.evaluation.run_folders import is_repeat, load_run, require_one_card
from check2.evaluation.tables.scorer import agreement

RUN_FACTS = ("gpu", "order_sha256", "versions")
NOT_OF_RECORD = "not_of_record.json"


def not_of_record_reason(folder: Path) -> str | None:
    """The reason the folder's marker gives, or None where the folder carries no marker.

    A marker without a reason in one non-empty string is refused.
    """
    path = Path(folder) / NOT_OF_RECORD
    if not path.is_file():
        return None
    reason = json.loads(path.read_text(encoding="utf-8")).get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"{Path(folder).name}/{NOT_OF_RECORD} gives no reason")
    return reason


def _folders(results: Path) -> tuple[list[dict], list[dict], list[str], list[dict]]:
    """The runs the tables use, the Original run's repeats, the ids of the neutral follow-up runs, and the
    folders not of record, each as `{"folder", "reason"}`.

    Every complete run folder of record counts towards the one-card check, whether or not it is used.
    """
    runs, repeats, neutral, metas, marked = [], [], [], [], []
    for folder in sorted(p for p in Path(results).iterdir() if (p / "run.json").is_file()):
        reason = not_of_record_reason(folder)
        if reason is not None:
            marked.append({"folder": folder.name, "reason": reason})
            continue
        meta, records = load_run(folder)
        metas.append(meta)
        if not records:
            continue
        role, variant = run_role(meta["config"])
        run = {"id": meta["run_id"], "split": meta["split"], "role": role, "variant": variant, "records": records,
               "base": base_role(meta["config"]), "named": named_settings(meta["config"]), "agreement": agreement(folder),
               **{k: meta.get(k) for k in RUN_FACTS}}
        if NEUTRAL_FOLLOW_UP in variant:
            neutral.append(run)
        elif is_repeat(meta["run_id"]):
            repeats.append(run)
        else:
            runs.append(run)
    require_one_card(metas)
    return runs, repeats, neutral, marked


def discover(results: Path) -> tuple[list[dict], list[str]]:
    """Every run folder of record under `results` that scored episodes, in run-id order, each with its role.

    Returns the runs the tables use and the ids of the runs they leave out: the neutral
    follow-up runs and the Original run's repeat. Folders not of record are in neither list.
    """
    runs, repeats, neutral, _ = _folders(results)
    return runs, sorted([r["id"] for r in neutral] + [r["id"] for r in repeats])
