"""A run's folder: `run.json`, the scored records written batch by batch, and `numbers.json`.

`episodes.jsonl.gz` grows by one gzip member per batch. After each member is on disk,
`progress.json` records how many batches are complete, the file's length at that point, the
seconds and replies each format took, and the order hash. A run that resumes cuts the file
back to that length, so a batch half written when a session was lost is written again, and
checks that the records it keeps are the first batches of the same order.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

from check2 import record
from check2.evaluation.rates import rate, rates
from check2.evaluation.roles import base_role
from check2.evaluation.runs import appended_check
from check2.evaluation.runs.parts import RunError

EPISODES_FILE = "episodes.jsonl.gz"
PROGRESS_FILE = "progress.json"
NUMBERS_FILE = "numbers.json"
RUN_FILE = "run.json"


def run_folder(run_id: str) -> Path:
    return record.ROOT / record.RESULTS / run_id


def read_records(folder: Path) -> list[dict]:
    path = folder / EPISODES_FILE
    if not path.exists():
        return []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_atomic(path: Path, value: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


class RunOutput:
    """The folder of one run, opened for a fresh start or for resuming."""

    def __init__(self, run_id: str, settings: dict, order_hash: str, expected: int) -> None:
        self.run_id, self.order_hash, self.expected = run_id, order_hash, expected
        self.swap = base_role(settings) == "swap"
        self.folder = run_folder(run_id)
        self.run_file = self.folder / RUN_FILE
        self.progress = {"batches": 0, "bytes": 0, "order_sha256": order_hash, "seconds": {}, "replies": {}}
        if self.run_file.exists():
            self._resume(settings)
        else:
            if (self.folder / EPISODES_FILE).exists() or (self.folder / PROGRESS_FILE).exists():
                raise RunError(f"{self.folder} holds outputs without a run record; it is not overwritten")
            record.start_run(run_id, settings)
            self._save_progress()

    def _resume(self, settings: dict) -> None:
        meta = json.loads(self.run_file.read_text(encoding="utf-8"))
        if meta.get("status") == "complete":
            raise RunError(f"run {self.run_id} is already complete")
        if meta.get("config") != json.loads(json.dumps(settings)):
            raise RunError(f"run {self.run_id} was started with other settings; it is not resumed")
        state = record.source_state()
        if meta.get("commit") != state["commit"]:
            raise RunError(f"run {self.run_id} was started at commit {meta.get('commit')}, not {state['commit']}; "
                           "a run is resumed only at its own commit")
        progress_file = self.folder / PROGRESS_FILE
        if progress_file.exists():
            self.progress = json.loads(progress_file.read_text(encoding="utf-8"))
        if self.progress["order_sha256"] != self.order_hash:
            raise RunError(f"run {self.run_id} was started over another episode order")
        path = self.folder / EPISODES_FILE
        if path.exists():
            with path.open("r+b") as f:
                f.truncate(self.progress["bytes"])
        elif self.progress["bytes"]:
            raise RunError(f"run {self.run_id}: its progress names written batches, and its records are missing")

    def _save_progress(self) -> None:
        _write_atomic(self.folder / PROGRESS_FILE, self.progress)

    def note(self, key: str, value) -> None:
        """Keep a fact about the run's inputs; a resumed run must find the same value."""
        notes = self.progress.setdefault("notes", {})
        if key in notes and notes[key] != value:
            raise RunError(f"run {self.run_id}: {key} was {notes[key]!r} and is now {value!r}; it is not resumed")
        notes[key] = value
        self._save_progress()

    @property
    def batches_done(self) -> int:
        return self.progress["batches"]

    def append(self, rows: list[dict], fmt: str, seconds: float) -> None:
        """Write one batch's records as a gzip member, then record the batch as complete."""
        path = self.folder / EPISODES_FILE
        text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
        with path.open("ab") as f:
            f.write(gzip.compress(text.encode("utf-8"), mtime=0))
            f.flush()
            os.fsync(f.fileno())
            size = f.tell()
        self.progress["batches"] += 1
        self.progress["bytes"] = size
        self.progress["seconds"][fmt] = self.progress["seconds"].get(fmt, 0.0) + seconds
        self.progress["replies"][fmt] = self.progress["replies"].get(fmt, 0) + len(rows)
        self._save_progress()

    def fail(self, written: int) -> None:
        record.finish_run(self.run_file, "failed", self.expected, written, self.order_hash)

    def finish(self, expected_ids: list[str]) -> Path:
        """Check the records against the order; complete the run and write its numbers, or mark it failed."""
        records = read_records(self.folder)
        ids = [r["episode_id"] for r in records]
        if ids != expected_ids:
            self.fail(len(records))
            raise RunError(f"run {self.run_id}: {len(records)} of {len(expected_ids)} episodes written, or not in "
                           "the run's order; the run is not complete")
        numbers = summarise(records, self.progress, swap=self.swap)
        _write_atomic(self.folder / NUMBERS_FILE, numbers)
        record.finish_run(self.run_file, "complete", self.expected, len(records), self.order_hash)
        return self.folder


def summarise(records: list[dict], progress: dict, swap: bool = False) -> dict:
    """`numbers.json`: the rates, the check on the appended read, and the replies a second per format.

    A Swap run's `recheck` episodes are left out of its rates; their records stay in the
    episode file. A run whose appended read agrees with the stated answers on less than
    97% of the checked sample also carries its rates on stated answers only.
    """
    speed = {fmt: {"replies": progress["replies"][fmt], "seconds": round(progress["seconds"][fmt], 3),
                   "replies_per_second": progress["replies"][fmt] / progress["seconds"][fmt]
                   if progress["seconds"][fmt] else None}
             for fmt in sorted(progress["replies"])}
    check = appended_check.agreement(records)
    numbers = {"appended_read_check": check, "speed": speed,
               "inputs": progress.get("notes", {}),
               "every_reply_has_a2": all(r["a2"] is not None for r in records)}
    if any(r["form"] == "neutral" for r in records):
        # The neutral follow-up reports the share of episodes ending on the first answer.
        by_format = {}
        for r in records:
            n = by_format.setdefault(r["format"], [0, 0])
            n[0] += r["a2"] == r["a1"]
            n[1] += 1
        numbers["ends_on_first_answer"] = {fmt: rate(*n) for fmt, n in sorted(by_format.items())}
    else:
        scored = [r for r in records if r["situation"] != "recheck"] if swap else records
        numbers["rates"] = rates(scored)
        numbers["rates_wrong_first"] = rates(scored, wrong_first_only=True)
        if appended_check.below_least_agreement(check):
            numbers["rates_stated_only"] = rates(scored, stated_only=True)
    return numbers
