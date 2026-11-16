"""Run execution: one config, its episodes in fixed batches through the split at the read token, its record.

    python experiments/run.py evaluate <config> [--epoch <k>]

calls `execute`, which writes `results/<run-id>/run.json`, `episodes.jsonl.gz` and, once
every episode is written, `numbers.json`.
"""

from check2.evaluation.runs.execution import execute, run, run_id_for, settings_of
from check2.evaluation.runs.parts import RunError

__all__ = ["RunError", "execute", "run", "run_id_for", "settings_of"]
