"""The result tables, written as CSV and as Markdown from the run folders of a results folder.

    python -m check2.evaluation.tables <results folder> <output folder>

Every run whose `run.json` is present must be complete; otherwise nothing is written. Each
run's role (Original, Check², Off and so on) is read from its configuration's settings.
Every row of every table names the runs it was computed from.

Tables: the rates of every run by format, form and situation, over all questions and
over wrong-first questions; every paired difference from the Original run over the same
episodes; the paired differences between named runs, in the same form; the pass bar per
format, on the whole of `test` and on each part; the pass bar's two quantities with no
verdict for the runs added beside Check² and for the cross-writer runs; the causal
readings, with the presence and content readings of the wide-signal variant from its own
runs; the neutral follow-up's share of episodes ending on the first answer; and the
snap-against-short comparison. A run of the second stage of signal forming is compared
with the formed model's run over the same split, the formed model with the Original run,
and the readings hold the excess of the second stage over Check².

The scorer table gives each run's appended-read agreement on its checked sample, the
shares of replies scored each way and, in the snap format, the count of replies that did
not end inside the cap. A run whose agreement is below the least the scorer accepts also
has its rates computed on the replies that state a final answer only, marked as such in the
rates table's `replies` column.

Each of the twelve gated settings of Gates on paper, under its first rule and under
adoption only, rebuilt from its newest record, has its rates and its paired differences
from the Original run the record names; an adoption-only setting carries `adoption only` in
its role and its variant. Beside every paired difference and pass-bar row of a split
stands, per format, the share of episodes with the same final answer in the Original run
and its repeat, empty where the split has no repeat.

Two kinds of run are left out of the rates and named: a neutral follow-up run, which reports
the share of episodes ending on the first answer in its own table, and the Original run's
repeat, which is the reference of nothing. A folder that carries the marker of a run that is
not of record is not read at all, and the table `not_of_record` lists it with its reason.
The command refuses, in one line and with a non-zero exit, runs made on more than one card,
two compared runs made in different orders or under different library versions, and any
folder it cannot read as the run table.
"""

from __future__ import annotations

import csv
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from check2.evaluation.derived_runs import gated_runs
from check2.evaluation.roles import run_role
from check2.evaluation.run_folders import IncompleteRun, is_repeat, load_run
from check2.evaluation.tables.added import neutral_rows, pass_bar_quantity_rows, wide_readings
from check2.evaluation.tables.folders import _folders, discover
from check2.evaluation.tables.named_pairs import kept_runs, rows_of_named_pairs
from check2.evaluation.tables.scorer import scorer_rows
from check2.evaluation.tables.rows import (_agreement_columns, _agreements, _one, _original_for, _rows_of_differences,
                                           _rows_of_rates, _rows_of_readings)

__all__ = ["IncompleteRun", "discover", "load_run", "main", "run_role", "write_table", "write_tables",
           "_one", "_original_for"]


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def write_table(rows: Sequence[Mapping[str, Any]], output: Path, name: str) -> None:
    """Write the rows as `<name>.csv` and `<name>.md` under `output`."""
    columns: list[str] = []
    for row in rows:
        columns += [c for c in row if c not in columns]
    with open(output / f"{name}.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(_cell(row.get(c)) for c in columns) + " |" for row in rows]
    (output / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tables(results: Path, output: Path) -> tuple[list[str], list[str], list[dict]]:
    """Compute every table from the runs under `results` and write them under `output`.

    Returns the table names, the ids of the runs left out of the rates, and the folders not
    of record with their reasons. Nothing is written when a table cannot be computed.
    """
    runs, repeats, neutral, marked = _folders(Path(results))
    agreements = _agreements(runs, repeats)
    gated = gated_runs(Path(results), runs)
    bar, causal, snap = _rows_of_readings(runs, agreements)
    on_test = _agreement_columns(agreements, "test")
    tables = {
        "rates": _rows_of_rates(runs + gated),
        "differences_from_original": _rows_of_differences(runs + gated, agreements),
        "differences_between_runs": rows_of_named_pairs(runs + gated, kept_runs(Path(results)), agreements),
        "pass_bar": bar,
        "pass_bar_quantities": pass_bar_quantity_rows(runs, on_test),
        "readings": causal + wide_readings(runs, on_test),
        "neutral_follow_up": neutral_rows(neutral),
        "scorer": scorer_rows(runs + repeats + neutral),
        "snap_against_short": snap,
        "not_of_record": marked,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_table(rows, output, name)
    return list(tables), sorted([r["id"] for r in neutral] + [r["id"] for r in repeats]), marked


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("usage: python -m check2.evaluation.tables <results folder> <output folder>", file=sys.stderr)
        return 2
    try:
        names, left_out, marked = write_tables(Path(args[0]), Path(args[1]))
    except ValueError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 1
    print("wrote " + ", ".join(names) + f" under {args[1]}")
    repeats = [r for r in left_out if is_repeat(r)]
    neutral = [r for r in left_out if not is_repeat(r)]
    if neutral:
        print("left out, as neutral follow-up runs: " + ", ".join(neutral))
    if repeats:
        print("left out, as the Original run's repeat: " + ", ".join(repeats))
    for found in marked:
        print(f"left out, as not of record: {found['folder']} ({found['reason']})")
    return 0
