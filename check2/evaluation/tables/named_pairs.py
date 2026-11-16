"""Paired differences between named runs, in the form of the differences from the Original run.

Each pair is a first run minus a second, over one split, on the episodes both runs score: a
Swap run leaves its `recheck` episodes out at scoring, so its partner is compared on the
Swap run's episodes. Two runs that hold different episodes are refused, and a pair is never
taken across splits. Per format the differences are given on reasoned pushback and, where
both runs hold it, on named pushback, in wrong adoption, right adoption and `correct_end`,
with the intervals of every other paired difference.

The pairs, where both runs are present:

- on `test`, Check² against System prompt, Connector, Adapter only, each dial setting, each
  band setting, the second seed, the wide-signal variant and Check² with the answer weight;
- on `test`, the wide-signal variant against its run with the write off and its Swap run;
- on `heldout`, the perfect-signal control with the answer weight against the perfect-signal
  control, each at the epoch the checkpoint rule kept;
- on `test`, the reader's gate of Gates on paper at each setting under each rule against
  Check² and against System prompt.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from check2.config.schema import WIDE_WRITER_INPUT
from check2.evaluation import readings
from check2.evaluation.derived_runs import GATES_ON_PAPER_ROLE
from check2.evaluation.run_folders import require_comparable
from check2.evaluation.scoring import FORMATS
from check2.evaluation.tables.rows import _agreement_columns, _interval_columns, _scored

Run = Mapping[str, Any]
Predicate = Callable[[Run], bool]

PAIR_RATES = ("wrong_adoption", "right_adoption", "correct_end")
PAIR_FORMS = ("reasoned", "named")
CHECKPOINT_RULE_RECORDS = Path("measurements") / "checkpoint_rule"
EPOCHS = "epochs"
WIDE = f"input={WIDE_WRITER_INPUT}"


def _is(base: str, *named: str) -> Predicate:
    return lambda run: run.get("base") == base and tuple(run.get("named", ())) == named


def _starts(prefix: str) -> Predicate:
    return lambda run: (run.get("base") or "").startswith(prefix) and not run.get("named")


def _names_one(base: str, setting: str) -> Predicate:
    """The base role with one named setting, `setting=<any value>`."""
    def match(run: Run) -> bool:
        named = tuple(run.get("named", ()))
        return run.get("base") == base and len(named) == 1 and named[0].startswith(f"{setting}=")
    return match


CHECK2 = _is("check2")
SYSTEM_PROMPT = _is("system prompt")
WIDE_CHECK2 = _is("check2", WIDE)
SECOND_SEED = _names_one("check2", "seed")
ANSWER_WEIGHTED = _names_one("check2", "answer_weight")
PERFECT = _is("perfect signal")
PERFECT_ANSWER_WEIGHTED = _names_one("perfect signal", "answer_weight")


def _reader_gate(run: Run) -> bool:
    """A gated setting of the reader's score, the last of the three scores of Gates on paper."""
    # Imported here: Gates on paper reads run folders through this package.
    from check2.evaluation.gates_on_paper import SCORES
    reader_score = SCORES[-1]
    return run["role"].startswith(f"{GATES_ON_PAPER_ROLE} {reader_score}@")


PARTNERS_OF_CHECK2 = (SYSTEM_PROMPT, _is("connector"), _is("adapter only"), _starts("dial "), _starts("bands off "),
                      SECOND_SEED, WIDE_CHECK2, ANSWER_WEIGHTED)


def kept_runs(results: Path) -> set[str]:
    """The run ids the checkpoint rule kept among a training's epochs, each from the newest record over its set of
    runs; the rule's records that rank steering strengths keep several runs and are not read here."""
    newest: dict[frozenset, str] = {}
    folder = Path(results) / CHECKPOINT_RULE_RECORDS
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["config"].get("kind") != EPOCHS:
            continue
        newest[frozenset(record["config"]["runs"])] = record["values"]["kept"]["label"]
    return set(newest.values())


def _all(runs: Sequence[Run], split: str, match: Predicate) -> list[Run]:
    return [r for r in runs if r["split"] == split and r["variant"] in ("", "adoption only") and match(r)]


def _single(runs: Sequence[Run], split: str, match: Predicate, kept: set[str] | None = None) -> Run | None:
    found = [r for r in _all(runs, split, match) if kept is None or r["id"] in kept]
    if len(found) > 1:
        raise ValueError(f"{len(found)} runs on {split} take one place in a named pair: {[r['id'] for r in found]}")
    return found[0] if found else None


def named_pairs(runs: Sequence[Run], kept: set[str]) -> list[tuple[Run, Run]]:
    """Every pair of the list above whose two runs are present, first run first."""
    pairs: list[tuple[Run, Run]] = []
    check2, system_prompt = _single(runs, "test", CHECK2), _single(runs, "test", SYSTEM_PROMPT)
    if check2:
        for match in PARTNERS_OF_CHECK2:
            pairs += [(check2, partner) for partner in _all(runs, "test", match)]
    wide = _single(runs, "test", WIDE_CHECK2)
    if wide:
        for partner in (_single(runs, "test", _is("off", WIDE)), _single(runs, "test", _is("swap", WIDE))):
            if partner:
                pairs.append((wide, partner))
    weighted = _single(runs, "heldout", PERFECT_ANSWER_WEIGHTED, kept)
    perfect = _single(runs, "heldout", PERFECT, kept)
    if weighted and perfect:
        pairs.append((weighted, perfect))
    for gate in _all(runs, "test", _reader_gate):
        pairs += [(gate, other) for other in (check2, system_prompt) if other]
    return pairs


def on_common_episodes(a: Run, b: Run) -> tuple[list, list]:
    """The two runs' records on the episodes both score; two runs over different episodes are refused."""
    if a["split"] != b["split"]:
        raise ValueError(f"runs {a['id']} and {b['id']} are over different splits; a pair is never taken across them")
    held_a = {r["episode_id"] for r in a["records"]}
    held_b = {r["episode_id"] for r in b["records"]}
    if held_a != held_b:
        raise ValueError(f"runs {a['id']} and {b['id']} do not hold the same episodes; they are not paired")
    common = {r["episode_id"] for r in _scored(a)} & {r["episode_id"] for r in _scored(b)}
    return ([r for r in a["records"] if r["episode_id"] in common],
            [r for r in b["records"] if r["episode_id"] in common])


def _name(run: Run) -> str:
    return run["role"]


def rows_of_named_pairs(runs: Sequence[Run], kept: set[str], agreements: Mapping[str, Mapping[str, Any]]) -> list[dict]:
    """One row per pair, format, form and rate, in the columns of the differences from the Original run."""
    rows = []
    for a, b in named_pairs(runs, kept):
        require_comparable(a, b)
        records_a, records_b = on_common_episodes(a, b)
        for fmt in FORMATS:
            for form in PAIR_FORMS:
                pairs = readings.paired_records(records_a, records_b, readings.selection(fmt, (form,)))
                if not pairs:
                    continue
                for rate_name in PAIR_RATES:
                    d = readings.difference(pairs, rate_name)
                    if d["episodes"] == 0:
                        continue
                    rows.append({"runs": f"{a['id']};{b['id']}", "split": a["split"], "role": a["role"],
                                 "variant": a["variant"], "question_set": "all questions", "format": fmt, "form": form,
                                 "situation": "all", "difference": f"{rate_name}, {_name(a)} minus {_name(b)}, points",
                                 **_interval_columns(d), **_agreement_columns(agreements, a["split"])})
    return rows
