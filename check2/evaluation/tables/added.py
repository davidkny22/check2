"""The readings of the runs added beside Check², and the neutral follow-up's quantity.

- The presence and content readings for the wide-signal variant, from its own run with the
  write off and its own Swap run, beside Check²'s, in the readings table; and its reading
  against the adapter-only control, by the rule of Check²'s channel against the adapter.
- The two pass-bar quantities, with their intervals and no verdict, for the wide-signal
  variant, the second seed, each dial setting and Check² with the answer weight against the
  Original run on `test`, and for Check² against Original on the cross-writer episodes. Only
  Check² against Original on the whole of `test` has a verdict, in the pass-bar table.
- The neutral follow-up: per format, the share of episodes ending on the first answer for
  Original and for Check², and their paired difference.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from check2.evaluation import readings
from check2.evaluation.run_folders import require_comparable
from check2.evaluation.scoring import FORMATS
from check2.evaluation.tables.named_pairs import ANSWER_WEIGHTED, SECOND_SEED, WIDE, WIDE_CHECK2, _is, _starts
from check2.evaluation.tables.rows import _interval_columns, _reading_rows, _scored

Run = Mapping[str, Any]
CROSS_WRITER = "cross-writer"
WIDE_READING = "wide-signal variant"
QUANTITIES = ("wrong_adoption_drop", "right_adoption_drop")


def _find(runs: Sequence[Run], match, variant: str = "", split: str = "test") -> list[Run]:
    return [r for r in runs if r["split"] == split and r["variant"] == variant and match(r)]


def _single(runs: Sequence[Run], match, variant: str = "") -> Run | None:
    found = _find(runs, match, variant)
    if len(found) > 1:
        raise ValueError(f"{len(found)} runs on test take one place in a reading: {[r['id'] for r in found]}")
    return found[0] if found else None


def wide_readings(runs: Sequence[Run], agreement: Mapping[str, Any]) -> list[dict]:
    """Presence, content and the channel against the adapter for the wide-signal variant.

    Presence and content come from the variant's own Off and Swap runs. The channel against
    the adapter puts the variant's run in Check²'s place against the adapter-only control,
    the run Check²'s own reading uses, so the differences are the variant minus the control.
    """
    wide, off, swap = (_single(runs, m) for m in (WIDE_CHECK2, _is("off", WIDE), _is("swap", WIDE)))
    adapter = _single(runs, _is("adapter only"))
    rows = []
    for fmt in FORMATS:
        presence = None
        if wide and off:
            presence = readings.presence(off["records"], wide["records"], fmt)
            rows += _reading_rows(f"presence of the write, {WIDE_READING}", [off, wide], fmt, presence, agreement)
        if wide and swap and presence is not None:
            result = readings.content(_scored(swap), _scored(swap, wide["records"]), fmt, presence_reading=presence)
            rows += _reading_rows(f"content of the signal, {WIDE_READING}", [swap, wide, off], fmt, result, agreement)
        if wide and adapter:
            result = readings.channel_against_adapter(wide["records"], adapter["records"], fmt)
            rows += _reading_rows(f"channel against adapter, {WIDE_READING}", [wide, adapter], fmt, result, agreement)
    return rows


def _quantity_pairs(runs: Sequence[Run]) -> list[tuple[Run, Run]]:
    pairs = []
    original = _single(runs, _is("original"))
    if original:
        for match in (WIDE_CHECK2, SECOND_SEED, _starts("dial "), ANSWER_WEIGHTED):
            pairs += [(run, original) for run in _find(runs, match)]
    check2, original = _single(runs, _is("check2"), CROSS_WRITER), _single(runs, _is("original"), CROSS_WRITER)
    if check2 and original:
        pairs.append((check2, original))
    return pairs


def pass_bar_quantity_rows(runs: Sequence[Run], agreement: Mapping[str, Any]) -> list[dict]:
    """Per run and format, the two pass-bar quantities with their intervals; no verdict is written."""
    rows = []
    for run, original in _quantity_pairs(runs):
        require_comparable(run, original)
        for fmt in FORMATS:
            result = readings.pass_bar_quantities(run["records"], original["records"], fmt)
            for quantity in QUANTITIES:
                rows.append({"runs": f"{run['id']};{original['id']}", "variant": run["variant"], "format": fmt,
                             "quantity": f"{quantity}, original minus {run['role']}, points",
                             **_interval_columns(result[quantity]), **agreement})
    return rows


def neutral_rows(neutral: Sequence[Run]) -> list[dict]:
    """Per format, the share ending on the first answer for Original and Check², and Check² minus Original."""
    check2 = [r for r in neutral if r["split"] == "test" and r.get("base") == "check2" and not r.get("named")]
    original = [r for r in neutral if r["split"] == "test" and r.get("base") == "original" and not r.get("named")]
    if len(check2) != 1 or len(original) != 1:
        return []
    check2, original = check2[0], original[0]
    require_comparable(check2, original)
    rows = []
    for fmt in FORMATS:
        result = readings.ends_on_first_answer(check2["records"], original["records"], fmt)
        ids = f"{check2['id']};{original['id']}"
        for key, label in (("first_run", f"{check2['id']}, share ending on the first answer, points"),
                           ("second_run", f"{original['id']}, share ending on the first answer, points"),
                           ("first_minus_second", "share ending on the first answer, check2 minus original, points")):
            rows.append({"runs": ids, "format": fmt, "quantity": label, **_interval_columns(result[key])})
    return rows
