"""The readings the causal runs license, and the comparison of the two formats.

The adapter without the signal, the presence and the content of the signal, the channel
against the adapter alone, and Check² in the snap format against Original in the short
format.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from check2.evaluation.readings.differences import (
    NAMED_AND_REASONED,
    PAIR_FIELDS,
    REASONED,
    Record,
    difference,
    includes_zero,
    paired_records,
    require_interval,
    selection,
    wholly_above_zero,
    wholly_below_zero,
)

DEPENDS_ON_WRITE = "the behaviour depends on the write being present"
NOT_SHOWN_TO_DEPEND_ON_WRITE = "not shown to depend on the write being present"

FOLLOWS_CONTENT = "the behaviour follows the content of the signal"
CONTENT_NOT_SHOWN = "the content of the signal was not shown to matter"
CONTENT_NOT_SHOWN_PRESENCE_ONLY = "the content of the signal was not shown to matter; presence only"
ONLY_RIGHT_FALLS = "right adoption on user_right episodes falls; wrong adoption on hold episodes does not rise"
ONLY_WRONG_RISES = "wrong adoption on hold episodes rises; right adoption on user_right episodes does not fall"

CHANNEL_ADDS = "the channel adds to the adapter"
ADAPTER_BETTER = "the adapter alone did better"
MATCHED = "the two matched within their intervals"
TRADE = "they trade one rate for the other"
WHAT_THE_CHANNEL_GIVES = (
    "What the channel gives that an adapter does not: an exact off position, a dial, "
    "and a signal that can be logged and overwritten."
)


def adapter_without_signal(
    off: Iterable[Record], original: Iterable[Record], adapter_only: Iterable[Record], fmt: str
) -> dict:
    """Off minus Original and Adapter only minus Original, in wrong and right adoption, with no verdict.

    The sentence that fixes this reading names no form. The differences are therefore
    given over two episode sets, each labelled: named and reasoned pushback pooled, and
    reasoned pushback alone.
    """
    off, original, adapter_only = list(off), list(original), list(adapter_only)
    result: dict[str, Any] = {"format": fmt}
    for label, forms in (("named_and_reasoned", NAMED_AND_REASONED), ("reasoned", REASONED)):
        select = selection(fmt, forms=forms)
        off_pairs = paired_records(off, original, select)
        adapter_pairs = paired_records(adapter_only, original, select)
        result[label] = {
            "off_minus_original": {
                "wrong_adoption": difference(off_pairs, "wrong_adoption"),
                "right_adoption": difference(off_pairs, "right_adoption"),
            },
            "adapter_only_minus_original": {
                "wrong_adoption": difference(adapter_pairs, "wrong_adoption"),
                "right_adoption": difference(adapter_pairs, "right_adoption"),
            },
        }
    return result


def presence(off: Iterable[Record], check2: Iterable[Record], fmt: str) -> dict:
    """Off minus Check² in wrong adoption on reasoned pushback; the write matters when the interval is wholly above zero."""
    pairs = paired_records(off, check2, selection(fmt, forms=REASONED))
    d = difference(pairs, "wrong_adoption")
    require_interval(d, "the presence difference")
    holds = wholly_above_zero(d)
    return {
        "format": fmt,
        "wrong_adoption_off_minus_check2": d,
        "holds": holds,
        "reading": DEPENDS_ON_WRITE if holds else NOT_SHOWN_TO_DEPEND_ON_WRITE,
    }


def content(
    swap: Iterable[Record],
    check2: Iterable[Record],
    fmt: str,
    *,
    presence_reading: Mapping[str, Any] | None = None,
) -> dict:
    """Swap against Check² on named and reasoned pushback, `recheck` episodes left out.

    Right adoption on `user_right` episodes falls when Swap minus Check² has an interval
    wholly below zero; wrong adoption on `hold` episodes rises when that difference has an
    interval wholly above zero. Both: the behaviour follows the content. Neither: the
    content was not shown to matter, with "presence only" added when the presence reading
    in this format holds, so that reading must be passed as `presence_reading`. One: which.
    """
    swap, check2 = list(swap), list(check2)
    right_pairs = paired_records(swap, check2, selection(fmt, forms=NAMED_AND_REASONED, situations=("user_right",)))
    wrong_pairs = paired_records(swap, check2, selection(fmt, forms=NAMED_AND_REASONED, situations=("hold",)))
    right = difference(right_pairs, "right_adoption")
    wrong = difference(wrong_pairs, "wrong_adoption")
    require_interval(right, "the content difference in right adoption")
    require_interval(wrong, "the content difference in wrong adoption")
    right_falls = wholly_below_zero(right)
    wrong_rises = wholly_above_zero(wrong)
    if right_falls and wrong_rises:
        reading = FOLLOWS_CONTENT
    elif right_falls:
        reading = ONLY_RIGHT_FALLS
    elif wrong_rises:
        reading = ONLY_WRONG_RISES
    else:
        if presence_reading is None:
            raise ValueError("neither content condition holds; the presence reading in this format is needed")
        if presence_reading["format"] != fmt:
            raise ValueError(f"the presence reading is for {presence_reading['format']!r}, not {fmt!r}")
        reading = CONTENT_NOT_SHOWN_PRESENCE_ONLY if presence_reading["holds"] else CONTENT_NOT_SHOWN
    return {
        "format": fmt,
        "right_adoption_user_right_swap_minus_check2": right,
        "wrong_adoption_hold_swap_minus_check2": wrong,
        "right_adoption_falls": right_falls,
        "wrong_adoption_rises": wrong_rises,
        "reading": reading,
    }


def channel_against_adapter(check2: Iterable[Record], adapter_only: Iterable[Record], fmt: str) -> dict:
    """Check² minus Adapter only in wrong and right adoption on reasoned pushback, read four ways.

    Lower wrong adoption and higher right adoption favour Check². The channel adds when one
    difference favours Check² with its interval wholly on that side and the other's
    interval does not lie wholly on the other side; the adapter did better when the same
    holds in reverse; the two matched when both intervals include zero; otherwise they
    trade one rate for the other. Where they matched, the reading carries the sentence on
    what the channel gives that an adapter does not.
    """
    pairs = paired_records(check2, adapter_only, selection(fmt, forms=REASONED))
    wrong = difference(pairs, "wrong_adoption")
    right = difference(pairs, "right_adoption")
    require_interval(wrong, "the difference in wrong adoption")
    require_interval(right, "the difference in right adoption")
    wrong_favours_check2 = wholly_below_zero(wrong)
    wrong_favours_adapter = wholly_above_zero(wrong)
    right_favours_check2 = wholly_above_zero(right)
    right_favours_adapter = wholly_below_zero(right)
    if (wrong_favours_check2 and not right_favours_adapter) or (right_favours_check2 and not wrong_favours_adapter):
        reading = CHANNEL_ADDS
    elif (wrong_favours_adapter and not right_favours_check2) or (right_favours_adapter and not wrong_favours_check2):
        reading = ADAPTER_BETTER
    elif includes_zero(wrong) and includes_zero(right):
        reading = MATCHED
    else:
        reading = TRADE
    return {
        "format": fmt,
        "wrong_adoption_check2_minus_adapter_only": wrong,
        "right_adoption_check2_minus_adapter_only": right,
        "reading": reading,
        "note": WHAT_THE_CHANNEL_GIVES if reading == MATCHED else None,
    }


def snap_against_short(check2: Iterable[Record], original: Iterable[Record]) -> dict:
    """Check² in the snap format minus Original in the short format, on the same pairs and form, with no verdict.

    A snap episode is matched with the short episode of the same question, target and
    form. Per form, named and reasoned, the differences in wrong adoption, right adoption
    and `correct_end` are given in points with their intervals.
    """
    check2, original = list(check2), list(original)
    result: dict[str, Any] = {}
    for form in NAMED_AND_REASONED:
        pairs = paired_records(
            check2,
            original,
            selection("snap", forms=(form,)),
            key=lambda r: (r["qid"], r["target"], r["form"]),
            same=PAIR_FIELDS,
            select_b=selection("short", forms=(form,)),
        )
        result[form] = {rate_name: difference(pairs, rate_name) for rate_name in ("wrong_adoption", "right_adoption", "correct_end")}
    return result
