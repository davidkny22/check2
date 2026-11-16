"""The faults the reviews of scoring, evaluation and the readings committed, each planted in the running code.

A fault is one exact replacement of text in one function of a product file, the change the
reviewer's patch makes. Planting it compiles the changed file and gives the function the
changed code while the context is open, so that every caller runs the fault, whatever name
it imported the function under. The file on disk is never changed.

A test that a fault must turn red runs its checks under `planted(name)` and expects an
assertion to fail; `fails_under` does exactly that.
"""

from __future__ import annotations

import contextlib
import importlib
import types
from collections.abc import Callable
from pathlib import Path

import pytest

# name: (module, function, text in the function, the text the fault puts in its place)
FAULTS = {
    "fault-channel-adapter-better-through-wrong-only": (
        "check2.evaluation.readings.causal", "channel_against_adapter",
        "    elif (wrong_favours_adapter and not right_favours_check2) or (right_favours_adapter and not "
        "wrong_favours_check2):\n",
        "    elif wrong_favours_adapter and not right_favours_check2:\n"),
    "fault-parts-differ-only-above-zero": (
        "check2.evaluation.readings.passing", "two_parts",
        "differed = wholly_above_zero(between) or wholly_below_zero(between)",
        "differed = wholly_above_zero(between)"),
    "fault-pass-bar-limit-exclusive": (
        "check2.evaluation.readings.passing", "right_adoption_drop_verdict",
        'if drop["high"] <= RIGHT_ADOPTION_DROP_LIMIT:', 'if drop["high"] < RIGHT_ADOPTION_DROP_LIMIT:'),
    "pass-bar-upper-end-at-ten": (
        "check2.evaluation.readings.passing", "right_adoption_drop_verdict",
        'if drop["high"] <= RIGHT_ADOPTION_DROP_LIMIT:', 'if drop["high"] < RIGHT_ADOPTION_DROP_LIMIT:'),
    "pass-bar-lower-end-at-ten": (
        "check2.evaluation.readings.passing", "right_adoption_drop_verdict",
        'if drop["low"] > RIGHT_ADOPTION_DROP_LIMIT:', 'if drop["low"] >= RIGHT_ADOPTION_DROP_LIMIT:'),
    "perfect-signal-upper-end-at-ten": (
        "check2.gates.perfect_signal", "evaluate",
        'elif wrong["high"] <= 0 and drop["high"] <= RIGHT_ADOPTION_DROP_LIMIT:',
        'elif wrong["high"] <= 0 and drop["high"] < RIGHT_ADOPTION_DROP_LIMIT:'),
    "swap-recheck-kept-at-scoring": (
        "check2.evaluation.tables.rows", "_scored",
        '    if run.get("base", run["role"]) == "swap":\n        return [r for r in records if r["situation"] != "recheck"]\n',
        ""),
    "marked-folder-still-read": (
        "check2.evaluation.tables.folders", "_folders",
        '            marked.append({"folder": folder.name, "reason": reason})\n            continue\n',
        '            marked.append({"folder": folder.name, "reason": reason})\n'),
    "pair-on-unmatched-episodes": (
        "check2.evaluation.tables.named_pairs", "on_common_episodes",
        '    common = {r["episode_id"] for r in _scored(a)} & {r["episode_id"] for r in _scored(b)}\n',
        "    common = held_a\n"),
    "system-prompt-suffix-dropped": (
        "check2.evaluation.runs.episodes", "with_system_suffix",
        "    from check2.data.pool import complying, load_pool\n",
        "    return episodes\n    from check2.data.pool import complying, load_pool\n"),
    "steering-strength-ignored": (
        "check2.evaluation.steering", "install_steering",
        "added = mask.unsqueeze(-1) * strength * vector", "added = mask.unsqueeze(-1) * vector"),
    "appended-read-at-full-strength": (
        "check2.evaluation.runs.execution", "run_batch",
        "appended = pipeline.appended_read(prefilled, used, replies, alpha, disabled)",
        "appended = pipeline.appended_read(prefilled, used, replies, 1.0, disabled)"),
    "pass-bar-first-condition-on-estimate": (
        "check2.evaluation.readings.passing", "pass_bar",
        'first_holds = wrong_drop["low"] > 0', 'first_holds = wrong_drop["estimate"] > 0'),
    "pass-bar-quantities-on-every-form": (
        "check2.evaluation.readings.passing", "pass_bar_quantities",
        "pairs = paired_records(original, run, selection(fmt, forms=REASONED))",
        "pairs = paired_records(original, run, selection(fmt))"),
    "presence-on-named-and-reasoned": (
        "check2.evaluation.readings.causal", "presence",
        "pairs = paired_records(off, check2, selection(fmt, forms=REASONED))",
        "pairs = paired_records(off, check2, selection(fmt, forms=NAMED_AND_REASONED))"),
    "content-on-reasoned-only": (
        "check2.evaluation.readings.causal", "content",
        'wrong_pairs = paired_records(swap, check2, selection(fmt, forms=NAMED_AND_REASONED, situations=("hold",)))',
        'wrong_pairs = paired_records(swap, check2, selection(fmt, forms=REASONED, situations=("hold",)))'),
    "channel-on-named-and-reasoned": (
        "check2.evaluation.readings.causal", "channel_against_adapter",
        "pairs = paired_records(check2, adapter_only, selection(fmt, forms=REASONED))",
        "pairs = paired_records(check2, adapter_only, selection(fmt, forms=NAMED_AND_REASONED))"),
}


def _changed_code(module: types.ModuleType, function: str, old: str, new: str) -> types.CodeType:
    path = Path(module.__file__)
    source = path.read_text(encoding="utf-8")
    if source.count(old) != 1:
        raise AssertionError(f"the fault's text occurs {source.count(old)} times in {path.name}, not once")
    compiled = compile(source.replace(old, new), str(path), "exec")
    found = [c for c in compiled.co_consts if isinstance(c, types.CodeType) and c.co_name == function]
    if len(found) != 1:
        raise AssertionError(f"{path.name} defines {len(found)} functions named {function}")
    return found[0]


def _identity(code: types.CodeType) -> tuple:
    return code.co_code, code.co_consts, code.co_names


@contextlib.contextmanager
def planted(name: str):
    """While open, the named function runs with the fault in it."""
    module_name, function, old, new = FAULTS[name]
    module = importlib.import_module(module_name)
    target = getattr(module, function)
    original = target.__code__
    changed = _changed_code(module, function, old, new)
    if _identity(changed) == _identity(original):
        raise AssertionError(f"the fault {name} does not change {function}")
    target.__code__ = changed
    try:
        yield
    finally:
        target.__code__ = original


def fails_under(name: str, check: Callable[[], None]) -> None:
    """The check passes as the code stands; with the fault planted, one of its assertions fails."""
    with planted(name):
        with pytest.raises(AssertionError):
            check()
