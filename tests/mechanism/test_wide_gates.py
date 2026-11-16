"""The wide-signal gate, and the mechanism gates run with wide writers, on the tiny model."""

from __future__ import annotations

import pytest

from check2.gates import cached_equals_uncached, gating, identity, padding_invariance, write_formula
from check2.gates import wide_signal as wide_gate

WIDE = "probabilities_and_state"


def _fewer(episodes):
    return {"snap": episodes["snap"], "short": episodes["short"][:2]}


def test_the_wide_signal_gate_is_green(bench, episodes):
    values = wide_gate.check(bench, episodes, "tiny")
    assert values["green"], values
    for fmt, result in values["formats"].items():
        assert result["signals_checked"] == 4 and result["max_relative_difference"] <= wide_gate.TOLERANCE, fmt


@pytest.mark.parametrize("plant", wide_gate.PLANTS)
def test_the_wide_signal_gate_is_red_under_each_planted_fault(bench, episodes, plant):
    values = wide_gate.check(bench, episodes, "tiny", plant=plant)
    assert not values["green"]
    assert all(r["max_relative_difference"] > 100 * wide_gate.TOLERANCE for r in values["formats"].values())


def test_identity_with_wide_writers(bench, episodes):
    assert identity.check(bench, episodes, writer_input=WIDE)["green"]
    assert not identity.check(bench, episodes, plant=True, writer_input=WIDE)["green"]


def test_gating_with_wide_writers(bench, episodes):
    assert gating.check(bench, episodes, writer_input=WIDE)["green"]
    assert not gating.check(bench, episodes, plant=True, writer_input=WIDE)["green"]


def test_the_write_formula_with_wide_writers(bench, episodes):
    values = write_formula.check(bench, episodes, writer_input=WIDE)
    assert values["green"], values


def test_padding_invariance_with_wide_writers(bench, episodes):
    assert padding_invariance.check(bench, _fewer(episodes), writer_input=WIDE)["green"]
    assert not padding_invariance.check(bench, _fewer(episodes), plant=True, writer_input=WIDE)["green"]


def test_cached_equals_uncached_with_wide_writers(bench, episodes):
    assert cached_equals_uncached.check(bench, _fewer(episodes), writer_input=WIDE)["green"]
    assert not cached_equals_uncached.check(bench, _fewer(episodes), plant=True, writer_input=WIDE)["green"]
