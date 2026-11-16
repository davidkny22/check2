"""The wide signal's formula, and writers that take it."""

from __future__ import annotations

import math

import pytest
import torch

from check2.mechanism import wide_signal
from check2.mechanism.write import Writers


def _n_by_hand(row: list[float]) -> list[float]:
    mean = sum(row) / len(row)
    var = sum((x - mean) ** 2 for x in row) / len(row)
    return [(x - mean) / math.sqrt(var + 1e-5) for x in row]


def test_the_state_is_normalised_over_each_rows_own_values():
    rows = [[1.0, 2.0, 3.0, 6.0], [100.0, -50.0, 0.5, 7.0]]
    got = wide_signal.normalise(torch.tensor(rows, dtype=torch.bfloat16).to(torch.float32))
    for b, row in enumerate(rows):
        assert torch.allclose(got[b], torch.tensor(_n_by_hand(row)), rtol=1e-5, atol=1e-6)
    # A row normalised alone equals the same row normalised in a batch.
    assert torch.equal(wide_signal.normalise(torch.tensor(rows[1:])), got[1:])


def test_a_bf16_state_is_normalised_in_float32():
    state = torch.randn(2, 16).to(torch.bfloat16)
    got = wide_signal.normalise(state)
    assert got.dtype == torch.float32
    assert torch.allclose(got, wide_signal.normalise(state.to(torch.float32)))


def test_s_is_p_beside_n():
    p = torch.tensor([[0.2, 0.5, 0.3], [1.0, 0.0, 0.0]])
    state = torch.tensor([[1.0, 2.0, 3.0, 6.0], [4.0, 4.5, -1.0, 0.0]])
    s = wide_signal.signal(p, state)
    assert s.shape == (2, 7) and s.dtype == torch.float32
    for b in range(2):
        expected = p[b].tolist() + _n_by_hand(state[b].tolist())
        assert torch.allclose(s[b], torch.tensor(expected), rtol=1e-5, atol=1e-6)
    assert wide_signal.size(2560) == 2563


def test_layer_18_is_found_at_its_position_in_the_kept_layers():
    assert wide_signal.state_position([14, 15, 18, 19, 22, 23]) == 2
    assert wide_signal.state_position([18]) == 0
    assert wide_signal.state_position([14, 18, 22]) == 1
    with pytest.raises(wide_signal.WideSignalError):
        wide_signal.state_position([15, 19, 23])


def test_mismatched_rows_are_refused():
    with pytest.raises(wide_signal.WideSignalError):
        wide_signal.signal(torch.zeros(2, 3), torch.zeros(3, 8))
    with pytest.raises(wide_signal.WideSignalError):
        wide_signal.signal(torch.zeros(2, 4), torch.zeros(2, 8))


def test_wide_writers_take_s_and_start_at_zero():
    writers = Writers((3, 6), hidden_size=8, input_size=wide_signal.size(8), state_layer=4)
    assert writers.wide and writers.first["3"].in_features == 11
    vectors = writers(torch.randn(4, 11))
    assert all(bool((v == 0).all()) and v.shape == (4, 8) for v in vectors.values())
    with pytest.raises(ValueError):
        writers(torch.randn(4, 3))
    assert not Writers((3,), hidden_size=8).wide


def test_wide_writers_without_a_state_layer_are_refused():
    with pytest.raises(ValueError):
        Writers((3,), hidden_size=8, input_size=wide_signal.size(8))
