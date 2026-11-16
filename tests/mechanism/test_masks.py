"""The mask over a forward call's tokens, checked against hand-built cases."""

from __future__ import annotations

import torch

from check2.mechanism.masks import after_read_mask, read_columns


def test_a_call_over_the_prefix_is_all_zero():
    # Two rows left-padded so that both read tokens sit in the last column, 4.
    attention = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    read_column = read_columns(torch.tensor([2, 0]), torch.tensor([2, 4]))
    assert read_column.tolist() == [4, 4]
    assert after_read_mask(attention, read_column, 5).sum() == 0


def test_the_tokens_after_the_read_are_one_and_padding_is_zero():
    attention = torch.tensor([[0, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])
    read_column = torch.tensor([2, 2])
    mask = after_read_mask(attention, read_column, 6)
    assert mask.tolist() == [[0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 1]]


def test_a_continuation_call_covers_only_its_own_columns():
    attention = torch.tensor([[0, 1, 1, 1, 1, 1, 0]])
    mask = after_read_mask(attention, torch.tensor([2]), 4)
    assert mask.tolist() == [[1, 1, 1, 0]]


def test_cached_decoding_gives_one():
    attention = torch.ones((3, 9), dtype=torch.long)
    attention[0, :2] = 0
    mask = after_read_mask(attention, torch.tensor([5, 5, 5]), 1)
    assert mask.tolist() == [[1.0], [1.0], [1.0]]
