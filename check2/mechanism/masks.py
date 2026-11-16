"""The mask that says, per batch row, which tokens of a forward call lie after the read token.

Prompts are left-padded so that every row's read token sits in one column of the padded
batch. A row's read column is the column of its own first token plus its read index. For a
call over the last `length` columns of the sequence so far, the mask is 1 where a column
lies after the row's read column and holds a real token, and 0 elsewhere. In a call over
tokens `0..r` it is therefore all 0, and in cached decoding, one token per call, it is 1.
"""

from __future__ import annotations

import torch


def read_columns(first_token_column: torch.Tensor, read_index: torch.Tensor) -> torch.Tensor:
    """The padded-batch column of each row's read token, counted from that row's first token."""
    return first_token_column + read_index


def after_read_mask(attention_mask: torch.Tensor, read_column: torch.Tensor, length: int) -> torch.Tensor:
    """The float mask `[B, length]` for a call over the last `length` columns of `attention_mask`.

    `attention_mask` is `[B, C]`, covering every column up to the end of the call, with 1 at
    real tokens. `read_column` is `[B]`.
    """
    total = attention_mask.shape[1]
    columns = torch.arange(total - length, total, device=attention_mask.device)
    after = columns.unsqueeze(0) > read_column.to(attention_mask.device).unsqueeze(1)
    real = attention_mask[:, total - length:].bool()
    return (after & real).to(torch.float32)
