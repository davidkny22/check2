"""The three decode settings, each switched on for the length of a `with` block and off after it.

- `preallocated_cache`: the library's preallocated cache (`StaticCache`) in place of the
  growing one. The pipeline's first call over a batch, the call over tokens `0..r`, then
  starts from a preallocated cache with room for the prefix and `room` further tokens, which
  the caller sets: the rest of the prompt and the reply cap for generation, the longest
  appended text for the appended read. Later calls continue from copies of it, as they do
  from the growing one.
- `cublaslt`: `torch.backends.cuda.preferred_blas_library("cublaslt")`.
- `cudnn_attention`: the cuDNN attention backend, as the only one scaled dot-product
  attention may use.
"""

from __future__ import annotations

import contextlib

import torch
from transformers import StaticCache

from check2.evaluation.scoring import APPENDED_READ_SUFFIX
from check2.mechanism.generate import Pipeline, Reply

SETTINGS = ("preallocated_cache", "cublaslt", "cudnn_attention")


class Room:
    """The number of tokens a preallocated cache must hold after the call over `0..r`."""

    def __init__(self, tokens: int) -> None:
        self.tokens = tokens

    def for_generation(self, batch: list[dict], cap: int) -> None:
        self.tokens = max(e["n_prompt_tokens"] - e["read_index"] - 1 for e in batch) + cap

    def for_appended_read(self, pipeline: Pipeline, batch: list[dict], replies: list[Reply]) -> None:
        texts = [e["prompt"] + r.text + APPENDED_READ_SUFFIX for e, r in zip(batch, replies)]
        encoded = pipeline.tokenizer(texts)["input_ids"]
        self.tokens = max(len(ids) - e["read_index"] - 1 for e, ids in zip(batch, encoded))


@contextlib.contextmanager
def _preallocated_cache(pipeline: Pipeline, room: Room):
    run = pipeline._run

    def preallocating(ids, attention, read_column, cache, *args, **kwargs):
        if cache is None and kwargs.get("use_cache", True):
            cache = StaticCache(config=pipeline.model.config, max_cache_len=ids.shape[1] + room.tokens)
        return run(ids, attention, read_column, cache, *args, **kwargs)

    pipeline._run = preallocating
    try:
        yield
    finally:
        del pipeline._run


@contextlib.contextmanager
def _cublaslt():
    before = torch.backends.cuda.preferred_blas_library()
    torch.backends.cuda.preferred_blas_library("cublaslt")
    try:
        yield
    finally:
        torch.backends.cuda.preferred_blas_library(before)


@contextlib.contextmanager
def setting_on(setting: str, pipeline: Pipeline, batch: list[dict], cap: int):
    """Switch the setting on; yields the `Room` a preallocated cache is sized by."""
    room = Room(0)
    room.for_generation(batch, cap)
    if setting == "preallocated_cache":
        context = _preallocated_cache(pipeline, room)
    elif setting == "cublaslt":
        context = _cublaslt()
    elif setting == "cudnn_attention":
        from torch.nn.attention import SDPBackend, sdpa_kernel

        context = sdpa_kernel([SDPBackend.CUDNN_ATTENTION])
    else:
        raise ValueError(f"the setting must be one of {SETTINGS}, not {setting!r}")
    with context:
        yield room
