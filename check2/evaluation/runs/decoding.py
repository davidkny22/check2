"""The decode settings a run's config switches on for each batch: the preallocated cache and the cuDNN attention backend.

Each is switched on through `check2.mechanism.timing.settings.setting_on`, the function the
timing pass and the gates switch it on through, for every call a batch makes: the call over
tokens `0..r`, generation and the appended read. A preallocated cache is sized for the
batch's prompt tokens after the read and its format's reply cap. The appended read's texts
are longer than that by the appended suffix, so under the preallocated cache the appended
read continues from a call over `0..r` of its own, with a cache sized for those texts, as
the timing pass reads its final answers. `blas: cublaslt` is a setting of the whole process,
made where the runner loads the model.
"""

from __future__ import annotations

import contextlib

from check2.config.schema import Config
from check2.mechanism.generate import Pipeline, Prefilled, Reply
from check2.mechanism.timing import settings as decode_settings


def per_batch_settings(config: Config) -> list[str]:
    """The names, as `setting_on` takes them, of the decode settings the config switches on for each batch."""
    names = []
    if config.generation_cache == "preallocated":
        names.append("preallocated_cache")
    if config.attention_backend == "cudnn":
        names.append("cudnn_attention")
    return names


@contextlib.contextmanager
def settings_on(config: Config, pipeline: Pipeline, batch: list[dict], cap: int):
    """The config's per-batch settings on; yields the preallocated cache's `Room`, or None with the growing cache."""
    room = None
    with contextlib.ExitStack() as stack:
        for name in per_batch_settings(config):
            entered = stack.enter_context(decode_settings.setting_on(name, pipeline, batch, cap))
            if name == "preallocated_cache":
                room = entered
        yield room


def prefill_for_appended_read(pipeline: Pipeline, batch: list[dict], replies: list[Reply],
                              room: "decode_settings.Room") -> Prefilled:
    """The call over `0..r` again, its preallocated cache sized for the appended texts of these replies."""
    room.for_appended_read(pipeline, batch, replies)
    return pipeline.prefill(batch)
