"""The wide signal of each training conversation: its saved `p` beside its saved state at the read token.

`p` is the reader's cross-fitted output, read as for every other training. The state is
the output of layer 18 at the read token, from the file `check2.reader.wide_states` wrote
from the reader's cache of the original model. A conversation with no saved `p` or no
saved state is an error.
"""

from __future__ import annotations

from pathlib import Path

import torch

from check2.config.schema import WIDE_WRITER_INPUT, Config
from check2.mechanism import wide_signal
from check2.reader import wide_states
from check2.training import inputs

STATES_TRAIN = wide_states.states_file("train")


def is_wide(config: Config) -> bool:
    return config.writer.input == WIDE_WRITER_INPUT


def signals_for(config: Config, episodes: list[dict], p_file: Path, states_file: Path,
                forming: dict | None = None) -> dict[str, torch.Tensor]:
    """Each episode's `S`, `[3 + d]` in float32, from its saved `p` and its saved state."""
    p = inputs.signals_for(config, episodes, p_file, forming)
    states = wide_states.load_states(states_file)
    missing = [e["episode_id"] for e in episodes if e["episode_id"] not in states]
    if missing:
        raise inputs.InputError(f"{len(missing)} training episodes have no saved state in {states_file}, "
                                f"e.g. {missing[:3]}")
    return {e["episode_id"]: wide_signal.signal(p[e["episode_id"]][None], states[e["episode_id"]][None])[0]
            for e in episodes}
