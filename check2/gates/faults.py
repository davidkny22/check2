"""Planted faults that show the gates on the mechanism can fail.

Each is a context manager that patches the running code while it is open. They are used
only by a gate's `--plant` switch and by the fast tests, never by a run.
"""

from __future__ import annotations

import contextlib

import torch

from check2.mechanism import masks
from check2.mechanism.write import Writers

PLANTED_BIAS = 1e-2


@contextlib.contextmanager
def _patched(owner, name: str, replacement):
    original = getattr(owner, name)
    setattr(owner, name, replacement(original))
    try:
        yield
    finally:
        setattr(owner, name, original)


def writer_bias():
    """Identity: every writer's last-layer bias starts at 1e-2 instead of zero."""
    def replacement(original):
        def reset_parameters(self):
            original(self)
            with torch.no_grad():
                for j in self.layers:
                    self.second[str(j)].bias.fill_(PLANTED_BIAS)
        return reset_parameters
    return _patched(Writers, "reset_parameters", replacement)


def mask_includes_read():
    """Gating: the mask shifted one token earlier, so that it includes the read token."""
    def replacement(original):
        return lambda first_token_column, read_index: original(first_token_column, read_index) - 1
    return _patched(masks, "read_columns", replacement)


def mask_from_column_zero():
    """Padding invariance: the read column counted from column 0 of the padded batch, not the row's first token."""
    def replacement(original):
        return lambda first_token_column, read_index: read_index.clone()
    return _patched(masks, "read_columns", replacement)


def additions_skipped_in_decoding():
    """Cached equals uncached: the additions skipped in a call over one token."""
    def replacement(original):
        def after_read_mask(attention_mask, read_column, length):
            mask = original(attention_mask, read_column, length)
            return torch.zeros_like(mask) if length == 1 else mask
        return after_read_mask
    return _patched(masks, "after_read_mask", replacement)


def recurrent_state_zeroed():
    """Split training equals plain: each pass's copy of the prefix cache has its recurrent state set to zero."""
    from check2.training import forms

    def replacement(original):
        def continuation_cache(prefix_cache):
            cache = original(prefix_cache)
            for layer in cache.layers:
                states = getattr(layer, "recurrent_states", None) or {}
                for index, state in states.items():
                    if state is not None:
                        states[index] = torch.zeros_like(state)
            return cache
        return continuation_cache
    return _patched(forms, "continuation_cache", replacement)


FAULTS = {
    "identity": writer_bias,
    "gating": mask_includes_read,
    "padding_invariance": mask_from_column_zero,
    "cached_equals_uncached": additions_skipped_in_decoding,
    "split_training": recurrent_state_zeroed,
}


def planted(gate: str, on: bool):
    """The gate's fault when `on`, else a context that changes nothing."""
    return FAULTS[gate]() if on else contextlib.nullcontext()
