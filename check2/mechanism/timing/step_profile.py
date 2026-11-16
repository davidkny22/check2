"""Where one decoding step's time goes: matrix multiplies, the two kinds of layer, hooks, cache handling.

The step is profiled from the start of its forward call to the start of the next one, so
it holds the model call, the logits and the choice of token, and the mask and control
state set for the next call. Named ranges mark, while the profile runs: each full-attention
mixer (`self_attn`) and each Gated DeltaNet mixer (`linear_attn`); every forward hook
installed on the model; the cache's reads and writes; and the pipeline's mask and control
state. The matrix multiplies are the operators that launch them. Ranges can nest (a cache
write runs inside a mixer), so their shares do not add up to one.
"""

from __future__ import annotations

import contextlib
import functools
import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import cache_utils

from check2.mechanism import masks
from check2.mechanism.generate import Pipeline
from check2.mechanism.model_parts import decoder_layers

FULL_ATTENTION, DELTANET, HOOKS, CACHE, CONTROL = (
    "full_attention_layers", "deltanet_layers", "hooks", "cache_handling", "control_state")
MATRIX_MULTIPLY_OPERATORS = ("aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm", "aten::_scaled_mm")
CACHE_METHODS = ("update", "update_conv_state", "update_recurrent_state", "get_conv_state", "get_recurrent_state")
LISTED_OPERATORS = 15


def _ranged(function, name: str):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        with record_function(name):
            return function(*args, **kwargs)
    return wrapped


@contextlib.contextmanager
def _ranges(pipeline: Pipeline):
    """Mark the named ranges for as long as the block runs, and take every mark away after it."""
    undo = []
    for module in pipeline.model.modules():
        for table in (module._forward_hooks, module._forward_pre_hooks):
            for key, hook in list(table.items()):
                table[key] = _ranged(hook, HOOKS)
                undo.append(lambda table=table, key=key, hook=hook: table.__setitem__(key, hook))
    for layer in decoder_layers(pipeline.model):
        mixer, name = (layer.self_attn, FULL_ATTENTION) if hasattr(layer, "self_attn") else (layer.linear_attn,
                                                                                            DELTANET)
        open_ranges = []

        def enter(module, args, name=name, open_ranges=open_ranges):
            context = record_function(name)
            context.__enter__()
            open_ranges.append(context)

        def leave(module, args, output, open_ranges=open_ranges):
            open_ranges.pop().__exit__(None, None, None)
        undo += [mixer.register_forward_pre_hook(enter).remove, mixer.register_forward_hook(leave).remove]
    for method in CACHE_METHODS:
        original = getattr(cache_utils.Cache, method)
        setattr(cache_utils.Cache, method, _ranged(original, CACHE))
        undo.append(lambda method=method, original=original: setattr(cache_utils.Cache, method, original))
    original_mask = masks.after_read_mask
    masks.after_read_mask = _ranged(original_mask, CONTROL)
    undo.append(lambda: setattr(masks, "after_read_mask", original_mask))
    pipeline.control.set = _ranged(pipeline.control.set, CONTROL)
    undo.append(lambda: delattr(pipeline.control, "set"))
    try:
        yield
    finally:
        for step in reversed(undo):
            step()


def _times(event) -> tuple[float, float, float, float]:
    """CPU total, CPU self, device total and device self time of an averaged event, in microseconds."""
    device_total = getattr(event, "device_time_total", getattr(event, "cuda_time_total", 0.0))
    device_self = getattr(event, "self_device_time_total", getattr(event, "self_cuda_time_total", 0.0))
    return float(event.cpu_time_total), float(event.self_cpu_time_total), float(device_total), float(device_self)


def summarise(events, wall_seconds: float) -> dict:
    """Each part's CPU and device time, and its share of the step's wall time."""
    averaged = {e.key: e for e in events.key_averages()}
    wall_us = wall_seconds * 1e6
    device_us = sum(_times(e)[3] for e in averaged.values())
    parts = {}
    for name in (FULL_ATTENTION, DELTANET, HOOKS, CACHE, CONTROL):
        cpu_total, _, device_total, _ = _times(averaged[name]) if name in averaged else (0.0, 0.0, 0.0, 0.0)
        parts[name] = {"cpu_us": cpu_total, "device_us": device_total, "calls": averaged[name].count
                       if name in averaged else 0}
    multiplies = [_times(averaged[op]) for op in MATRIX_MULTIPLY_OPERATORS if op in averaged]
    parts["matrix_multiplies"] = {"cpu_us": sum(t[1] for t in multiplies), "device_us": sum(t[3] for t in multiplies),
                                  "calls": sum(averaged[op].count for op in MATRIX_MULTIPLY_OPERATORS
                                               if op in averaged)}
    for part in parts.values():
        part["cpu_share_of_wall"] = part["cpu_us"] / wall_us
        part["device_share_of_wall"] = part["device_us"] / wall_us
        part["device_share_of_device_time"] = part["device_us"] / device_us if device_us > 0 else 0.0
    operators = [e for e in averaged.values() if e.key.startswith("aten::")]
    return {"wall_seconds": wall_seconds, "device_kernel_us": device_us, "parts": parts,
            "top_operators_by_self_cpu": [(e.key, _times(e)[1], e.count) for e in
                                          sorted(operators, key=lambda e: -_times(e)[1])[:LISTED_OPERATORS]],
            "top_operators_by_self_device": [(e.key, _times(e)[3], e.count) for e in
                                             sorted(operators, key=lambda e: -_times(e)[3])[:LISTED_OPERATORS]]}


def profile_decoding_step(pipeline: Pipeline, batch: list[dict], step: int, cap: int) -> dict:
    """Generate the batch and profile its `step`-th one-token call, counting from 1, to the start of the next call.

    The pipeline's first call covers tokens `0..r` and its second the rest of the prompt;
    the one-token calls follow.
    """
    if not 1 <= step < cap:
        raise ValueError(f"the step must lie inside the reply cap of {cap}, not {step}")
    target = 1 + step
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
    profiler = profile(activities=activities)
    state = {"calls": 0, "started": None, "wall": None}

    def synchronise():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def stop():
        synchronise()
        state["wall"] = time.perf_counter() - state["started"]
        profiler.stop()

    def at_each_call(module, args, kwargs):
        if state["calls"] == target:
            synchronise()
            profiler.start()
            state["started"] = time.perf_counter()
        elif state["calls"] == target + 1:
            stop()
        state["calls"] += 1

    with _ranges(pipeline), torch.inference_mode():
        # Registered after the ranges, so that this counter is not itself counted as a hook.
        handle = pipeline.body.register_forward_pre_hook(at_each_call, with_kwargs=True)
        try:
            replies = pipeline.generate(pipeline.prefill(batch), None, cap)
            if state["started"] is not None and state["wall"] is None:
                stop()
        finally:
            handle.remove()
    if state["started"] is None:
        raise RuntimeError(f"every reply ended before decoding step {step}; there was no such step to profile")
    return {"step": step, "batch": len(batch), "rows_still_running": sum(len(r.token_ids) > step for r in replies),
            **summarise(profiler, state["wall"])}
