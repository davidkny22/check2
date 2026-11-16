"""What the gates on the mechanism share: their command line, model, episodes, signal and additions.

Every such gate runs as `python -m check2.gates.<name> [--plant] --model tiny|full
[--episodes <file>] [--tail-position K] [--limit N] [--writer-input probabilities|probabilities_and_state]`.
With the wide input, the gate's writers take `S`: the same one-hot `p` beside each row's
own state at its read token, which the pipeline keeps. It takes, per format, the episodes
`random.Random(2035).sample(sorted(episode_ids), N)` draws from its file, sets `p` to the
one-hot vector of each episode's situation, writes its record through `write_record`, and
exits non-zero when red.

Random additions are scaled as in the design's hook test: each write at half the mean norm
of its layer's input, and each adapter update at the same fraction of its module's output,
both measured on the prompts of a few episodes run alone with no additions.

A gate that takes `--setting preallocated_cache|cublaslt|cudnn_attention` runs every call of
every pipeline it builds, the measurement of these norms included, with that decode setting
on, through `check2.mechanism.timing.settings.setting_on`, sized for the batch the pipeline
runs and its format's reply cap. A call that makes no cache runs as it is under the cache
setting. With no `--setting` nothing changes.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from check2.config.schema import WIDE_WRITER_INPUT, WRITER_INPUTS, Config, RANK_TO_ALPHA_LORA
from check2.data import paths
from check2.data.episodes import FORMATS, TAIL_LENGTH, check_tail_position
from check2.mechanism.control import ControlState
from check2.mechanism.generate import REPLY_CAPS, Pipeline
from check2.mechanism.loading import Layout, fast_kernel_importable, load_full, load_tiny, require_builtin_path
from check2.mechanism.model_parts import decoder_layers
from check2.mechanism.timing import settings as decode_settings
from check2.mechanism import wide_signal
from check2.mechanism.write import WRITER_HIDDEN, Writers
from check2.record import write_record

if TYPE_CHECKING:
    from check2.mechanism.adapter.hooks import HookAdapter

EPISODE_DRAW_SEED = 2035
ADDITIONS_SEED = Config.seed
ADDITION_FRACTION = 0.5
MEASURE_EPISODES = 4
RANK = Config().adapter.rank
ALPHA_LORA = RANK_TO_ALPHA_LORA[RANK]
GATING_LAYER = 3


def parser(description: str) -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=description)
    out.add_argument("--plant", action="store_true", help="apply this gate's planted fault; the gate must go red")
    out.add_argument("--model", choices=("tiny", "full"), required=True)
    out.add_argument("--episodes", type=Path, default=None, help="the episode file to draw from")
    out.add_argument("--tail-position", type=int, default=0, help="the read position, as an index into the tail")
    out.add_argument("--limit", type=int, default=None, help="take only the first N drawn episodes per format")
    out.add_argument("--writer-input", choices=WRITER_INPUTS, default=WRITER_INPUTS[0],
                     help="what the writers receive: the three probabilities, or those and the read token's state")
    return out


def writers_for(bench: "Bench", writer_input: str = WRITER_INPUTS[0]) -> Writers:
    """Writers at their starting values, taking `p` or, with the wide input, `S`."""
    hidden = bench.model.config.hidden_size
    if writer_input == WIDE_WRITER_INPUT:
        return wide_signal.wide_writers(bench.layout, hidden, WRITER_HIDDEN)
    return Writers(bench.layout.write_layers, hidden)


def add_setting_switch(parser: argparse.ArgumentParser) -> None:
    """`--setting`, for the gates a decode setting must pass before it is adopted."""
    parser.add_argument("--setting", choices=decode_settings.SETTINGS, default=None,
                        help="a decode setting, on for every call of every pipeline the gate builds")


@contextlib.contextmanager
def setting_for(setting: str | None, pipeline: Pipeline, batch: list[dict], cap: int):
    """The decode setting on for the pipeline's calls inside, sized for `batch` at reply cap `cap`; with None, nothing."""
    if setting is None:
        yield
        return
    with decode_settings.setting_on(setting, pipeline, batch, cap):
        yield


@dataclass
class Bench:
    model: torch.nn.Module
    tokenizer: object
    layout: Layout
    dtype: torch.dtype
    builtin_path: bool


def load(model: str, exact: bool) -> Bench:
    """Load the model. `exact` gates run in float32 on the built-in path; the others in the runs' bf16."""
    if exact:
        require_builtin_path()
    if model == "tiny":
        m, tok, layout = load_tiny(torch.float32)
        return Bench(m, tok, layout, torch.float32, not fast_kernel_importable())
    dtype = torch.float32 if exact else torch.bfloat16
    m, tok, layout = load_full(dtype)
    return Bench(m, tok, layout, dtype, not fast_kernel_importable())


def read_episodes(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def at_tail_position(episode: dict, tail_position: int) -> dict:
    """The episode with its read index moved to the given tail token."""
    check_tail_position(tail_position)
    return {**episode, "read_index": episode["n_prompt_tokens"] - TAIL_LENGTH + tail_position}


def draw(path: Path, count: int, tail_position: int, limit: int | None) -> dict[str, list[dict]]:
    """Per format, `random.Random(2035).sample(sorted(ids), count)`, then the first `limit` of the draw."""
    by_format: dict[str, dict[str, dict]] = {fmt: {} for fmt in FORMATS}
    for e in read_episodes(path):
        if e["format"] in by_format and e.get("situation") is not None:
            by_format[e["format"]][e["episode_id"]] = e
    out = {}
    for fmt in FORMATS:
        ids = random.Random(EPISODE_DRAW_SEED).sample(sorted(by_format[fmt]), count)
        if limit is not None:
            ids = ids[:limit]
        out[fmt] = [at_tail_position(by_format[fmt][i], tail_position) for i in ids]
    return out


def episode_file(args, default_split: str) -> Path:
    return args.episodes if args.episodes is not None else paths.EPISODES / f"{default_split}.jsonl"


def plain_pipeline(bench: Bench) -> Pipeline:
    return Pipeline(bench.model, bench.tokenizer, ControlState())


@contextlib.contextmanager
def _norm_probes(bench: Bench, writers: Writers | None, adapter: "HookAdapter | None"):
    """Record, over every token of the calls inside, each write layer's input norm and each module's norms."""
    sums: dict[str, list[float]] = {}

    def add(key: str, norms: torch.Tensor) -> None:
        total = sums.setdefault(key, [0.0, 0.0])
        total[0] += float(norms.sum())
        total[1] += norms.numel()

    handles = []
    blocks = decoder_layers(bench.model)
    for j in (writers.layers if writers is not None else ()):
        def pre(module, args, kwargs, j=j):
            hidden = args[0] if args else kwargs["hidden_states"]
            add(f"input:{j}", hidden.float().norm(dim=-1).flatten())
        handles.append(blocks[j].register_forward_pre_hook(pre, with_kwargs=True))
    for name, module in (adapter.modules if adapter is not None else ()):
        def post(module, inputs, output, name=name):
            add(f"output:{name}", output.float().norm(dim=-1).flatten())
            add(f"update:{name}", adapter.update(name, inputs[0]).norm(dim=-1).flatten())
        handles.append(module.register_forward_hook(post))
    try:
        yield sums
    finally:
        for handle in handles:
            handle.remove()


def random_additions(bench: Bench, control: ControlState, measure_on: list[dict], seed: int = ADDITIONS_SEED,
                     writers_on: bool = True, adapter_on: bool = True,
                     setting: str | None = None,
                     writer_input: str = WRITER_INPUTS[0]) -> "tuple[Writers | None, HookAdapter | None]":
    """Random non-zero writers and adapter factors, scaled to the model's own norms.

    The norms are measured with the decode setting on, where one is given. The adapter's
    hooks are not installed; the pipeline that takes it installs them. Wide writers are
    scaled on `S` made from each one-hot `p` and the measured episodes' own read states.
    """
    torch.manual_seed(seed)
    device = next(bench.model.parameters()).device
    writers = adapter = None
    if writers_on:
        writers = writers_for(bench, writer_input).to(device)
        with torch.no_grad():
            for parameter in writers.parameters():
                parameter.normal_()
    if adapter_on:
        from check2.mechanism.adapter.hooks import HookAdapter

        adapter = HookAdapter(bench.model, control, RANK, ALPHA_LORA, bench.layout.adapter_first,
                              bench.layout.adapter_last)
        with torch.no_grad():
            for name, _ in adapter.modules:
                adapter.b[name].normal_()
    pipeline = plain_pipeline(bench)
    states = []
    state_layers = [writers.state_layer] if writers is not None and writers.wide else []
    with _norm_probes(bench, writers, adapter) as sums, torch.inference_mode():
        for episode in measure_on[:MEASURE_EPISODES]:
            with setting_for(setting, pipeline, [episode], REPLY_CAPS[episode["format"]]):
                with layer_outputs(bench.model, state_layers) as seen:
                    prefilled = pipeline.prefill([episode])
                pipeline.prompt_pass(prefilled, None)
            states += [seen[j][0][:, -1] for j in state_layers]
    pipeline.close()
    mean = {key: total / count for key, (total, count) in sums.items()}
    with torch.no_grad():
        if writers is not None:
            every_situation = torch.eye(3, device=device)
            if writers.wide:
                state = torch.cat(states).repeat_interleave(3, 0)
                every_situation = wide_signal.signal(every_situation.repeat(len(states), 1), state)
            vectors = writers(every_situation)
            for j in writers.layers:
                factor = ADDITION_FRACTION * mean[f"input:{j}"] / float(vectors[j].norm(dim=-1).mean())
                writers.second[str(j)].weight.mul_(factor)
                writers.second[str(j)].bias.mul_(factor)
        if adapter is not None:
            for name, _ in adapter.modules:
                factor = ADDITION_FRACTION * mean[f"output:{name}"] / mean[f"update:{name}"]
                adapter.b[name].mul_(factor)
    return writers, adapter


@contextlib.contextmanager
def layer_outputs(model: torch.nn.Module, layers=None):
    """Collect each listed decoder layer's output, one tensor per forward call, in call order."""
    blocks = decoder_layers(model)
    chosen = range(len(blocks)) if layers is None else layers
    seen: dict[int, list[torch.Tensor]] = {j: [] for j in chosen}
    handles = []
    for j in chosen:
        def keep(module, args, output, j=j):
            seen[j].append(output.detach().clone())
        handles.append(blocks[j].register_forward_hook(keep))
    try:
        yield seen
    finally:
        for handle in handles:
            handle.remove()


def finish(name: str, args, bench: Bench | None, values: dict, green: bool, started: float) -> int:
    """Write the gate's record, print it, and return the exit code."""
    verdict = "green" if green else "red"
    seconds = round(time.time() - started, 1)
    extra = {"planted": bool(args.plant), "model_kind": args.model, "tail_position": args.tail_position,
             "limit": args.limit, "seconds": seconds,
             "dtype": None if bench is None else str(bench.dtype).replace("torch.", ""),
             "builtin_path": None if bench is None else bench.builtin_path}
    if getattr(args, "writer_input", WRITER_INPUTS[0]) != WRITER_INPUTS[0]:
        extra["writer_input"] = args.writer_input
    record = write_record("gate", name, values, verdict=verdict, extra=extra)
    print(json.dumps({"gate": name, **extra, "values": values}, indent=1, default=str))
    print(f"verdict: {verdict}  seconds: {seconds}  record: {record}")
    return 0 if green else 1


def run_main(main) -> None:
    sys.exit(main())


def batches(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def prompt_pass_outputs(pipeline: Pipeline, batch: list[dict], p: torch.Tensor | None):
    """Every layer's output in the call over `0..r` and in the call over the later prompt tokens.

    Returns `(prefilled, outputs)`, where `outputs[layer]` is `[prefix call, later call]`.
    """
    with layer_outputs(pipeline.model) as seen:
        prefilled = pipeline.prefill(batch)
        pipeline.prompt_pass(prefilled, p)
    return prefilled, {layer: calls[:2] for layer, calls in seen.items()}


def max_difference_up_to_read(a, b, prefilled) -> dict[int, float]:
    """Per layer, the largest absolute difference at the real tokens `0..r` of the prefix call."""
    real = prefilled.prefix_attention.bool()
    return {layer: float((a[layer][0][real] - b[layer][0][real]).abs().max()) for layer in a}
