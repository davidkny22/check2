"""Gradient parity: the fast DeltaNet kernel's bf16 gradients in the training form match a float32 reference.

On 8 training episodes, 4 of each format, run as one forward batch per format, with writers
and adapter factors random at scale 0.01: the gradient of the consumer loss, computed in
bf16 on the fast DeltaNet kernel in the form training uses, is compared with the same
gradient computed in float32 on the built-in path over whole sequences, one episode at a
time. Cosines are taken per layer, over all of that layer's trained parameters together (a
writer counts with its write layer), and recorded. Green: the lowest is at least 0.99.

The library chooses its DeltaNet kernel when it is first imported, so the two gradients
come from two processes, each saving its gradients to a folder, and a third step compares
them and writes the record:

    python -m check2.gates.gradient_parity --model full --part fast --form <name> --out DIR
    python -m check2.gates.gradient_parity --model full --part builtin --out DIR   # the fast kernel not importable
    python -m check2.gates.gradient_parity --model full --part compare --out DIR

The random values are drawn on the CPU from one seeded generator, so both processes load
the same ones. `--implementation hooks|wrappers|library` (default `hooks`) names the adapter
implementation they are copied into; both parts must name the same one, and the record
names it. The gate has no tiny-model form: the fast kernel is not on this machine.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from check2.gates import mechanism_setup as setup
from check2.gates import training_setup
from check2.mechanism.control import ControlState
from check2.mechanism.generate import Pipeline, one_hot_situations
from check2.mechanism.loading import fast_kernel_importable
from check2.record import gpu_name, versions, write_record
from check2.training import bench as training_bench
from check2.training.forms import FORMS
from check2.training.objectives import consumer_step

NAME = "gradient_parity"
EPISODES_PER_FORMAT = 4
THRESHOLD = 0.99
PARTS = ("fast", "builtin")


def small_random_additions(bench: setup.Bench, control: ControlState, implementation: str = "hooks") -> Pipeline:
    """The pipeline with the writers and adapter factors both training checks draw at scale 0.01."""
    writers, adapter = training_setup.random_additions(bench, control, implementation)
    return Pipeline(bench.model, bench.tokenizer, control, writers=writers, adapter=adapter)


def _kernel_facts() -> dict:
    from transformers.models.qwen3_5 import modeling_qwen3_5

    chunk = modeling_qwen3_5.torch_chunk_gated_delta_rule
    return {"fast_kernel_importable": fast_kernel_importable(), "fast_kernel_imported": "fla" in sys.modules,
            "chunk_rule": f"{getattr(chunk, '__module__', None)}.{getattr(chunk, '__qualname__', None)}"}


def save_part(args, part: str) -> int:
    """Compute and save one process's gradients and the facts of its run."""
    started = time.time()
    exact = part == "builtin"
    if not exact and not fast_kernel_importable():
        raise SystemExit("the fast part needs the fast DeltaNet kernel package importable")
    bench = setup.load(args.model, exact=exact)
    episodes = training_setup.draw(args, EPISODES_PER_FORMAT)
    pipeline = small_random_additions(bench, ControlState(), args.implementation)
    parameters = training_setup.trained_parameters(pipeline)
    chosen = episodes["snap"] + episodes["short"]
    p = one_hot_situations(chosen).to(pipeline.device)
    examples = training_bench.examples(chosen, bench.tokenizer, args.targets)
    beta, step_size = training_bench.BETA, len(examples)
    training_setup.clear_gradients(parameters)
    try:
        if exact:
            loss = training_setup.one_at_a_time(pipeline, examples, p, beta, step_size)
            how = "whole sequences, one episode at a time"
        else:
            form = FORMS[args.form](pipeline)
            loss, start = 0.0, 0
            for fmt in ("snap", "short"):
                batch = examples[start:start + len(episodes[fmt])]
                losses = consumer_step(form, batch, p[start:start + len(batch)], beta, step_size)
                loss += float(losses.total.double().sum()) / step_size
                start += len(batch)
            how = f"{args.form}, one forward batch per format"
    finally:
        pipeline.close()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(training_setup.gradients(parameters), out / f"{part}.pt")
    facts = {"part": part, "implementation": args.implementation,
             "dtype": str(bench.dtype).replace("torch.", ""), "builtin_path": bench.builtin_path,
             "how": how, "loss": loss, "episode_ids": [e.episode_id for e in examples], "gpu": gpu_name(),
             "targets": training_setup.targets_fact(args.targets),
             "versions": versions(), "seconds": round(time.time() - started, 1), **_kernel_facts()}
    (out / f"{part}.json").write_text(json.dumps(facts, indent=1), encoding="utf-8")
    print(json.dumps(facts, indent=1))
    return 0


def per_layer_cosines(fast: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]) -> dict[int, float]:
    """Per decoder layer, the cosine between the two gradients of all that layer's trained parameters together."""
    if set(fast) != set(reference):
        raise ValueError("the two parts saved different parameters")
    layers: dict[int, list[str]] = {}
    for name in sorted(reference):
        layers.setdefault(training_setup.layer_of(name), []).append(name)
    out = {}
    for layer, names in sorted(layers.items()):
        a = torch.cat([fast[n].double().flatten() for n in names])
        b = torch.cat([reference[n].double().flatten() for n in names])
        scale = float(a.norm() * b.norm())
        out[layer] = float(a @ b) / scale if scale > 0 else float("nan")
    return out


def compare(args) -> int:
    started = time.time()
    out = Path(args.out)
    facts = {part: json.loads((out / f"{part}.json").read_text(encoding="utf-8")) for part in PARTS}
    grads = {part: torch.load(out / f"{part}.pt") for part in PARTS}
    cosines = per_layer_cosines(grads["fast"], grads["builtin"])
    lowest = min(cosines, key=lambda layer: cosines[layer])
    paths_right = (facts["fast"]["fast_kernel_imported"] and facts["fast"]["dtype"] == "bfloat16"
                   and facts["builtin"]["builtin_path"] and not facts["builtin"]["fast_kernel_importable"]
                   and facts["builtin"]["dtype"] == "float32")
    same_episodes = (facts["fast"]["episode_ids"] == facts["builtin"]["episode_ids"]
                     and facts["fast"]["targets"] == facts["builtin"]["targets"])
    # A part saved before parts named their implementation built the hook adapter, the only one then.
    implementation = {part: facts[part].get("implementation", "hooks") for part in PARTS}
    same_implementation = implementation["fast"] == implementation["builtin"]
    values = {"implementation": implementation["fast"] if same_implementation else implementation,
              "per_layer_cosine": {str(layer): c for layer, c in cosines.items()}, "lowest_cosine": cosines[lowest],
              "lowest_layer": lowest, "threshold": THRESHOLD, "losses": {p: facts[p]["loss"] for p in PARTS},
              "paths_as_required": paths_right, "same_episodes": same_episodes,
              "same_implementation": same_implementation, "parts": facts}
    green = (all(c == c and c >= THRESHOLD for c in cosines.values()) and paths_right and same_episodes
             and same_implementation)
    extra = {"model_kind": args.model, "form": facts["fast"]["how"], "seconds": round(time.time() - started, 1)}
    record = write_record("gate", NAME, values, verdict="green" if green else "red", extra=extra)
    print(json.dumps({k: v for k, v in values.items() if k != "parts"}, indent=1))
    print(f"verdict: {'green' if green else 'red'}  record: {record}")
    return 0 if green else 1


def main(argv=None) -> int:
    parser = training_setup.parser(__doc__.splitlines()[0])
    parser.add_argument("--part", choices=(*PARTS, "compare"), required=True)
    parser.add_argument("--form", choices=sorted(FORMS), default=None, help="the form training uses; the fast part")
    parser.add_argument("--out", required=True, help="the folder the parts save their gradients to")
    args = parser.parse_args(argv)
    if args.model != "full":
        raise SystemExit("gradient parity has no tiny-model form: the fast kernel is not on this machine")
    if args.plant:
        raise SystemExit("gradient parity has no planted fault")
    if args.part == "compare":
        return compare(args)
    if args.part == "fast" and args.form is None:
        raise SystemExit("the fast part needs --form, the form training uses")
    return save_part(args, args.part)


if __name__ == "__main__":
    setup.run_main(main)
