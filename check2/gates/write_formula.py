"""Write matches its formula: the write enters at exactly the write layers, at the strength its run says.

In float32 on the built-in path, with random non-zero writers: write hooks sit on exactly
the write layers; at each, what the hook passes on minus what it received equals
`mask * alpha * v_j` at every token, with relative difference at most 1e-4, at `alpha` 1 and
0.5; and it is exactly zero at `alpha` 0, and at the layers of a switched-off band for each
band. The expected mask is built from the episode structure: 0 in the call over tokens
`0..r`, 1 at every later token. `v_j` is computed from the writer's parameters directly.
"""

from __future__ import annotations

import contextlib
import math
import time

import torch

from check2.config.schema import WRITER_INPUTS
from check2.gates import mechanism_setup as setup
from check2.mechanism.control import ControlState
from check2.mechanism.generate import Pipeline, one_hot_situations
from check2.mechanism.model_parts import decoder_layers
from check2.mechanism.write import WRITE_HOOK_TAG, Writers

NAME = "write_formula"
EPISODES_PER_FORMAT = 16
DECODE_STEPS = 3
TOLERANCE = 1e-4
STRENGTHS = (1.0, 0.5)


def hooked_layers(model) -> list[int]:
    """The decoder layers that carry a write hook."""
    return [index for index, block in enumerate(decoder_layers(model))
            if any(hasattr(h, WRITE_HOOK_TAG) for h in block._forward_pre_hooks.values())]


def expected_vectors(writers: Writers, p: torch.Tensor) -> dict[int, torch.Tensor]:
    """`v_j = Linear_b(GELU(Linear_a(p)))` from the parameters, with GELU written out."""
    out = {}
    p = p.to(torch.float64)
    for j in writers.layers:
        a, b = writers.first[str(j)], writers.second[str(j)]
        u = p @ a.weight.to(torch.float64).t() + a.bias.to(torch.float64)
        u = 0.5 * u * (1.0 + torch.erf(u / math.sqrt(2.0)))
        out[j] = u @ b.weight.to(torch.float64).t() + b.bias.to(torch.float64)
    return out


@contextlib.contextmanager
def write_spies(model, layers):
    """What each write layer's hooks receive and pass on, per forward call."""
    blocks = decoder_layers(model)
    seen = {j: {"received": [], "passed": []} for j in layers}
    handles = []
    for j in layers:
        def received(module, args, kwargs, j=j):
            seen[j]["received"].append((args[0] if args else kwargs["hidden_states"]).detach().clone())

        def passed(module, args, kwargs, j=j):
            seen[j]["passed"].append((args[0] if args else kwargs["hidden_states"]).detach().clone())
        handles.append(blocks[j].register_forward_pre_hook(received, with_kwargs=True, prepend=True))
        handles.append(blocks[j].register_forward_pre_hook(passed, with_kwargs=True))
    try:
        yield seen
    finally:
        for handle in handles:
            handle.remove()


def compare(seen, vectors, alpha: float, silent: set[int]) -> dict:
    """The worst relative difference from the formula, and whether every required zero is exact.

    The first call recorded is the one over tokens `0..r`, where the write must be exactly zero.
    """
    worst, exact_zero, calls = 0.0, True, 0
    for j, record in seen.items():
        for call, (got_in, got_out) in enumerate(zip(record["received"], record["passed"])):
            calls += 1
            change = (got_out - got_in).to(torch.float64)
            if call == 0 or j in silent or alpha == 0.0:
                exact_zero = exact_zero and bool((change == 0).all())
                continue
            expected = alpha * vectors[j].to(change.device).unsqueeze(1).expand_as(change)
            worst = max(worst, float((change - expected).norm() / expected.norm()))
    return {"max_relative_difference": worst, "zeros_exact": exact_zero, "calls_checked": calls}


def check(bench: setup.Bench, episodes: dict[str, list[dict]], writer_input: str = WRITER_INPUTS[0]) -> dict:
    control = ControlState()
    measure = [e for fmt in episodes for e in episodes[fmt]]
    writers, _ = setup.random_additions(bench, control, measure, adapter_on=False, writer_input=writer_input)
    values = {"write_layers": list(bench.layout.write_layers), "arms": {}}
    pipeline = Pipeline(bench.model, bench.tokenizer, control, writers=writers)
    try:
        values["hooked_layers"] = hooked_layers(bench.model)
        settings = [(f"alpha={a}", a, ()) for a in STRENGTHS + (0.0,)]
        settings += [(f"band={list(band)}", 1.0, band) for band in bench.layout.bands]
        for fmt, batch in episodes.items():
            p = one_hot_situations(batch).to(pipeline.device)
            # Wide writers take S, built from each row's own read state; the formula is then applied to S.
            vectors = expected_vectors(writers, pipeline.writer_signal(p, pipeline.prefill(batch)))
            for label, alpha, band in settings:
                with write_spies(bench.model, bench.layout.write_layers) as seen:
                    prefilled = pipeline.prefill(batch)
                    pipeline.generate(prefilled, p, DECODE_STEPS, alpha=alpha, disabled_layers=band)
                values["arms"][f"{fmt}:{label}"] = compare(seen, vectors, alpha, set(band))
    finally:
        pipeline.close()
    placed = values["hooked_layers"] == list(bench.layout.write_layers)
    formula = all(arm["max_relative_difference"] <= TOLERANCE and arm["zeros_exact"] for arm in values["arms"].values())
    values["hooks_on_exactly_the_write_layers"] = placed
    values["green"] = placed and formula
    return values


def main(argv=None) -> int:
    args = setup.parser(__doc__.splitlines()[0]).parse_args(argv)
    if args.plant:
        raise SystemExit("this gate has no planted fault of its own")
    started = time.time()
    bench = setup.load(args.model, exact=True)
    episodes = setup.draw(setup.episode_file(args, "heldout"), EPISODES_PER_FORMAT, args.tail_position, args.limit)
    values = check(bench, episodes, writer_input=args.writer_input)
    return setup.finish(NAME, args, bench, values, values["green"], started)


if __name__ == "__main__":
    setup.run_main(main)
