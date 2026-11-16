"""Wide signal is the state it names: every row's writers receive its own `p` and its own state at its read token.

    python -m check2.gates.wide_signal --model tiny|full [--plant rows|position]
        [--episodes <file>] [--tail-position K] [--limit N]

In float32 on the built-in path, with random non-zero wide writers, 16 `heldout` episodes
per format run as one left-padded batch per format through the pipeline the runs use, with
a reader's kept layers in place. Every signal the writers receive (the prompt pass, cached
decoding, the recomputed sequence and the appended read) is compared, row by row, with
`concat(p, n)`, where `n` is computed here from the output of the state layer at that
row's own read token in an unpadded pass of that row alone. Green when the largest
relative difference is at most 1e-4.

Planted faults, each red: `rows`, the rows of the kept states rotated by one; `position`,
the state taken at the token after the read token.
"""

from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path

import torch

from check2.config.schema import WIDE_WRITER_INPUT, Reader
from check2.data import paths
from check2.gates import mechanism_setup as setup
from check2.mechanism import wide_signal
from check2.mechanism.control import ControlState
from check2.mechanism.generate import REPLY_CAPS, Pipeline, Reply, one_hot_situations

NAME = "wide_signal"
EPISODES_PER_FORMAT = 16
TOLERANCE = 1e-4
DECODE_STEPS = 3
PLANTS = ("rows", "position")
# The reader's kept layers in the gate's pipeline: the default variant's on the full model, and on
# the tiny model three of its layers around its state layer.
READ_LAYERS = {"full": tuple(Reader.layers), "tiny": (2, wide_signal.TINY_STATE_LAYER, 6)}


class _KeptLayers(torch.nn.Module):
    """A stand-in reader: keeps its layers' outputs over the window, as a reader does, and gives a uniform p."""

    def __init__(self, layers) -> None:
        super().__init__()
        self.layers = tuple(layers)

    def forward(self, states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.full((states.shape[0], 3), 1 / 3, dtype=torch.float32, device=states.device)


def parser(description: str) -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=description)
    out.add_argument("--plant", choices=PLANTS, default=None, help="apply one planted fault; the gate must go red")
    out.add_argument("--model", choices=("tiny", "full"), required=True)
    out.add_argument("--episodes", default=None, help="the episode file to draw from")
    out.add_argument("--tail-position", type=int, default=0, help="the read position, as an index into the tail")
    out.add_argument("--limit", type=int, default=None, help="take only the first N drawn episodes per format")
    return out


def _with_state(prefilled, state: torch.Tensor):
    prefilled.read_state = state
    return prefilled


@contextlib.contextmanager
def planted(fault: str | None):
    """The pipeline's kept state rotated by one row, or taken one token after the read token."""
    if fault is None:
        yield
        return
    original = Pipeline.prefill

    def rows(self, episodes):
        prefilled = original(self, episodes)
        return _with_state(prefilled, prefilled.read_state.roll(1, 0))

    def position(self, episodes):
        prefilled = original(self, episodes)
        ids = torch.cat([prefilled.prefix_ids, prefilled.rest_ids[:, :1]], 1)
        attention = torch.cat([prefilled.prefix_attention, torch.ones_like(prefilled.rest_ids[:, :1])], 1)
        layer = self.writers.state_layer
        with setup.layer_outputs(self.model, [layer]) as seen, torch.inference_mode():
            self.model(input_ids=ids, attention_mask=attention,
                       position_ids=(attention.cumsum(-1) - 1).clamp(min=0))
        return _with_state(prefilled, seen[layer][0][:, -1].clone())

    Pipeline.prefill = rows if fault == "rows" else position
    try:
        yield
    finally:
        Pipeline.prefill = original


def expected_signal(bench: setup.Bench, episode: dict, p_row: torch.Tensor, layer: int) -> torch.Tensor:
    """`concat(p, n)` with `n` from an unpadded pass of the conversation alone, the formula written out."""
    ids = bench.tokenizer([episode["prompt"]])["input_ids"][0][:episode["read_index"] + 1]
    device = next(bench.model.parameters()).device
    with torch.inference_mode():
        out = bench.model(input_ids=torch.tensor([ids], device=device), output_hidden_states=True)
    h = out.hidden_states[layer + 1][0, -1].to(torch.float32)
    centred = h - h.mean()
    n = centred / torch.sqrt((centred ** 2).mean() + 1e-5)
    return torch.cat([p_row.to(device=h.device, dtype=torch.float32), n])


def check(bench: setup.Bench, episodes: dict[str, list[dict]], model_kind: str, plant: str | None = None) -> dict:
    control = ControlState()
    measure = [e for fmt in episodes for e in episodes[fmt]]
    writers, _ = setup.random_additions(bench, control, measure, adapter_on=False, writer_input=WIDE_WRITER_INPUT)
    reader = _KeptLayers(READ_LAYERS[model_kind])
    values: dict = {"state_layer": writers.state_layer, "read_layers": list(reader.layers),
                    "tolerance": TOLERANCE, "plant": plant, "formats": {}}
    seen: list[torch.Tensor] = []
    spy = writers.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
    pipeline = Pipeline(bench.model, bench.tokenizer, control, writers=writers, reader=reader)
    try:
        for fmt, batch in episodes.items():
            p = one_hot_situations(batch).to(pipeline.device)
            expected = torch.stack([expected_signal(bench, e, p[b], writers.state_layer) for b, e in enumerate(batch)])
            seen.clear()
            with planted(plant):
                prefilled = pipeline.prefill(batch)
            pipeline.prompt_pass(prefilled, p)
            replies = pipeline.generate(prefilled, p, min(DECODE_STEPS, REPLY_CAPS[fmt]))
            pipeline.teacher_forced_logits(prefilled, p, [r.token_ids for r in replies], cached=False)
            pipeline.appended_letter_logits(prefilled, p, [Reply("A", [], True) for _ in batch])
            worst, worst_row = 0.0, None
            for signal in seen:
                rows = (signal.to(torch.float32) - expected).norm(dim=-1) / expected.norm(dim=-1)
                if float(rows.max()) > worst or worst_row is None:
                    worst, worst_row = float(rows.max()), int(rows.argmax())
            values["formats"][fmt] = {"episodes": len(batch), "signals_checked": len(seen),
                                      "signal_width": int(seen[0].shape[1]) if seen else None,
                                      "distinct_read_indices": len({e["read_index"] for e in batch}),
                                      "max_relative_difference": worst, "row_of_max": worst_row}
    finally:
        spy.remove()
        pipeline.close()
    width = wide_signal.size(bench.model.config.hidden_size)
    values["green"] = all(v["signals_checked"] == 4 and v["signal_width"] == width
                          and v["max_relative_difference"] <= TOLERANCE for v in values["formats"].values())
    return values


def main(argv=None) -> int:
    args = parser(__doc__.splitlines()[0]).parse_args(argv)
    started = time.time()
    bench = setup.load(args.model, exact=True)
    path = Path(args.episodes) if args.episodes else paths.EPISODES / "heldout.jsonl"
    episodes = setup.draw(path, EPISODES_PER_FORMAT, args.tail_position, args.limit)
    values = check(bench, episodes, args.model, plant=args.plant)
    return setup.finish(NAME, args, bench, values, values["green"], started)


if __name__ == "__main__":
    setup.run_main(main)
