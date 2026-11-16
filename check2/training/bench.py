"""What the training measurements share: the pipeline with fresh additions, and the examples they run.

A measurement of a training step runs the frozen model in the number format and on the
DeltaNet path training uses, with writers and an adapter at their initial values, on
`named` and `reasoned` `train` episodes that have a target reply.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from check2.config.schema import RANK_TO_ALPHA_LORA, Config
from check2.data import paths
from check2.gates import mechanism_setup as setup
from check2.mechanism.adapter.base import GatedAdapter
from check2.mechanism.adapter.implementations import IMPLEMENTATIONS
from check2.mechanism.control import ControlState
from check2.mechanism.generate import Pipeline
from check2.mechanism.write import Writers
from check2.training.examples import TrainingExample, make_example, read_targets
from check2.training.forms import FORMS

ADAPTER_CHOICES = ("none", *IMPLEMENTATIONS)
TRAINING_EPISODE_FORMS = ("named", "reasoned")
RANK = Config().adapter.rank
ALPHA_LORA = RANK_TO_ALPHA_LORA[RANK]
BETA = Config().adapter.beta


def parser(description: str) -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=description)
    out.add_argument("--model", choices=("tiny", "full"), default="full")
    out.add_argument("--form", choices=sorted(FORMS), required=True)
    out.add_argument("--adapter", choices=ADAPTER_CHOICES, default="hooks")
    out.add_argument("--episodes", default=None, help="the episode file; train by default")
    out.add_argument("--targets", default=None, help="the targets file; the build's by default")
    out.add_argument("--tail-position", type=int, default=0, help="the read position, as an index into the tail")
    return out


def adapter_class(implementation: str) -> type[GatedAdapter] | None:
    """The adapter class of an implementation, or None for `none`."""
    return None if implementation == "none" else IMPLEMENTATIONS[implementation]


def training_pipeline(bench: setup.Bench, implementation: str) -> Pipeline:
    """The pipeline with writers and, unless `none`, an adapter, each at its initial values."""
    control = ControlState()
    device = next(bench.model.parameters()).device
    writers = Writers(bench.layout.write_layers, bench.model.config.hidden_size).to(device)
    cls = adapter_class(implementation)
    adapter = None if cls is None else cls(bench.model, control, RANK, ALPHA_LORA, bench.layout.adapter_first,
                                           bench.layout.adapter_last)
    return Pipeline(bench.model, bench.tokenizer, control, writers=writers, adapter=adapter)


def candidate_episodes(args, fmt: str) -> list[dict]:
    """The episodes of one format that training would use: `named` or `reasoned`, with a target."""
    episode_file = Path(args.episodes) if args.episodes else paths.EPISODES / "train.jsonl"
    targets = read_targets(args.targets or paths.TARGETS)
    return [setup.at_tail_position(e, args.tail_position) for e in setup.read_episodes(episode_file)
            if e["format"] == fmt and e["form"] in TRAINING_EPISODE_FORMS and e["episode_id"] in targets]


def examples(episodes: list[dict], tokenizer, targets_file) -> list[TrainingExample]:
    targets = read_targets(targets_file or paths.TARGETS)
    return [make_example(e, targets[e["episode_id"]]["reply"], tokenizer) for e in episodes]


def drawn(episodes: list[dict], count: int) -> list[dict]:
    """`random.Random(2035).sample(sorted(ids), count)`, the draw a gate makes from a split."""
    by_id = {e["episode_id"]: e for e in episodes}
    return [by_id[i] for i in random.Random(setup.EPISODE_DRAW_SEED).sample(sorted(by_id), count)]
