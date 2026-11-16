"""The Steering baseline: one fixed direction per reply format, added to layer 16's output after the read.

Each format `f` has its own direction `d16[f]`: the raw mean difference of the output of
layer 16 at `r` over `train` named and reasoned episodes of format `f` with
`target != gold`, the mean where the original model did not adopt minus the mean where it
adopted, as one greedy Original pass over `train` decided. The difference is taken inside
each format because the original model holds mostly in short replies and adopts mostly in
snap replies, so a difference over both formats together measures the formats' own
difference. That pass is a run with `eval_split: train`; while it runs, the output of layer
16 at `r` of every episode is saved on the session's disk only, one file per batch.

    python -m check2.evaluation.steering --run <the train pass's run folder>

writes `data/steering/direction.json`. A Steering run adds `g` times the direction of a
batch's format to the output of layer 16 wherever the control state's mask is 1. Each
format is run by itself, so every batch has one format.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from check2.data import paths
from check2.evaluation.scoring import FORMATS
from check2.evaluation.tables import load_run, run_role
from check2.mechanism.control import ControlState
from check2.mechanism.model_parts import decoder_layers

STEERING_LAYER = 16
STATES_ROOT = Path("/content/steering")  # the session's own disk; never committed
DIRECTION_FILE = paths.DATA / "steering" / "direction.json"


class SteeringError(ValueError):
    """The direction cannot be computed or loaded as the baseline defines it."""


def states_folder(run_id: str) -> Path:
    return STATES_ROOT / run_id


@contextlib.contextmanager
def read_token_states(model: torch.nn.Module, control: ControlState, layer: int | None = None):
    """Collect the layer's output at the read token in each call over tokens `0..r`, where the mask is all 0.

    The read token sits in the last column of that call. Yields the list the states are added to.
    """
    kept: list[torch.Tensor] = []

    def keep(module, args, output):
        if control.mask is not None and not control.any_after:
            kept.append(output[:, -1].detach().float().cpu())

    handle = decoder_layers(model)[STEERING_LAYER if layer is None else layer].register_forward_hook(keep)
    try:
        yield kept
    finally:
        handle.remove()


def save_batch_states(run_id: str, index: int, episode_ids: Sequence[str], states: torch.Tensor) -> Path:
    folder = states_folder(run_id)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"batch-{index:05d}.pt"
    torch.save({"episode_ids": list(episode_ids), "states": states.clone()}, path)
    return path


def load_states(run_id: str) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for path in sorted(states_folder(run_id).glob("batch-*.pt")):
        saved = torch.load(path, weights_only=True)
        for i, eid in enumerate(saved["episode_ids"]):
            out[eid] = saved["states"][i]
    return out


def direction(records: Sequence[Mapping[str, Any]],
              states: Mapping[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, int]]]:
    """Per format: the mean state where the model did not adopt minus the mean where it adopted, with the counts.

    Only named and reasoned records with `target != gold` take part, each in its own format's groups.
    """
    groups: dict[str, dict[bool, list[torch.Tensor]]] = {fmt: {True: [], False: []} for fmt in FORMATS}
    for r in records:
        if r["form"] not in ("named", "reasoned") or r["target"] == r["gold"]:
            continue
        if r["format"] not in groups:
            raise SteeringError(f"{r['episode_id']}: unknown format {r['format']!r}; expected one of {FORMATS}")
        if r["episode_id"] not in states:
            raise SteeringError(f"{r['episode_id']}: no saved state at the read token")
        groups[r["format"]][r["a2"] == r["target"]].append(states[r["episode_id"]].double())
    vectors: dict[str, torch.Tensor] = {}
    counts: dict[str, dict[str, int]] = {}
    for fmt in FORMATS:
        held, adopted = groups[fmt][False], groups[fmt][True]
        counts[fmt] = {"not_adopted": len(held), "adopted": len(adopted)}
        if not held or not adopted:
            raise SteeringError(f"the {fmt} format's direction needs both groups: {len(held)} not adopted, "
                                f"{len(adopted)} adopted")
        vectors[fmt] = (torch.stack(held).mean(0) - torch.stack(adopted).mean(0)).float()
    return vectors, counts


def write_direction(run_folder: Path, output: Path | None = None) -> Path:
    """Compute each format's direction from the train pass's run and saved states; write them with their sources."""
    meta, records = load_run(run_folder)
    if meta.get("split") != "train" or run_role(meta["config"]) != ("original", ""):
        raise SteeringError(f"run {meta.get('run_id')} is not the Original pass over train")
    vectors, counts = direction(records, load_states(meta["run_id"]))
    value = {"layer": STEERING_LAYER,
             "vectors": {fmt: [float(x) for x in v.tolist()] for fmt, v in vectors.items()},
             "norms": {fmt: float(v.norm()) for fmt, v in vectors.items()},
             "counts": counts, "run_id": meta["run_id"], "run_commit": meta.get("commit"),
             "run_order_sha256": meta.get("order_sha256")}
    output = DIRECTION_FILE if output is None else output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8", newline="\n")
    return output


def load_direction(path: Path | None = None) -> tuple[dict[str, torch.Tensor], str]:
    """The saved direction of each format and the SHA-256 of its file."""
    path = DIRECTION_FILE if path is None else path
    if not path.exists():
        raise SteeringError(f"{path} does not exist; the train pass and the direction come first")
    data = path.read_bytes()
    saved = json.loads(data).get("vectors")
    if not isinstance(saved, dict) or sorted(saved) != sorted(FORMATS):
        raise SteeringError(f"{path} does not hold one direction for each of the formats {FORMATS}")
    vectors = {fmt: torch.tensor(saved[fmt], dtype=torch.float32) for fmt in FORMATS}
    return vectors, hashlib.sha256(data).hexdigest()


def batch_vector(vectors: Mapping[str, torch.Tensor], batch: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    """The direction of the one format a batch's episodes share; a batch of two formats is refused."""
    formats = sorted({e["format"] for e in batch})
    if len(formats) != 1:
        raise SteeringError(f"a steered batch has one format; this one has {formats}")
    if formats[0] not in vectors:
        raise SteeringError(f"no direction for the format {formats[0]!r}")
    return vectors[formats[0]]


@contextlib.contextmanager
def steered(model: torch.nn.Module, control: ControlState, vectors: Mapping[str, torch.Tensor],
            batch: Sequence[Mapping[str, Any]], strength: float, layer: int | None = None):
    """The direction of the batch's format, times `strength`, added while the batch runs, and removed after it."""
    handle = install_steering(model, control, batch_vector(vectors, batch), strength, layer)
    try:
        yield
    finally:
        handle.remove()


def install_steering(model: torch.nn.Module, control: ControlState, vector: torch.Tensor, strength: float,
                     layer: int | None = None):
    """Add `strength * vector` to the layer's output at every token the control state's mask marks."""
    def hook(module, args, output):
        if control.mask is None or not control.any_after:
            return None
        mask = control.mask_for(output)
        added = mask.unsqueeze(-1) * strength * vector.to(output.device).unsqueeze(0).unsqueeze(0)
        return output + added.to(output.dtype)

    return decoder_layers(model)[STEERING_LAYER if layer is None else layer].register_forward_hook(hook)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m check2.evaluation.steering", description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, type=Path, help="the Original pass over train")
    args = parser.parse_args(argv)
    path = write_direction(args.run)
    value = json.loads(path.read_text(encoding="utf-8"))
    each = "; ".join(f"{fmt} norm {value['norms'][fmt]:.4f}, {value['counts'][fmt]['not_adopted']} not adopted "
                     f"against {value['counts'][fmt]['adopted']} adopted" for fmt in FORMATS)
    print(f"direction from {value['run_id']}: {each}; wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
