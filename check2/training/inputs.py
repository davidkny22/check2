"""What a training reads: its episodes at the read position, their targets, their `p`, and the token budget.

`p` for a training episode is the reader's cross-fitted output saved in
`data/reader/p_train.jsonl`, or, with `signal: perfect`, the one-hot vector of the
episode's situation. An episode with no saved `p` is an error. The forward batches' token
budget is the newest measurement record of `token_budget` made on this card, on the same
DeltaNet path, in the form the training uses.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from check2.config.schema import Config
from check2.data import paths
from check2.data.episodes import TAIL_LENGTH, check_tail_position
from check2.mechanism.generate import one_hot_situations
from check2 import record
from check2.training.batches import NAME as BUDGET_RECORD
from check2.training.examples import TrainingExample, make_example, read_targets
from check2.training.sampling import training_episodes

P_TRAIN = paths.DATA / "reader" / "p_train.jsonl"
EPISODES_TRAIN = paths.EPISODES / "train.jsonl"


class InputError(ValueError):
    """A training input that is missing or does not fit the episodes."""


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def at_read_position(episode: dict, position: int) -> dict:
    """The episode with its read index at the given index of the template tail."""
    check_tail_position(position)
    return {**episode, "read_index": episode["n_prompt_tokens"] - TAIL_LENGTH + position}


def episodes_for(config: Config, episodes_file: Path, targets_file: Path) -> list[dict]:
    """The training episodes, sorted by id, with the read index of the config's resolved read position."""
    position = config.reader.read_position
    if not isinstance(position, int) or isinstance(position, bool):
        raise InputError(f"the read position {position!r} is not resolved; load the config for a run")
    targets = read_targets(targets_file)
    return [at_read_position(e, position) for e in training_episodes(read_jsonl(episodes_file), targets)]


def examples_for(episodes: list[dict], targets_file: Path, tokenizer) -> dict[str, TrainingExample]:
    targets = read_targets(targets_file)
    return {e["episode_id"]: make_example(e, targets[e["episode_id"]]["reply"], tokenizer) for e in episodes}


def files_for(config: Config) -> dict:
    """The `p` and targets files a training of record reads: the formed model's for the second stage of
    signal forming, otherwise the defaults of the training's options."""
    if config.signal_forming == "none":
        return {}
    from check2.training.forming import refit, targets

    return {"p_file": refit.P_TRAIN, "targets_file": targets.TARGETS}


def signals_for(config: Config, episodes: list[dict], p_file: Path,
                forming: dict | None = None) -> dict[str, torch.Tensor]:
    """Each episode's `p`, `[3]` in float32: the saved reader output, or the one-hot situation.

    Saved outputs are read through the stale-signal guard, with the forming update's factors
    when the training is on the formed model.
    """
    if config.signal == "perfect":
        return {e["episode_id"]: row for e, row in zip(episodes, one_hot_situations(episodes))}
    if config.signal != "reader":
        raise InputError(f"a training's signal is the reader's or the perfect one, not {config.signal!r}")
    from check2.training.forming.guard import load_p

    saved = load_p(p_file, forming)
    rows = read_jsonl(p_file)
    if len(saved) != len(rows):
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["episode_id"]] = counts.get(r["episode_id"], 0) + 1
        repeated = [episode_id for episode_id, count in counts.items() if count > 1]
        raise InputError(f"{p_file}: more than one saved p for {len(repeated)} of its episodes, first {repeated[:3]}")
    missing = [e["episode_id"] for e in episodes if e["episode_id"] not in saved]
    if missing:
        raise InputError(f"{len(missing)} training episodes have no saved p in {p_file}, e.g. {missing[:3]}")
    out = {}
    for e in episodes:
        p = saved[e["episode_id"]]
        if len(p) != 3:
            raise InputError(f"{e['episode_id']}: p has {len(p)} values")
        out[e["episode_id"]] = torch.tensor(p, dtype=torch.float32)
    return out


def file_fact(path: Path) -> dict:
    """A file a training read: its path as given and its SHA-256."""
    path = Path(path)
    return {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def measured_token_budget(form: str, gpu: str | None, builtin_path: bool) -> dict:
    """The newest `token_budget` record made on this card, on this DeltaNet path, in this form; its budget and file."""
    folder = record.ROOT / record.RESULTS / "measurements" / BUDGET_RECORD
    found = None
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else ():
        measured = json.loads(path.read_text(encoding="utf-8"))
        values = measured.get("values", {})
        if (values.get("form") == form and measured.get("gpu") == gpu and measured.get("model_kind") == "full"
                and measured.get("builtin_path") == builtin_path):
            found = {"token_budget": values["token_budget"], "record": path.name}
    if found is None:
        raise InputError(f"no token_budget record for the form {form} on {gpu} with builtin_path={builtin_path}; "
                         "measure it on this card first")
    return found
