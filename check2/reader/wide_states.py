"""The read token's state of every episode of a split, taken from the reader's cache of the original model.

    python -m check2.reader.wide_states --split train|heldout [--cache <dir>] [--out <dir>]

The cache keeps each episode's read-layer outputs from its window start to its last tail
token. The read token's row is found through the index and the selected read position, not
by taking the last saved row, which is the last prompt token; the state layer is found by
its position in the cache's own list of layers. One file per split,
`/content/reader/wide/states_<split>.pt`, holds the episode ids in order and their states
`[N, d]` in the cache's number format. A cache that lacks any episode of the split is
refused before anything is written. The file holds hidden states and is never committed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from check2.mechanism import wide_signal
from check2.reader import cache, files
from check2.reader.window import window_length

WIDE_ROOT = files.SESSION_ROOT / "wide"
SPLITS = cache.CACHED_SPLITS


class StatesError(ValueError):
    """A cache that lacks an episode, or a states file that does not hold what is asked of it."""


def states_file(split: str, root: Path = WIDE_ROOT) -> Path:
    if split not in SPLITS:
        raise StatesError(f"states are saved for {SPLITS}, not {split!r}")
    return Path(root) / f"states_{split}.pt"


def read_states(cache_dir: Path, episode_ids: list[str], read_position: int,
                layer: int = wide_signal.STATE_LAYER) -> torch.Tensor:
    """The state layer's output at each episode's read token, `[N, d]`, in the cache's number format."""
    index = cache.read_index(str(cache_dir))
    missing = [e for e in episode_ids if e not in index]
    if missing:
        raise StatesError(f"the cache {cache_dir} lacks {len(missing)} episodes, e.g. {missing[:3]}")
    position = wide_signal.state_position(cache.read_meta(cache_dir)["layers"], layer)
    rows = []
    for episode_id in episode_ids:
        saved = cache.episode_rows(cache_dir, episode_id)
        rows.append(saved[window_length(saved.shape[0], read_position) - 1, position])
    return torch.stack(rows)


def write_states(split: str, episode_ids: list[str], read_position: int, cache_dir: Path,
                 root: Path = WIDE_ROOT, layer: int = wide_signal.STATE_LAYER) -> Path:
    """Save the split's states, ids sorted, after checking that the cache holds every episode."""
    ids = sorted(episode_ids)
    states = read_states(cache_dir, ids, read_position, layer)
    path = states_file(split, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"episode_ids": ids, "states": states.clone(), "layer": layer, "read_position": read_position,
                "split": split}, path)
    return path


def load_states(path: Path) -> dict[str, torch.Tensor]:
    """Each saved episode's state `[d]`, by episode id."""
    saved = torch.load(Path(path), weights_only=True)
    if len(saved["episode_ids"]) != saved["states"].shape[0]:
        raise StatesError(f"{path} holds {len(saved['episode_ids'])} ids and {saved['states'].shape[0]} states")
    return dict(zip(saved["episode_ids"], saved["states"]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Save the read token's state of every episode of a split.")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--cache", type=Path, default=files.CONVERSATION_CACHE)
    parser.add_argument("--out", type=Path, default=WIDE_ROOT)
    args = parser.parse_args(argv)
    read_position = int(files.read_json(files.SELECTION)["read_position"])
    ids = [e["episode_id"] for e in files.load_split(args.split)]
    path = write_states(args.split, ids, read_position, args.cache, args.out)
    print(f"{len(ids)} states of {args.split} at tail position {read_position}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
