"""The cache of the original model's hidden states that the reader trains on.

One forward pass per episode, in left-padded batches, saves the outputs of the read layers
at every token from the first token of the last user turn's content through the last tail
token, in the model's own number format. Episodes are taken shortest first; each run of
`SHARD_SIZE` of them is one shard, a tensor `[tokens, layers, d]`. The index maps each
episode id to `[shard, offset, saved_length]`, where the offset is the episode's first
token row in its shard. No state of a `test` episode is ever written.
"""
import json
from functools import lru_cache
from pathlib import Path

import torch

from check2.reader.window import spans, window_length

READ_LAYERS = (14, 15, 18, 19, 22, 23)
SHARD_SIZE = 512
BATCH_SIZE = 32
CACHED_SPLITS = ("train", "heldout")
INDEX_FILE = "index.json"
META_FILE = "meta.json"


def shard_name(shard: int) -> str:
    return f"shard_{shard:04d}.pt"


def tokenize(tokenizer, episodes: list[dict]) -> list[dict]:
    """Each episode's token ids, as `tokenizer(prompt)` gives them, and where its window starts."""
    encoded = tokenizer([e["prompt"] for e in episodes])["input_ids"]
    out = []
    for e, ids in zip(episodes, encoded):
        window_start, _ = spans(ids, tokenizer)
        out.append({"episode_id": e["episode_id"], "ids": list(ids), "window_start": window_start})
    return out


def batch_order(items: list[dict]) -> list[dict]:
    """Shortest prompt first, ties by episode id."""
    return sorted(items, key=lambda it: (len(it["ids"]), it["episode_id"]))


@torch.inference_mode()
def saved_states(model, items: list[dict], layers: tuple[int, ...] | list[int],
                 pad_id: int) -> list[torch.Tensor]:
    """One left-padded forward pass over `items`; for each, the read layers' outputs from its
    window start to its last token, as `[tokens, len(layers), d]`."""
    device = next(model.parameters()).device
    width = max(len(it["ids"]) for it in items)
    input_ids = torch.full((len(items), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(items), width), dtype=torch.long)
    for b, it in enumerate(items):
        n = len(it["ids"])
        input_ids[b, width - n:] = torch.tensor(it["ids"], dtype=torch.long)
        attention_mask[b, width - n:] = 1
    out = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device),
                output_hidden_states=True, logits_to_keep=1)
    hidden = torch.stack([out.hidden_states[l + 1] for l in layers], dim=2)  # B, T, L, d
    states = []
    for b, it in enumerate(items):
        first = width - len(it["ids"]) + it["window_start"]
        states.append(hidden[b, first:].detach().to("cpu").clone())
    return states


def _refuse_uncached_splits(episodes: list[dict]) -> None:
    bad = sorted({e.get("split") for e in episodes} - set(CACHED_SPLITS), key=str)
    if bad:
        raise ValueError(f"only {CACHED_SPLITS} episodes are cached; got episodes of {bad}")


def cache_states(model, tokenizer, episodes: list[dict], out_dir: str | Path,
                 layers: tuple[int, ...] | list[int] = READ_LAYERS,
                 batch_size: int = BATCH_SIZE) -> dict[str, list[int]]:
    """Run the model once over every episode and save its read layers' states; returns the index.

    Each episode is a dict with `episode_id`, `split` and `prompt`. Writes the shards,
    `index.json` and `meta.json` to `out_dir`. Refuses any episode outside `train` and
    `heldout`, before anything is written.
    """
    _refuse_uncached_splits(episodes)
    ids = [e["episode_id"] for e in episodes]
    if len(set(ids)) != len(ids):
        raise ValueError("an episode id appears twice")
    if SHARD_SIZE % batch_size:
        raise ValueError(f"the batch size {batch_size} must divide the shard size {SHARD_SIZE}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    items = batch_order(tokenize(tokenizer, episodes))
    index: dict[str, list[int]] = {}
    batches: list[list[str]] = []
    shard_rows: list[torch.Tensor] = []
    shard, offset = 0, 0
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        batches.append([it["episode_id"] for it in chunk])
        for it, rows in zip(chunk, saved_states(model, chunk, layers, pad_id)):
            index[it["episode_id"]] = [shard, offset, rows.shape[0]]
            shard_rows.append(rows)
            offset += rows.shape[0]
        if (start + batch_size) % SHARD_SIZE == 0 or start + batch_size >= len(items):
            torch.save(torch.cat(shard_rows), out_dir / shard_name(shard))
            shard_rows, shard, offset = [], shard + 1, 0
    dtype = str(next(model.parameters()).dtype).removeprefix("torch.")
    meta = {"layers": list(layers), "shard_size": SHARD_SIZE, "batch_size": batch_size,
            "shards": shard, "episodes": len(index), "dtype": dtype, "batches": batches}
    (out_dir / INDEX_FILE).write_text(json.dumps(index, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / META_FILE).write_text(json.dumps(meta) + "\n", encoding="utf-8")
    read_index.cache_clear()
    _shard.cache_clear()
    return index


@lru_cache(maxsize=4)
def read_index(out_dir: str) -> dict[str, list[int]]:
    return json.loads((Path(out_dir) / INDEX_FILE).read_text(encoding="utf-8"))


def read_meta(out_dir: str | Path) -> dict:
    return json.loads((Path(out_dir) / META_FILE).read_text(encoding="utf-8"))


@lru_cache(maxsize=64)
def _shard(out_dir: str, shard: int) -> torch.Tensor:
    return torch.load(Path(out_dir) / shard_name(shard), mmap=True, weights_only=True)


def episode_rows(out_dir: str | Path, episode_id: str) -> torch.Tensor:
    """Every saved token row of one episode, `[saved_length, layers, d]`."""
    shard, offset, length = read_index(str(out_dir))[episode_id]
    return _shard(str(out_dir), shard)[offset:offset + length]


def right_align(windows: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack windows `[w, layers, d]` into `h [B, layers, W, d]`, each ending in the last column,
    with zeros before it, and `mask [B, W]`, true on the window's own columns."""
    width = max(w.shape[0] for w in windows)
    _, n_layers, d = windows[0].shape
    h = torch.zeros((len(windows), n_layers, width, d), dtype=windows[0].dtype)
    mask = torch.zeros((len(windows), width), dtype=torch.bool)
    for b, w in enumerate(windows):
        h[b, :, width - w.shape[0]:] = w.transpose(0, 1)
        mask[b, width - w.shape[0]:] = True
    return h, mask


def load_states(out_dir: str | Path, episode_ids: list[str],
                tail_position: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The saved window of each episode, with the read token at `tail[tail_position]`.

    Returns `h [B, layers, W, d]` in the saved number format, right-aligned so that the read
    token is the last column, and `mask [B, W]`, true on each episode's window tokens.
    """
    windows = []
    for eid in episode_ids:
        rows = episode_rows(out_dir, eid)
        windows.append(rows[:window_length(rows.shape[0], tail_position)])
    return right_align(windows)


class StateBank:
    """The saved states of a list of episodes, held as one flat tensor on one device.

    `window` returns the same `(h, mask)` as `load_states`, gathered on the bank's device.
    """

    def __init__(self, rows: dict[str, torch.Tensor], device: str | torch.device = "cpu"):
        """`rows` maps each episode id to its saved rows, `[saved_length, layers, d]`."""
        starts, lengths = {}, {}
        position = 0
        for eid, r in rows.items():
            starts[eid], lengths[eid] = position, r.shape[0]
            position += r.shape[0]
        first = next(iter(rows.values()))
        self.device = torch.device(device)
        # Filled row by row, so the states are never held twice in host memory.
        self.flat = torch.empty((position, *first.shape[1:]), dtype=first.dtype, device=self.device)
        for eid, r in rows.items():
            self.flat[starts[eid]:starts[eid] + lengths[eid]] = r.to(self.device)
        self.starts, self.lengths = starts, lengths

    @classmethod
    def from_cache(cls, out_dir: str | Path, episode_ids: list[str],
                   device: str | torch.device = "cpu") -> "StateBank":
        return cls({eid: episode_rows(out_dir, eid) for eid in episode_ids}, device)

    def window(self, episode_ids: list[str], tail_position: int,
               layer_index: list[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """`(h, mask)` for the episodes, optionally with only the layers at `layer_index`."""
        widths = [window_length(self.lengths[e], tail_position) for e in episode_ids]
        width = max(widths)
        columns = torch.arange(width)
        rows = []
        for eid, w in zip(episode_ids, widths):
            rows.append(self.starts[eid] + columns - (width - w))
        position = torch.stack(rows)
        mask = position >= torch.tensor([self.starts[e] for e in episode_ids])[:, None]
        position = torch.where(mask, position, torch.zeros_like(position)).to(self.device)
        mask = mask.to(self.device)
        h = self.flat[position]                               # B, W, L, d
        if layer_index is not None:
            h = h[:, :, layer_index]
        h = torch.where(mask[:, :, None, None], h, torch.zeros((), dtype=h.dtype, device=h.device))
        return h.transpose(1, 2), mask
