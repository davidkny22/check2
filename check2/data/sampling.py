"""Sampled replies of the original model for every named and reasoned training episode.

    python -m check2.data.sampling [--limit N] [--samples 4] [--cells FILE] [--out FILE]

The model runs in bf16 with no additions, through the library's own `generate`, on
left-padded batches. Each format's episodes are sorted by prompt length, ties broken by
episode id, and cut into batches. Before each batch the generator is seeded with the run's
seed plus the batch's index, and that seed is written beside every sample of the batch.
Each batch is written as a shard as soon as it is done; a run that is started again keeps
every complete shard that matches its plan and samples only the rest. When every batch is
done the shards are joined into one file in batch order.

`--limit N` keeps the first N episodes of each format, for a check at small size.
`--cells FILE` keeps the episodes of the listed (situation, format) cells, for sampling
those cells again with more samples; the file is a JSON list of objects with the keys
`situation` and `format`.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import time
from pathlib import Path

from check2.data import paths
from check2.data.episodes import END_OF_TURN, FORMATS
from check2.data.targets import TARGET_FORMS
from check2.data.tokenizer import MODEL_ID

# Sampler settings and reply caps, from the specification's target replies and conversation format.
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
REPLY_CAP = {"snap": 16, "short": 256}
# The configuration's `target_samples`: 4, and 8 for a short-format cell that drops too many.
TARGET_SAMPLES = 4
RESAMPLED_SAMPLES = 8
# The configuration's `seed`.
RUN_SEED = 0
# Episodes per batch: the batch size of the evaluation's generation settings.
BATCH_EPISODES = 96

TARGETS_DIR = paths.TARGETS.parent
SAMPLES = TARGETS_DIR / "samples.jsonl.gz"
RESAMPLED = TARGETS_DIR / "samples_resampled.jsonl.gz"
TRAIN_EPISODES = paths.EPISODES / "train.jsonl"
EPISODES_META = paths.EPISODES / "meta.json"
PINS = paths.ROOT / "experiments" / "pins.json"


class PlanMismatch(RuntimeError):
    """A shard on disk was written for another plan of batches."""


# ---------- files ----------

def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write JSON lines, gzipped when the name ends in `.gz`, through a temporary file and a rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if path.suffix == ".gz":
        with open(tmp, "wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
            for row in rows:
                gz.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
    else:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def load_cells(path: Path) -> set[tuple[str, str]]:
    cells = json.loads(Path(path).read_text(encoding="utf-8"))
    return {(c["situation"], c["format"]) for c in cells}


# ---------- what is sampled, in what order ----------

def training_episodes(path: Path = TRAIN_EPISODES,
                      cells: set[tuple[str, str]] | None = None) -> list[dict]:
    """The named and reasoned episodes of the training file, optionally only those of some cells.

    Refuses an episode outside `train` and a reasoned episode with no reason.
    """
    chosen = []
    for e in read_jsonl(Path(path)):
        if e["form"] not in TARGET_FORMS:
            continue
        if e["split"] != "train":
            raise ValueError(f"{e['episode_id']} is in {e['split']}; only train episodes are sampled")
        if e["form"] == "reasoned" and not e["reason"]:
            raise ValueError(f"{e['episode_id']} is reasoned but carries no reason; "
                             f"sampling waits for the complete data")
        if cells is None or (e["situation"], e["format"]) in cells:
            chosen.append(e)
    return chosen


def plan_batches(episodes: list[dict], batch_size: int, limit: int | None = None,
                 batch_tokens: dict[str, int] | None = None) -> list[list[dict]]:
    """Batches of each format in turn, its episodes sorted by prompt length, then episode id.

    A batch holds at most `batch_size` episodes and, where `batch_tokens` names its format, at
    most that many prompt tokens counted as its episodes times its longest prompt, since every
    row of a left-padded batch is as long as the longest.
    """
    batches = []
    for fmt in FORMATS:
        rows = sorted((e for e in episodes if e["format"] == fmt),
                      key=lambda e: (e["n_prompt_tokens"], e["episode_id"]))
        if limit is not None:
            rows = rows[:limit]
        budget = (batch_tokens or {}).get(fmt)
        batch: list[dict] = []
        for e in rows:
            full = len(batch) == batch_size or (
                budget is not None and (len(batch) + 1) * e["n_prompt_tokens"] > budget)
            if batch and full:
                batches.append(batch)
                batch = []
            batch.append(e)
        if batch:
            batches.append(batch)
    return batches


# ---------- generation ----------

def split_reply(new_ids: list[int], end_id: int) -> tuple[list[int], bool]:
    """The reply's tokens before the first end-of-turn token, and whether that token was generated."""
    if end_id in new_ids:
        return new_ids[:new_ids.index(end_id)], True
    return list(new_ids), False


def left_padded(prompt_ids: list[list[int]], pad_id: int):
    """Input ids padded on the left with the padding token, and the matching attention mask."""
    import torch

    width = max(len(p) for p in prompt_ids)
    ids = torch.full((len(prompt_ids), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(prompt_ids), width), dtype=torch.long)
    for i, p in enumerate(prompt_ids):
        ids[i, width - len(p):] = torch.tensor(p, dtype=torch.long)
        mask[i, width - len(p):] = 1
    return ids, mask


def generate_samples(model, prompt_ids: list[list[int]], cap: int, n_samples: int, seed: int,
                     end_id: int, pad_id: int) -> list[list[tuple[list[int], bool]]]:
    """`n_samples` sampled replies per prompt, each as (tokens before the end of turn, ended)."""
    import torch

    ids, mask = left_padded(prompt_ids, pad_id)
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    with torch.inference_mode():
        out = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device), do_sample=True,
                             temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K, max_new_tokens=cap,
                             num_return_sequences=n_samples, eos_token_id=end_id, pad_token_id=pad_id)
    new = out[:, ids.shape[1]:].tolist()
    # The library repeats each prompt n_samples times in a row.
    return [[split_reply(new[i * n_samples + s], end_id) for s in range(n_samples)]
            for i in range(len(prompt_ids))]


def sample_batch(model, tokenizer, batch: list[dict], n_samples: int, seed: int, batch_index: int,
                 end_id: int) -> list[dict]:
    """One batch's sample lines: each episode with its samples' text, end and length."""
    prompt_ids = [tokenizer(e["prompt"])["input_ids"] for e in batch]
    for e, ids in zip(batch, prompt_ids):
        if len(ids) != e["n_prompt_tokens"]:
            raise ValueError(f"{e['episode_id']}: the prompt has {len(ids)} tokens here and "
                             f"{e['n_prompt_tokens']} in the episode file")
    fmt = batch[0]["format"]
    replies = generate_samples(model, prompt_ids, REPLY_CAP[fmt], n_samples, seed, end_id,
                               tokenizer.pad_token_id)
    lines = []
    for e, samples in zip(batch, replies):
        lines.append({
            "episode_id": e["episode_id"], "format": e["format"], "form": e["form"],
            "situation": e["situation"], "batch": batch_index, "seed": seed,
            "samples": [{"text": tokenizer.decode(ids, skip_special_tokens=False), "ended": ended,
                         "n_tokens": len(ids) + int(ended)} for ids, ended in samples],
        })
    return lines


# ---------- shards ----------

def parts_dir(out: Path) -> Path:
    return out.with_name(out.name + ".parts")


def shard_path(out: Path, batch_index: int) -> Path:
    return parts_dir(out) / f"{batch_index:05d}.jsonl.gz"


def seconds_path(out: Path, batch_index: int) -> Path:
    return parts_dir(out) / f"{batch_index:05d}.seconds"


def complete_shard(out: Path, batch_index: int, batch: list[dict], seed: int,
                   n_samples: int) -> list[dict] | None:
    """A shard already on disk for this batch, checked against the plan, or None."""
    path = shard_path(out, batch_index)
    if not path.exists():
        return None
    lines = read_jsonl(path)
    if ([x["episode_id"] for x in lines] != [e["episode_id"] for e in batch]
            or any(x["seed"] != seed or len(x["samples"]) != n_samples for x in lines)):
        raise PlanMismatch(f"{path} was written for another plan; remove {parts_dir(out)} to start again")
    return lines


def peak_memory() -> str:
    """The GPU's peak allocated memory so far, for the progress line, or nothing without a GPU."""
    import torch

    if not torch.cuda.is_available():
        return ""
    return f", peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB"


def sample_all(model, tokenizer, batches: list[list[dict]], n_samples: int, run_seed: int,
               out: Path) -> dict:
    """Sample every batch not yet on disk, join the shards into `out`, and return the timing."""
    end_id = tokenizer.convert_tokens_to_ids(END_OF_TURN)
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == end_id:
        raise ValueError("the tokenizer needs a padding token that is not the end-of-turn token")
    parts_dir(out).mkdir(parents=True, exist_ok=True)
    all_lines, per_format = [], {}
    for b, batch in enumerate(batches):
        seed = run_seed + b
        fmt = batch[0]["format"]
        lines = complete_shard(out, b, batch, seed, n_samples)
        if lines is None:
            start = time.perf_counter()
            lines = sample_batch(model, tokenizer, batch, n_samples, seed, b, end_id)
            seconds = time.perf_counter() - start
            seconds_path(out, b).write_text(f"{seconds:.3f}\n", encoding="utf-8")
            write_jsonl(shard_path(out, b), lines)
            note = "sampled"
        else:
            note = "kept from disk"
        seconds = float(seconds_path(out, b).read_text(encoding="utf-8"))
        cell = per_format.setdefault(fmt, {"episodes": 0, "replies": 0, "ended": 0, "seconds": 0.0})
        cell["episodes"] += len(lines)
        cell["replies"] += sum(len(x["samples"]) for x in lines)
        cell["ended"] += sum(s["ended"] for x in lines for s in x["samples"])
        cell["seconds"] += seconds
        all_lines += lines
        print(f"batch {b + 1}/{len(batches)} {fmt}: {len(lines)} episodes, seed {seed}, "
              f"{seconds:.1f} s, {note}{peak_memory()}", flush=True)
    write_jsonl(out, all_lines)
    shutil.rmtree(parts_dir(out))
    for cell in per_format.values():
        cell["seconds"] = round(cell["seconds"], 3)
        cell["replies_per_second"] = round(cell["replies"] / cell["seconds"], 3) if cell["seconds"] else None
        cell["ended_share"] = round(cell["ended"] / cell["replies"], 4) if cell["replies"] else None
    return per_format


# ---------- the model ----------

def model_revision() -> str:
    """The pinned model revision, or the tokenizer revision the episodes were built with."""
    if PINS.exists():
        return json.loads(PINS.read_text(encoding="utf-8"))["model_revision"]
    return json.loads(EPISODES_META.read_text(encoding="utf-8"))["tokenizer_revision"]


def load_original_model(revision: str):
    """The original model in bf16 on the GPU with no additions, and its tokenizer padding on the left."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=revision, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa", device_map="cuda")
    return model.eval(), tokenizer


def parse_budgets(text: str) -> dict[str, int]:
    """`snap=8832,short=5000` as a map from format to token budget."""
    budgets = {}
    for part in text.split(","):
        fmt, _, value = part.partition("=")
        if fmt not in FORMATS or not value.isdigit():
            raise argparse.ArgumentTypeError(f"{part!r} is not FORMAT=TOKENS with a format of {FORMATS}")
        budgets[fmt] = int(value)
    return budgets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None, help="keep the first N episodes of each format")
    parser.add_argument("--samples", type=int, default=TARGET_SAMPLES, help="replies per episode")
    parser.add_argument("--cells", type=Path, default=None, help="JSON list of {situation, format}")
    parser.add_argument("--out", type=Path, default=None, help="the samples file to write")
    parser.add_argument("--batch", type=int, default=BATCH_EPISODES, help="episodes per batch")
    parser.add_argument("--batch-tokens", type=parse_budgets, default=None, metavar="FMT=N,...",
                        help="per format, at most N episodes times longest prompt per batch")
    parser.add_argument("--seed", type=int, default=RUN_SEED, help="the run's seed")
    args = parser.parse_args(argv)
    from check2.record import write_record

    out = args.out or (RESAMPLED if args.cells else SAMPLES)
    cells = load_cells(args.cells) if args.cells else None
    episodes = training_episodes(cells=cells)
    batches = plan_batches(episodes, args.batch, args.limit, args.batch_tokens)
    print(f"{sum(len(b) for b in batches)} episodes in {len(batches)} batches, "
          f"{args.samples} samples each, to {out}", flush=True)
    revision = model_revision()
    model, tokenizer = load_original_model(revision)
    per_format = sample_all(model, tokenizer, batches, args.samples, args.seed, out)
    config = {"model": MODEL_ID, "model_revision": revision, "temperature": TEMPERATURE, "top_p": TOP_P,
              "top_k": TOP_K, "reply_cap": REPLY_CAP, "samples": args.samples, "seed": args.seed,
              "batch": args.batch, "batch_tokens": args.batch_tokens, "limit": args.limit,
              "cells": sorted(map(list, cells)) if cells else None, "out": out.name}
    record = write_record("measurement", "target_sampling", {"per_format": per_format}, config=config)
    for fmt, cell in per_format.items():
        print(f"{fmt}: {json.dumps(cell)}")
    print(f"wrote {out}; record {record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
