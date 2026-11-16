"""The timed batch, the pipeline it runs through, and timed repeats of its generation."""

from __future__ import annotations

import hashlib
import json
import time

import torch

from check2.config.schema import RANK_TO_ALPHA_LORA, Config
from check2.data import paths
from check2.evaluation.scoring import score_stated
from check2.mechanism.adapter.implementations import build
from check2.mechanism.control import ControlState
from check2.mechanism.generate import REPLY_CAPS, Pipeline, Reply
from check2.mechanism.loading import load_full, load_tiny

FORMAT = "short"
CAP = REPLY_CAPS[FORMAT]
BATCH_SIZE = 96
REPEATS = 3
SPREAD_LIMIT = 0.10
RANK = Config().adapter.rank
ALPHA_LORA = RANK_TO_ALPHA_LORA[RANK]


def first_batch(limit: int | None = None) -> list[dict]:
    """The first 96 short-format `heldout` episodes by prompt length, ties by episode id; then the first `limit`."""
    with (paths.EPISODES / "heldout.jsonl").open(encoding="utf-8") as f:
        episodes = [json.loads(line) for line in f if line.strip()]
    ordered = sorted((e for e in episodes if e["format"] == FORMAT), key=lambda e: (e["n_prompt_tokens"],
                                                                                     e["episode_id"]))
    batch = ordered[:BATCH_SIZE]
    return batch if limit is None else batch[:limit]


def load(model: str):
    """The model in the runs' number format: bf16 on the card for the full model, float32 for the tiny one."""
    return load_full(torch.bfloat16) if model == "full" else load_tiny(torch.float32)


def make_pipeline(model, tokenizer, layout, implementation: str) -> Pipeline:
    """A pipeline with no write and, unless `none`, the adapter at initialisation in the named implementation."""
    control = ControlState()
    adapter = None
    if implementation != "none":
        adapter = build(implementation, model, control, RANK, ALPHA_LORA, layout.adapter_first, layout.adapter_last)
    return Pipeline(model, tokenizer, control, adapter=adapter)


def synchronise() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def generate(pipeline: Pipeline, batch: list[dict]) -> list[Reply]:
    with torch.inference_mode():
        return pipeline.generate(pipeline.prefill(batch), None, CAP)


def timed_generation(pipeline: Pipeline, batch: list[dict]) -> tuple[float, list[Reply]]:
    synchronise()
    started = time.perf_counter()
    replies = generate(pipeline, batch)
    synchronise()
    return time.perf_counter() - started, replies


def replies_digest(replies: list[Reply]) -> str:
    return hashlib.sha256(json.dumps([r.token_ids for r in replies]).encode("utf-8")).hexdigest()


def repeats(pipeline: Pipeline, batch: list[dict]) -> tuple[dict, list[Reply]]:
    """One warm-up run, then `REPEATS` timed runs of the same batch; returns the timings and the first replies.

    The spread is (slowest - fastest) / fastest.
    """
    generate(pipeline, batch)
    seconds, digests, first = [], [], None
    for _ in range(REPEATS):
        elapsed, replies = timed_generation(pipeline, batch)
        seconds.append(elapsed)
        digests.append(replies_digest(replies))
        first = first or replies
    spread = (max(seconds) - min(seconds)) / min(seconds)
    return {"seconds": seconds, "mean_seconds": sum(seconds) / len(seconds), "spread": spread,
            "spread_above_limit": spread > SPREAD_LIMIT, "repeats_identical": len(set(digests)) == 1,
            "replies_sha256": digests[0], "tokens_generated": sum(len(r.token_ids) for r in first),
            "rows_ended": sum(r.ended for r in first)}, first


def final_answers(pipeline: Pipeline, batch: list[dict], replies: list[Reply], room=None) -> list[str]:
    """Each reply's stated answer, or the appended read where it states none.

    `room`, where a preallocated cache is on, is sized for the appended texts first.
    """
    answers = [score_stated(r.text, FORMAT) for r in replies]
    missing = [i for i, a in enumerate(answers) if a is None]
    if missing:
        unstated, their_replies = [batch[i] for i in missing], [replies[i] for i in missing]
        if room is not None:
            room.for_appended_read(pipeline, unstated, their_replies)
        with torch.inference_mode():
            prefilled = pipeline.prefill(unstated)
            read = pipeline.appended_read(prefilled, None, their_replies)
        for i, letter in zip(missing, read):
            answers[i] = letter
    return answers
