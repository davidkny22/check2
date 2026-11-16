"""The episodes a run covers, their prompts, and the fixed order and batches they run in.

The episodes are those of the config's split, formats and forms. They run format by format,
snap then short, each format's episodes sorted by prompt length with ties broken by
`episode_id`, in batches of 96. The order is computed from the episode files alone, before
any prompt is changed, so two runs over the same episodes share an order whatever their
additions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from check2.config.schema import Config
from check2.data import paths
from check2.data.condition_episodes import reworded_file
from check2.data.episodes import FORMATS, TAIL_LENGTH, check_tail_position, render_prompt, tail_of

BATCH_SIZE = 96  # every reported run, in the generation settings
SPLIT_FILES = {"test": "test.jsonl", "heldout": "heldout.jsonl", "train": "train.jsonl"}
NEUTRAL_FILE = "neutral_test.jsonl"
CROSSWRITER_FILE = "crosswriter_test.jsonl"
EXPANDED_FILE = "expansion/episodes.jsonl"  # relative to the data folder
MAIN_WRITER = "opus"
PARTS = {"test_first": "first", "test_second": "second"}


class EpisodeError(ValueError):
    """The config asks for episodes the episode files do not hold."""


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _files(config: Config) -> list[str]:
    split, forms = config.data.eval_split, set(config.data.forms)
    if config.data.question_set == "expanded":
        return [EXPANDED_FILE]
    if config.data.base_sentence != "standard":
        return [reworded_file(config.data.base_sentence)]
    special = config.data.reason_writer != MAIN_WRITER or "neutral" in forms
    if special and split != "test":
        raise EpisodeError(f"the neutral follow-up and the cross-writer set exist only on test, not {split}")
    if config.data.reason_writer != MAIN_WRITER:
        if forms != {"reasoned"}:
            raise EpisodeError("the cross-writer set holds reasoned episodes only; its config takes forms [reasoned]")
        return [CROSSWRITER_FILE]
    files = [SPLIT_FILES[split]] if forms - {"neutral"} else []
    if "neutral" in forms:
        files.append(NEUTRAL_FILE)
    return files


def select(config: Config, folder: Path | None = None) -> list[dict]:
    """The config's episodes as the files hold them, with the read index of the config's read position."""
    folder = paths.EPISODES if folder is None else folder
    position = config.reader.read_position
    if not isinstance(position, int):
        raise EpisodeError(f"the read position {position!r} is not resolved; load the config for a run")
    check_tail_position(position)
    chosen = []
    for name in _files(config):
        # The expanded set's episodes are kept with the rest of that set, apart from the main episode files.
        path = folder.parent / name if name == EXPANDED_FILE else folder / name
        for e in _read(path):
            if e["format"] in config.data.formats and e["form"] in config.data.forms:
                chosen.append({**e, "read_index": e["n_prompt_tokens"] - TAIL_LENGTH + position})
    for fmt in config.data.formats:
        for form in config.data.forms:
            if not any(e["format"] == fmt and e["form"] == form for e in chosen):
                raise EpisodeError(f"no {form} episode in the {fmt} format for this config")
    if len({e["episode_id"] for e in chosen}) != len(chosen):
        raise EpisodeError("an episode id appears twice in the config's episodes")
    return chosen


def order(episodes: list[dict]) -> list[list[dict]]:
    """The batches: per format, snap then short, by prompt length then `episode_id`, 96 at a time."""
    batches = []
    for fmt in FORMATS:
        rows = sorted((e for e in episodes if e["format"] == fmt), key=lambda e: (e["n_prompt_tokens"], e["episode_id"]))
        batches += [rows[i:i + BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]
    return batches


def order_sha256(batches: list[list[dict]]) -> str:
    """The SHA-256 of the episode ids in batch order, one per line."""
    text = "\n".join(e["episode_id"] for batch in batches for e in batch)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parts(splits_file: Path | None = None) -> dict[str, str]:
    """The part of `test` each `test` question belongs to, `first` or `second`."""
    splits = json.loads((paths.SPLITS if splits_file is None else splits_file).read_text(encoding="utf-8"))
    return {qid: part for key, part in PARTS.items() for qid in splits[key]}


def with_system_suffix(episodes: list[dict], suffix: str) -> list[dict]:
    """The episodes with the suffix appended to the system turn, the read index found again from the tail.

    The prompt is tokenised by the model's own tokenizer, whose ids the template tail is
    defined in.
    """
    from check2.data.pool import complying, load_pool
    from check2.data.tokenizer import get_tokenizer

    tokenizer = get_tokenizer()
    questions = complying(load_pool())
    out = []
    for e in episodes:
        position = e["read_index"] - (e["n_prompt_tokens"] - TAIL_LENGTH)
        prompt = render_prompt(questions[e["qid"]], e["form"], e["format"], e["target"], e["reason"],
                               system_suffix=suffix)
        ids = tokenizer(prompt)["input_ids"]
        start, _ = tail_of(ids)
        out.append({**e, "prompt": prompt, "n_prompt_tokens": len(ids), "read_index": start + position})
    return out
