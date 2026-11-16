"""A training example: an episode's prompt and its target reply as token ids, with its read index.

The teacher-forced sequence is the prompt, the target reply and the end-of-turn token. The
prompt is tokenised as the episode records it, the reply on its own, and the end-of-turn
token is appended, so the reply's tokens and the end-of-turn token are the ones the loss
counts. The tokens fed to the model are every token of that sequence but the last, since
the last is only ever predicted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from check2.data.episodes import END_OF_TURN


class ExampleError(ValueError):
    """An episode or a target that cannot make a training example."""


@dataclass(frozen=True)
class TrainingExample:
    episode_id: str
    format: str
    prompt_ids: tuple[int, ...]
    target_ids: tuple[int, ...]  # the reply's tokens, then the end-of-turn token
    read_index: int

    @property
    def prefix_length(self) -> int:
        """The tokens `0..r`."""
        return self.read_index + 1

    @property
    def rest_length(self) -> int:
        """The prompt tokens after `r`."""
        return len(self.prompt_ids) - self.read_index - 1

    @property
    def continuation_ids(self) -> tuple[int, ...]:
        """The tokens fed after `r`: the rest of the prompt and the reply, without the end-of-turn token."""
        return self.prompt_ids[self.read_index + 1:] + self.target_ids[:-1]

    @property
    def sequence_ids(self) -> tuple[int, ...]:
        """Every token fed to the model: the prompt and the reply, without the end-of-turn token."""
        return self.prompt_ids + self.target_ids[:-1]


def make_example(episode: dict, reply: str, tokenizer) -> TrainingExample:
    """The training example of one episode and its target reply's text."""
    prompt_ids = tuple(tokenizer(episode["prompt"])["input_ids"])
    if len(prompt_ids) != episode["n_prompt_tokens"]:
        raise ExampleError(f"{episode['episode_id']}: {len(prompt_ids)} prompt tokens, the record says "
                           f"{episode['n_prompt_tokens']}")
    r = episode["read_index"]
    if not 0 <= r < len(prompt_ids) - 1:
        raise ExampleError(f"{episode['episode_id']}: read index {r} leaves no prompt token after it")
    target_ids = tuple(tokenizer.encode(reply, add_special_tokens=False)) + (
        tokenizer.convert_tokens_to_ids(END_OF_TURN),)
    return TrainingExample(episode_id=episode["episode_id"], format=episode["format"], prompt_ids=prompt_ids,
                           target_ids=target_ids, read_index=r)


def read_targets(path: Path) -> dict[str, dict]:
    """The target records of a targets file, by episode id; an episode with two targets is refused."""
    out: dict[str, dict] = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["episode_id"] in out:
                raise ExampleError(f"{path}: two targets for {record['episode_id']}")
            out[record["episode_id"]] = record
    return out
