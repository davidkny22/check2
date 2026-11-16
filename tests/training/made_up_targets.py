"""Made-up target replies for development runs of the training checks, of the lengths each format gives.

    python -m tests.training.made_up_targets <out file>

writes one made-up reply for every `named` and `reasoned` `train` episode, in the targets
file's record form, with `source` set to `made_up`. The replies mean nothing as language.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from check2.data import paths
from check2.gates import mechanism_setup as setup
from check2.training.bench import TRAINING_EPISODE_FORMS

WORDS = ("the", "option", "reason", "answer", "question", "holds", "fails", "because", "check", "again", "states")


def made_up_reply(episode: dict) -> str:
    """A bare letter for snap; for short, some words and a final answer line."""
    rng = random.Random(episode["episode_id"])
    letter = rng.choice("ABCD")
    if episode["format"] == "snap":
        return f"({letter})"
    words = " ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 40)))
    return f"{words.capitalize()}.\n\nFinal answer: ({letter})"


def training_candidates() -> list[dict]:
    return [e for e in setup.read_episodes(paths.EPISODES / "train.jsonl") if e["form"] in TRAINING_EPISODE_FORMS]


def write(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in training_candidates():
            f.write(json.dumps({"episode_id": e["episode_id"], "reply": made_up_reply(e), "a2": None,
                                "source": "made_up", "sample_index": 0}) + "\n")
    return path


if __name__ == "__main__":
    print(write(Path(sys.argv[1])))
