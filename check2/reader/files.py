"""Where the reader's inputs and outputs live, and how its record files are read and written."""
import json
import random
from collections.abc import Iterable
from pathlib import Path

from check2.data import paths

READER_DATA = paths.DATA / "reader"
FIRST_TURN = READER_DATA / "first_turn.jsonl"
CACHE_INDEX = paths.CACHE_INDEX
SELECTION = READER_DATA / "selection.json"
P_TRAIN = READER_DATA / "p_train.jsonl"
P_HELDOUT = READER_DATA / "p_heldout.jsonl"
FOLD_FITS = READER_DATA / "fold_fits.json"
REASON_ONLY_P_TRAIN = READER_DATA / "reason_only_p_train.jsonl"
REASON_ONLY_P_HELDOUT = READER_DATA / "reason_only_p_heldout.jsonl"

# On a session: the caches and the fitted readers stay on its disk.
SESSION_ROOT = Path("/content/reader")
CONVERSATION_CACHE = SESSION_ROOT / "cache" / "conversation"
REASON_ONLY_CACHE = SESSION_ROOT / "cache" / "reason_only"
REASON_ONLY_ITEMS = REASON_ONLY_CACHE / "items.jsonl"
READER_WEIGHTS = SESSION_ROOT / "reader.pt"
REASON_ONLY_WEIGHTS = SESSION_ROOT / "reason_only_reader.pt"

# The main runs use seed 0; a draw of N episodes for a small run follows the gates' draw.
SEED = 0
SMALL_DRAW_SEED = 2035


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_split(split: str) -> list[dict]:
    """The episode records of one split, in the file's order."""
    return read_jsonl(paths.EPISODES / f"{split}.jsonl")


def small_draw(episodes: list[dict], n: int | None) -> list[dict]:
    """All episodes, or `n` of them per format drawn as a gate draws them, in id order."""
    if n is None:
        return episodes
    chosen = set()
    for fmt in sorted({e["format"] for e in episodes}):
        ids = sorted(e["episode_id"] for e in episodes if e["format"] == fmt)
        chosen.update(random.Random(SMALL_DRAW_SEED).sample(ids, min(n, len(ids))))
    return sorted((e for e in episodes if e["episode_id"] in chosen), key=lambda e: e["episode_id"])


def p_rows(p: dict[str, list[float]]) -> list[dict]:
    return [{"episode_id": e, "p": [float(v) for v in p[e]]} for e in sorted(p)]


def read_p(path: Path) -> dict[str, list[float]]:
    return {r["episode_id"]: r["p"] for r in read_jsonl(path)}
