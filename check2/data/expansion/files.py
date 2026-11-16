"""Reading and writing the expanded set's JSON and JSON-lines files, and its counts file."""
import json
from pathlib import Path

from check2.data.expansion import paths


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8",
                    newline="\n")


def record_counts(name: str, counts: dict) -> None:
    """Store one part's counts under `name` in the counts file, keeping the other parts."""
    current = read_json(paths.COUNTS) if paths.COUNTS.exists() else {}
    current[name] = counts
    write_json(paths.COUNTS, dict(sorted(current.items())))
