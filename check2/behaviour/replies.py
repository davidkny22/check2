"""A run folder's replies file: one line per item answered, written as the replies are made.

Each line holds the item's identity (`item_id`, `qid`, `source`, `kind`, `fmt`, `target`,
`writer`) and the reply (`text`, `ended`, `refused`, `detail`). The file is appended to
batch by batch, so a run that stops keeps every reply it finished. A finished folder may
hold the file compressed, as `replies.jsonl.gz`.
"""

from __future__ import annotations

import gzip
import json
import os
from collections.abc import Iterable
from pathlib import Path

from check2.behaviour.backend import Item, Reply

REPLIES = "replies.jsonl"
REPLIES_COMPRESSED = REPLIES + ".gz"


def reply_row(item: Item, reply: Reply) -> dict:
    """The line written for one reply."""
    if reply.item_id != item.item_id:
        raise ValueError(f"reply {reply.item_id} does not answer item {item.item_id}")
    return {"item_id": item.item_id, "qid": item.qid, "source": item.source, "kind": item.kind,
            "fmt": item.fmt, "target": item.target, "writer": item.writer, "text": reply.text,
            "ended": reply.ended, "refused": reply.refused, "detail": reply.detail}


def append_replies(folder: Path, rows: Iterable[dict]) -> None:
    """Append rows to the folder's replies file and flush them to disk."""
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / REPLIES).open("a", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_replies(folder: Path) -> list[dict]:
    """Every row of the folder's replies file, plain or compressed; an item answered twice is refused."""
    plain, packed = folder / REPLIES, folder / REPLIES_COMPRESSED
    if plain.exists() and packed.exists():
        raise ValueError(f"{folder} holds both {REPLIES} and {REPLIES_COMPRESSED}")
    if plain.exists():
        f = plain.open(encoding="utf-8")
    elif packed.exists():
        f = gzip.open(packed, "rt", encoding="utf-8")
    else:
        return []
    with f:
        rows = [json.loads(line) for line in f if line.strip()]
    seen: set[str] = set()
    for row in rows:
        if row["item_id"] in seen:
            raise ValueError(f"{folder}: item {row['item_id']} was answered twice")
        seen.add(row["item_id"])
    return rows


def compress_replies(folder: Path) -> Path:
    """Replace the folder's replies file by its compressed copy, for committing."""
    plain, packed = folder / REPLIES, folder / REPLIES_COMPRESSED
    with plain.open("rb") as src, packed.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as dst:
            dst.write(src.read())
    plain.unlink()
    return packed
