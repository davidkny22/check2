"""Run the pushback episodes on one model with nothing added, first answers then follow-ups.

    python -m check2.behaviour.run --backend local --model <repo id> --out results/behaviour/<name>
        [--revision <commit>] [--batch-size N] [--limit N]

Every reply is appended to the folder's replies file as its batch ends, so a run that stops
resumes where it stopped: an item already answered is not asked again. The follow-ups are
built on the model's own first replies. When every item has its reply, a record is written
through `check2.record` with the model's repository id and revision, its thinking setting,
the card, the commit and the counts, and a copy of it is kept in the run folder.

`--limit N` keeps the first N questions of the pool and of the expanded sample, for a check
at small size.
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from check2 import record
from check2.behaviour.backend import Item
from check2.behaviour.items import SOURCES, first_answer_items, follow_up_items
from check2.behaviour.replies import append_replies, read_replies, reply_row
from check2.data.sampling import BATCH_EPISODES

BACKENDS = ("local",)


def limited(items: list[Item], limit: int | None) -> list[Item]:
    """The items of the first `limit` questions of each source, in question order."""
    if limit is None:
        return items
    kept: dict[str, list[str]] = {source: [] for source in SOURCES}
    for item in items:
        ids = kept[item.source]
        if item.qid not in ids and len(ids) < limit:
            ids.append(item.qid)
    allowed = {qid for ids in kept.values() for qid in ids}
    return [item for item in items if item.qid in allowed]


def answer(backend, items: Sequence[Item], folder: Path, log=print) -> int:
    """Ask every item that has no reply in the folder yet; returns how many were asked."""
    done = {row["item_id"] for row in read_replies(folder)}
    todo = [item for item in items if item.item_id not in done]
    by_id = {item.item_id: item for item in todo}
    log(f"{len(items)} items, {len(items) - len(todo)} already answered, {len(todo)} to ask")
    if not todo:
        return 0
    for replies in backend.generate_batches(todo):
        append_replies(folder, [reply_row(by_id[r.item_id], r) for r in replies])
    return len(todo)


def counts(rows: list[dict]) -> dict:
    """Replies per source, kind and format, and the replies that did not end or were refused."""
    per_cell = Counter(f"{r['source']}/{r['kind']}/{r['fmt'] or '-'}" for r in rows)
    return {"replies": len(rows), "per_cell": dict(sorted(per_cell.items())),
            "not_ended": sum(not r["ended"] for r in rows), "refused": sum(bool(r["refused"]) for r in rows)}


def run(backend, folder: Path, model: str, revision: str, limit: int | None, log=print,
        write_record=record.write_record) -> Path:
    """First answers, then follow-ups on them, then the record; returns the record's path."""
    firsts = limited(first_answer_items(), limit)
    answer(backend, firsts, folder, log)
    rows = read_replies(folder)
    first_replies = {r["qid"]: r["text"] for r in rows if r["kind"] == "first"}
    follow_ups = follow_up_items(first_replies)
    answer(backend, follow_ups, folder, log)
    rows = read_replies(folder)
    expected = {item.item_id for item in firsts} | {item.item_id for item in follow_ups}
    answered = {r["item_id"] for r in rows}
    if expected != answered:
        raise RuntimeError(f"{len(expected - answered)} items have no reply and "
                           f"{len(answered - expected)} replies answer no item")
    values = {"model": model, "revision": revision, "thinking": backend.thinking,
              "batch_size": backend.batch_size, "limit": limit, "first_items": len(firsts),
              "follow_up_items": len(follow_ups), **counts(rows)}
    config = {"backend": "local", "model": model, "revision": revision, "thinking": backend.thinking,
              "batch_size": backend.batch_size, "limit": limit}
    name = "behaviour/" + folder.name
    path = write_record("run", name, values, config=config)
    if path.parent.resolve() != folder.resolve():
        shutil.copy2(path, folder / path.name)
    log(f"record {path}")
    return path


def resolve_revision(repo_id: str, revision: str | None) -> str:
    """The commit of the model's repository that the run loads."""
    from huggingface_hub import HfApi

    return HfApi().model_info(repo_id, revision=revision).sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--model", required=True, help="the model's repository id")
    parser.add_argument("--revision", default=None, help="a commit of the repository; the newest by default")
    parser.add_argument("--out", type=Path, required=True, help="the run folder")
    parser.add_argument("--batch-size", type=int, default=BATCH_EPISODES)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    from check2.behaviour.local import load

    revision = resolve_revision(args.model, args.revision)
    print(f"model {args.model} at {revision}")
    backend = load(args.model, revision, args.batch_size)
    print(f"thinking: {backend.thinking}; end-of-turn ids {backend.end_ids}")
    run(backend, args.out, args.model, revision, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
