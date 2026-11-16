"""Put the pushback conversations to one Claude model, or estimate what that would cost.

    python -m check2.behaviour.claude_run --model <id> --out results/behaviour/<name> --key-file <.env>
    python -m check2.behaviour.claude_run --model <id> [--model <id> ...] --out <folder> --key-file <.env> --preflight

A run asks the model's first answers, then the follow-ups built on them, writing each reply
as it arrives so that a stopped run resumes where it stopped. Its first calls are timed, and
when the measured rate projects an end past the deadline, the questions not yet asked are
cut by a seeded draw and the record says so. The API key is read from the key file by this
process and handed to the client; it is never printed or written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Callable

from check2 import record
from check2.behaviour import claude_list
from check2.behaviour.backend import Item
from check2.behaviour.claude import SETTINGS, ClaudeBackend, SettingRejected, make_client
from check2.behaviour.claude_preflight import PLACEHOLDER_FIRST_ANSWER, estimate, format_estimates
from check2.behaviour.replies import append_replies, read_replies, reply_row
from check2.behaviour.spend import PRICES_FILE, CapReached, Spend, load_prices

KEY_NAME = "ANTHROPIC_API_KEY"
ENDPOINT = "https://api.anthropic.com/v1/messages"
TIMED_CALLS = 50
COUNTER = Path("behaviour") / "claude-spend.json"
PREFLIGHT_JSON = "preflight.json"


def read_key(path: Path) -> str:
    """The API key from a dotenv file; the value never leaves this process except to the client."""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        name, sep, value = line.strip().partition("=")
        if sep and name.strip().removeprefix("export ").strip() == KEY_NAME:
            return value.strip().strip('"').strip("'")
    raise KeyError(f"{KEY_NAME} is not set in {path}")


def _utc(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat(timespec="seconds")


def _deadline() -> float:
    data = json.loads(PRICES_FILE.read_text(encoding="utf-8"))
    return datetime.fromisoformat(data["deadline"]).timestamp()


class Run:
    """One model's run into one folder."""

    def __init__(self, model: str, out: Path, backend: ClaudeBackend, clock: Callable[[], float] = time.time,
                 deadline: float | None = None):
        self.model = model
        self.out = out
        self.backend = backend
        self.clock = clock
        self.deadline = _deadline() if deadline is None else deadline
        self.done = {r["item_id"]: r for r in read_replies(out)}
        self.kept: set[str] = set(claude_list.asked_qids(model))
        self.timing: dict | None = None
        self.sent_order: list[str] = []

    def _ask(self, items: list[Item]) -> None:
        by_id = {i.item_id: i for i in items}
        self.sent_order += [i.item_id for i in items]
        for reply in self.backend.generate(items):
            row = reply_row(by_id[reply.item_id], reply)
            append_replies(self.out, [row])
            self.done[row["item_id"]] = row

    def _first_replies(self) -> dict[str, str]:
        return {r["qid"]: r["text"] for r in self.done.values() if r["kind"] == "first" and r["qid"] in self.kept}

    def _open_first(self) -> list[Item]:
        return [i for i in claude_list.first_items(self.model) if i.qid in self.kept and i.item_id not in self.done]

    def _open_follow_ups(self) -> list[Item]:
        return [i for i in claude_list.follow_ups(self._first_replies())
                if i.qid in self.kept and i.item_id not in self.done]

    def _timed(self, items: list[Item]) -> list[Item]:
        """Ask the first calls of this process, time them, and cut the questions if the rate is short."""
        head, rest = items[:TIMED_CALLS], items[TIMED_CALLS:]
        start = self.clock()
        self._ask(head)
        elapsed = max(self.clock() - start, 1e-9)
        rate = len(head) / elapsed
        # Calls still to make, per question: its open first answer and its follow-ups. A question
        # whose first answer is not in yet is counted with the placeholder's follow-ups.
        firsts = self._first_replies()
        estimate_on = {q: firsts.get(q, PLACEHOLDER_FIRST_ANSWER) for q in self.kept}
        calls_of: Counter = Counter()
        for i in claude_list.follow_ups(estimate_on):
            if i.qid in self.kept and i.item_id not in self.done:
                calls_of[i.qid] += 1
        for i in self._open_first():
            calls_of[i.qid] += 1
        remaining = sum(calls_of.values())
        now = self.clock()
        projected_end = now + remaining / rate
        self.timing = {"timed_calls": len(head), "seconds": elapsed, "calls_per_second": rate,
                       "remaining_calls": remaining, "projected_end": _utc(projected_end),
                       "deadline": _utc(self.deadline), "cut": False}
        if projected_end > self.deadline:
            budget = int(rate * max(self.deadline - now, 0))
            # Questions already asked keep their follow-ups; only questions not yet asked are drawn.
            started = {q for q in self.kept if q in firsts}
            budget -= sum(calls_of[q] for q in started)
            open_qids = sorted(self.kept - started)
            kept_open = claude_list.cut_questions(open_qids, {q: calls_of[q] for q in open_qids},
                                                  max(budget, 0))
            before = len(self.kept)
            self.kept = started | set(kept_open)
            self.timing.update(cut=True, cut_seed=claude_list.CUT_SEED, questions_before=before,
                               questions_kept=len(self.kept))
            rest = [i for i in rest if i.qid in self.kept]
        return rest

    def execute(self) -> None:
        first = self._open_first()
        if first:
            rest = self._timed(first) if self.timing is None else first
            self._ask(rest)
        follow = self._open_follow_ups()
        if follow and self.timing is None:
            follow = self._timed(follow)
        self._ask(follow)

    def expected_ids(self) -> list[str]:
        """Every item of the kept questions: their first answers and the follow-ups their first replies give."""
        first = [i.item_id for i in claude_list.first_items(self.model) if i.qid in self.kept]
        follow = [i.item_id for i in claude_list.follow_ups(self._first_replies()) if i.qid in self.kept]
        return first + follow


def _run_id(out: Path) -> str:
    return out.resolve().relative_to((record.ROOT / record.RESULTS).resolve()).as_posix()


def run_model(model: str, out: Path, client, counter: Path, workers: int, chunk_size: int,
              clock: Callable[[], float] = time.time, deadline: float | None = None) -> int:
    cap, prices = load_prices()
    run_id = _run_id(out)
    config = {"model": model, "thinking": SETTINGS[model].label, "endpoint": ENDPOINT,
              "request_fields": SETTINGS[model].fields, "max_tokens": SETTINGS[model].max_tokens or "item cap",
              "workers": workers, "chunk_size": chunk_size, "cap_usd": str(cap),
              "prices_file": PRICES_FILE.relative_to(PRICES_FILE.parents[2]).as_posix()}
    run_json = record.ROOT / record.RESULTS / run_id / "run.json"
    if not run_json.exists():
        record.start_run(run_id, config)
    status, error = "failed", None
    with Spend(counter, cap) as spend:
        backend = ClaudeBackend(model, client, spend, prices[model], workers=workers, chunk_size=chunk_size)
        run = Run(model, out, backend, clock=clock, deadline=deadline)
        try:
            run.execute()
            status = "complete"
        except (CapReached, SettingRejected) as e:
            error = f"{type(e).__name__}: {e}"
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            expected = run.expected_ids()
            written = sum(1 for i in expected if i in run.done)
            if status == "complete" and written != len(expected):
                status, error = "failed", error or f"wrote {written} of {len(expected)} items"
            order = hashlib.sha256("\n".join(run.sent_order).encode("utf-8")).hexdigest()
            record.finish_run(run_json, status, len(expected), written, order)
            snap = spend.snapshot()
            kinds = Counter(r["kind"] for r in run.done.values())
            record.write_record("run", run_id, values={
                "model": model, "thinking": SETTINGS[model].label, "endpoint": ENDPOINT,
                "status": status, "error": error, "items_expected": len(expected), "items_written": written,
                "replies_by_kind": dict(sorted(kinds.items())),
                "refused": sum(bool(r["refused"]) for r in run.done.values()),
                "not_ended": sum(not r["ended"] for r in run.done.values()),
                "spend_this_model_usd": snap["by_model"].get(model, {}).get("spent_usd", "0"),
                "spend_all_models_usd": snap["spent_usd"], "cap_usd": snap["cap_usd"],
                "timing": run.timing, "questions_kept": len(run.kept)}, config=config)
    if error:
        print(f"{model}: run stopped: {error}", file=sys.stderr)
    return 0 if status == "complete" else 2


def preflight(models: list[str], out: Path, client, counter: Path, workers: int) -> int:
    cap, prices = load_prices()
    spent = Decimal(json.loads(counter.read_text(encoding="utf-8"))["spent_usd"]) if counter.exists() else Decimal(0)
    estimates = [estimate(client, m, prices[m], workers) for m in models]
    text = format_estimates(estimates, cap, spent)
    out.mkdir(parents=True, exist_ok=True)
    (out / PREFLIGHT_JSON).write_text(json.dumps({"estimates": estimates, "cap_usd": str(cap),
                                                  "spent_usd": str(spent)}, indent=1) + "\n",
                                      encoding="utf-8", newline="\n")
    print(text)
    return 0


def main(argv: list[str] | None = None, client_factory: Callable[[str], object] = make_client) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", action="append", required=True, choices=sorted(SETTINGS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True, help="a dotenv file that sets " + KEY_NAME)
    parser.add_argument("--preflight", action="store_true", help="estimate the cost; nothing billed is sent")
    parser.add_argument("--counter", type=Path, default=None,
                        help="the spending counter; by default results/behaviour/claude-spend.json")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args(argv)
    counter = args.counter or record.ROOT / record.RESULTS / COUNTER
    client = client_factory(read_key(args.key_file))
    if args.preflight:
        return preflight(args.model, args.out, client, counter, args.workers)
    if len(args.model) != 1:
        parser.error("a run takes one --model")
    return run_model(args.model[0], args.out, client, counter, args.workers, args.chunk_size)


if __name__ == "__main__":
    raise SystemExit(main())
