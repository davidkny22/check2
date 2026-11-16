"""What the Claude models have cost so far, kept on disk, with a cap no call can pass.

Every call is priced at its worst case before it is sent: its counted input tokens and its
`max_tokens`, at the list price. A chunk of calls goes only when what has been spent, what
is reserved for calls in flight and the chunk's worst case together stay under the cap.
When a reply arrives its reservation is replaced by its cost from the reply's own token
counts. The counter is one file; a lock file beside it lets one process hold it at a time.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Mapping

PRICES_FILE = Path(__file__).resolve().parents[2] / "experiments" / "behaviour" / "claude-prices.json"
MILLION = Decimal(1_000_000)
SAVE_ATTEMPTS = 50
SAVE_WAIT_S = 0.1


class CapReached(RuntimeError):
    """A reservation would take spending to or past the cap; nothing was reserved."""


class CounterHeld(RuntimeError):
    """Another process holds the spending counter."""


@dataclass(frozen=True)
class Price:
    """List prices of one model, in dollars per million tokens."""

    model: str
    input_per_mtok: Decimal
    output_per_mtok: Decimal
    cache_write_per_mtok: Decimal
    cache_read_per_mtok: Decimal
    max_prompt_tokens: int | None  # the prompt size up to which these prices hold, where the page gives one

    def worst_case(self, input_tokens: int, max_tokens: int) -> Decimal:
        """The most one call can cost: its counted input and every token `max_tokens` allows."""
        self._check_prompt(input_tokens)
        return (input_tokens * self.input_per_mtok + max_tokens * self.output_per_mtok) / MILLION

    def cost(self, input_tokens: int, output_tokens: int, cache_write_tokens: int = 0,
             cache_read_tokens: int = 0) -> Decimal:
        """What one call cost, from the token counts its reply reports."""
        self._check_prompt(input_tokens + cache_write_tokens + cache_read_tokens)
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok
                + cache_write_tokens * self.cache_write_per_mtok
                + cache_read_tokens * self.cache_read_per_mtok) / MILLION

    def _check_prompt(self, prompt_tokens: int) -> None:
        if self.max_prompt_tokens is not None and prompt_tokens > self.max_prompt_tokens:
            raise ValueError(f"{self.model}: a prompt of {prompt_tokens} tokens is past the "
                             f"{self.max_prompt_tokens} tokens these prices hold for")


def load_prices(path: Path = PRICES_FILE) -> tuple[Decimal, dict[str, Price]]:
    """The cap and each model's prices, from the prices file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    prices = {}
    for model, p in data["prices"].items():
        prices[model] = Price(model=model, input_per_mtok=Decimal(str(p["input_per_mtok"])),
                              output_per_mtok=Decimal(str(p["output_per_mtok"])),
                              cache_write_per_mtok=Decimal(str(p["cache_write_5m_per_mtok"])),
                              cache_read_per_mtok=Decimal(str(p["cache_read_per_mtok"])),
                              max_prompt_tokens=p["max_prompt_tokens"])
    return Decimal(str(data["cap_usd"])), prices


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Spend:
    """The spending counter, held by this process from `open` until `close`."""

    def __init__(self, path: Path, cap: Decimal):
        self.path = Path(path)
        self.cap = Decimal(cap)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self._guard = threading.Lock()
        self._held = False
        self.spent = Decimal(0)
        self.reserved: dict[str, Decimal] = {}
        self.by_model: dict[str, dict] = {}
        self.carried_over = Decimal(0)

    # Holding the counter

    def open(self) -> "Spend":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.lock_path.open("x", encoding="utf-8") as f:
                f.write(json.dumps({"pid": os.getpid(), "since": _now()}) + "\n")
        except FileExistsError:
            raise CounterHeld(f"the spending counter {self.path} is held by another process "
                              f"({self.lock_path} exists); if no runner is alive, remove that file "
                              f"by hand after checking the provider's usage") from None
        self._held = True
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.spent = Decimal(data["spent_usd"])
            self.by_model = data["by_model"]
            self.carried_over = Decimal(data.get("carried_over_usd", "0"))
            # Reservations left by a process that stopped are counted as spent at their worst case.
            left = sum((Decimal(v) for v in data["reserved"].values()), Decimal(0))
            self.spent += left
            self.carried_over += left
        self._save()
        return self

    def close(self) -> None:
        if not self._held:
            return
        with self._guard:
            self._save()
        self.lock_path.unlink()
        self._held = False

    def __enter__(self) -> "Spend":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # Reserving and settling

    def reserve(self, worst_case: Mapping[str, Decimal]) -> None:
        """Reserve the worst case of every call of a chunk, or nothing.

        Refuses with `CapReached` unless spent, reserved and the chunk's worst case together
        stay under the cap.
        """
        self._require_held()
        with self._guard:
            clash = sorted(set(worst_case) & set(self.reserved))
            if clash:
                raise ValueError(f"already reserved: {clash[:5]}")
            new = sum((Decimal(v) for v in worst_case.values()), Decimal(0))
            total = self.spent + sum(self.reserved.values(), Decimal(0)) + new
            if total >= self.cap:
                raise CapReached(f"spent {self.spent} and reserved {self.reserved_total()} with a "
                                 f"chunk of worst case {new} would reach the cap {self.cap}")
            self.reserved.update({k: Decimal(v) for k, v in worst_case.items()})
            self._save()

    def settle(self, key: str, actual: Decimal, model: str, input_tokens: int = 0,
               output_tokens: int = 0) -> None:
        """Replace the reservation `key` by what the call cost."""
        self._require_held()
        with self._guard:
            self.reserved.pop(key)
            self.spent += Decimal(actual)
            row = self.by_model.setdefault(model, {"calls": 0, "spent_usd": "0", "input_tokens": 0,
                                                   "output_tokens": 0})
            row["calls"] += 1
            row["spent_usd"] = str(Decimal(row["spent_usd"]) + Decimal(actual))
            row["input_tokens"] += input_tokens
            row["output_tokens"] += output_tokens
            self._save()

    def release(self, key: str) -> None:
        """Drop the reservation of a call the provider rejected before running it, so nothing was billed."""
        self._require_held()
        with self._guard:
            self.reserved.pop(key)
            self._save()

    def reserved_total(self) -> Decimal:
        return sum(self.reserved.values(), Decimal(0))

    def snapshot(self) -> dict:
        with self._guard:
            return self._state()

    # Disk

    def _require_held(self) -> None:
        if not self._held:
            raise RuntimeError("the spending counter is not held by this process")

    def _state(self) -> dict:
        return {"cap_usd": str(self.cap), "spent_usd": str(self.spent),
                "reserved": {k: str(v) for k, v in sorted(self.reserved.items())},
                "carried_over_usd": str(self.carried_over), "by_model": self.by_model,
                "updated": _now()}

    def _save(self) -> None:
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self._state(), indent=1, sort_keys=True) + "\n", encoding="utf-8",
                       newline="\n")
        # On Windows the rename fails while another process has the counter open for a moment.
        for attempt in range(SAVE_ATTEMPTS):
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:
                if attempt == SAVE_ATTEMPTS - 1:
                    raise
                time.sleep(SAVE_WAIT_S)
