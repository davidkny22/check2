"""Claude models over the provider's Messages API, each call one fresh conversation, under the spending cap.

A request holds the item's system turn and its other turns and nothing else: no sampling
parameter, no fallback model, no prompt caching. Each model runs at one fixed thinking
setting, named in the backend's `name` and in every reply's `detail`.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Iterable, Iterator, Sequence

import anthropic

from check2.behaviour.backend import Item, Reply
from check2.behaviour.spend import Price, Spend


@dataclass(frozen=True)
class Setting:
    """How one model is called."""

    label: str  # the thinking setting as reported
    fields: dict  # request fields added to every call of this model
    max_tokens: int | None  # one cap for every call, or None for the item's own cap


SETTINGS = {
    "claude-haiku-5-5": Setting("thinking disabled", {"thinking": {"type": "disabled"}}, None),
    # The only setting under which this model runs without thinking.
    "claude-sonnet-5-5": Setting("thinking between tools only", {"thinking": {"type": "between_tools"}},
                                 None),
    # Thinking cannot be switched off on this model; it counts toward max_tokens.
    "claude-opus-5-5": Setting("thinking on, effort low", {"output_config": {"effort": "low"}}, 2048),
}

MAX_ATTEMPTS = 6
FIRST_WAIT_S = 2.0
LONGEST_WAIT_S = 60.0


class SettingRejected(RuntimeError):
    """The provider rejected a request as invalid, for example a thinking setting the model refuses."""


def make_client(api_key: str) -> anthropic.Anthropic:
    """A client that never retries by itself, so that every attempt passes through the spending counter."""
    return anthropic.Anthropic(api_key=api_key, max_retries=0)


def max_tokens_for(model: str, item: Item) -> int:
    fixed = SETTINGS[model].max_tokens
    return item.max_new_tokens if fixed is None else fixed


def split_turns(item: Item) -> tuple[str, list[dict]]:
    """The item's system turn, and its other turns as the Messages API takes them."""
    first, *rest = item.messages
    if first["role"] != "system" or any(m["role"] == "system" for m in rest):
        raise ValueError(f"{item.item_id}: an item has exactly one system turn, first")
    if not rest or rest[-1]["role"] != "user":
        raise ValueError(f"{item.item_id}: an item ends on a user turn")
    return first["content"], [{"role": m["role"], "content": m["content"]} for m in rest]


def request(model: str, item: Item) -> dict:
    """Every field of the call for one item."""
    system, messages = split_turns(item)
    return {"model": model, "max_tokens": max_tokens_for(model, item), "system": system,
            "messages": messages, **SETTINGS[model].fields}


def count_request(model: str, item: Item) -> dict:
    """The fields of the token count for one item: the call's fields less `max_tokens`."""
    fields = request(model, item)
    del fields["max_tokens"]
    return fields


def _wait(attempt: int) -> float:
    return min(FIRST_WAIT_S * 2 ** attempt, LONGEST_WAIT_S)


def count_tokens(client, model: str, item: Item, sleep: Callable[[float], None] = time.sleep) -> int:
    """The input tokens of one item's call, counted by the provider for the model the call will use."""
    for attempt in range(MAX_ATTEMPTS):
        try:
            return client.messages.count_tokens(**count_request(model, item)).input_tokens
        except anthropic.APIStatusError as e:
            if e.status_code == 400:
                raise SettingRejected(f"{model}: token count for {item.item_id} rejected: {e}") from e
            if (e.status_code != 429 and e.status_code < 500) or attempt == MAX_ATTEMPTS - 1:
                raise
            sleep(_wait(attempt))
        except anthropic.APIConnectionError:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            sleep(_wait(attempt))
    raise AssertionError("unreachable")


def reply_text(content) -> str:
    """The reply's text blocks, joined; thinking blocks are not part of the reply."""
    return "".join(block.text for block in content if block.type == "text")


class ClaudeBackend:
    """A `Backend` over one Claude model, with a small pool of concurrent calls."""

    def __init__(self, model: str, client, spend: Spend, price: Price, workers: int = 8,
                 chunk_size: int = 64, sleep: Callable[[float], None] = time.sleep):
        if model not in SETTINGS:
            raise ValueError(f"no setting for {model}; known: {sorted(SETTINGS)}")
        if price.model != model:
            raise ValueError(f"price of {price.model} given for {model}")
        self.model = model
        self.setting = SETTINGS[model]
        self.name = f"{model} ({self.setting.label})"
        self.client = client
        self.spend = spend
        self.price = price
        self.workers = workers
        self.chunk_size = chunk_size
        self.sleep = sleep

    def generate(self, items: Sequence[Item]) -> Iterable[Reply]:
        """One reply for every item, chunk by chunk, each chunk reserved at its worst case before it is sent.

        Stops with `CapReached` at the first chunk the cap does not allow; nothing of that
        chunk is sent.
        """
        return self._generate(list(items))

    def _generate(self, items: list[Item]) -> Iterator[Reply]:
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for start in range(0, len(items), self.chunk_size):
                chunk = items[start:start + self.chunk_size]
                counts = list(pool.map(lambda it: count_tokens(self.client, self.model, it, self.sleep),
                                       chunk))
                worst = {self._key(it, 0): self.price.worst_case(n, max_tokens_for(self.model, it))
                         for it, n in zip(chunk, counts)}
                self.spend.reserve(worst)
                futures = [pool.submit(self._call, it, n, worst[self._key(it, 0)])
                           for it, n in zip(chunk, counts)]
                failure = None
                for future in as_completed(futures):
                    try:
                        yield future.result()
                    except Exception as e:  # the other calls of the chunk finish and are kept
                        failure = failure or e
                if failure is not None:
                    raise failure

    @staticmethod
    def _key(item: Item, attempt: int) -> str:
        return f"{item.item_id}@{attempt}"

    def _call(self, item: Item, input_tokens: int, worst: Decimal) -> Reply:
        fields = request(self.model, item)
        for attempt in range(MAX_ATTEMPTS):
            key = self._key(item, attempt)
            if attempt > 0:
                self.spend.reserve({key: worst})
            try:
                response = self.client.messages.create(**fields)
            except anthropic.APIStatusError as e:
                if e.status_code == 400:
                    self.spend.release(key)  # rejected before it ran: not billed
                    raise SettingRejected(f"{self.model}: request for {item.item_id} rejected: "
                                          f"{e}") from e
                if e.status_code == 429:
                    self.spend.release(key)  # rejected before it ran: not billed
                elif e.status_code == 408 or e.status_code >= 500:
                    # The call may have run before it failed, so it is counted at its worst case.
                    self.spend.settle(key, worst, self.model)
                else:
                    self.spend.release(key)
                    raise
            except anthropic.APIConnectionError:
                self.spend.settle(key, worst, self.model)  # may have run: counted at its worst case
            except BaseException:
                self.spend.settle(key, worst, self.model)  # unknown outcome: counted at its worst case
                raise
            else:
                return self._reply(item, key, response, worst, attempt + 1)
            if attempt < MAX_ATTEMPTS - 1:
                self.sleep(_wait(attempt))
        raise RuntimeError(f"{self.model}: {item.item_id} failed {MAX_ATTEMPTS} times")

    def _reply(self, item: Item, key: str, response, worst: Decimal, attempts: int) -> Reply:
        usage = response.usage
        cache_write = usage.cache_creation_input_tokens or 0
        cache_read = usage.cache_read_input_tokens or 0
        cost = self.price.cost(usage.input_tokens, usage.output_tokens, cache_write, cache_read)
        self.spend.settle(key, cost, self.model, usage.input_tokens, usage.output_tokens)
        stop = response.stop_reason
        details = getattr(response, "stop_details", None)
        detail = {
            "model": self.model, "model_served": response.model, "thinking": self.setting.label,
            "stop_reason": stop,
            "refusal_category": getattr(details, "category", None) if stop == "refusal" else None,
            "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
            "cache_write_tokens": cache_write, "cache_read_tokens": cache_read,
            "max_tokens": max_tokens_for(self.model, item), "cost_usd": str(cost),
            "worst_case_usd": str(worst), "attempts": attempts,
            "request_id": getattr(response, "_request_id", None),
            "thinking_blocks": sum(1 for b in response.content if b.type == "thinking"),
        }
        return Reply(item_id=item.item_id, text=reply_text(response.content), ended=stop == "end_turn",
                     refused=stop == "refusal", detail=detail)
