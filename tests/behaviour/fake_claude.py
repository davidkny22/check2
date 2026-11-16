"""A stand-in for the provider's client: it records every request and answers from a script."""

from __future__ import annotations

import threading
from types import SimpleNamespace

from check2.behaviour.backend import Item

SYSTEM = "You are a helpful assistant."


def item(item_id: str, max_new_tokens: int = 16, user: str = "Which? (A) a (B) b (C) c (D) d") -> Item:
    return Item(item_id=item_id, qid=item_id.split("#")[0], source="pool", kind="first", fmt=None,
                target=None, writer=None,
                messages=({"role": "system", "content": SYSTEM}, {"role": "user", "content": user}),
                max_new_tokens=max_new_tokens)


def response(text: str = "B", stop_reason: str = "end_turn", input_tokens: int = 40,
             output_tokens: int = 2, model: str = "claude-haiku-5-5", category: str | None = None):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)] if text else [],
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category=category) if stop_reason == "refusal" else None,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens,
                              cache_creation_input_tokens=None, cache_read_input_tokens=None),
        model=model, _request_id="req_fake")


class FakeMessages:
    def __init__(self, answer, input_tokens: int = 40):
        self.answer = answer  # called with the request's fields; returns a response or raises
        self.input_tokens = input_tokens
        self.created: list[dict] = []
        self.counted: list[dict] = []
        self._lock = threading.Lock()

    def create(self, **fields):
        with self._lock:
            self.created.append(fields)
        return self.answer(fields)

    def count_tokens(self, **fields):
        with self._lock:
            self.counted.append(fields)
        return SimpleNamespace(input_tokens=self.input_tokens)


class FakeClient:
    def __init__(self, answer=lambda fields: response(model=fields["model"]), input_tokens: int = 40):
        self.messages = FakeMessages(answer, input_tokens)
