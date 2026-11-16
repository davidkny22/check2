"""What a model is asked and what it replied, when the pushback episodes are run on a model with nothing added."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol, Sequence


@dataclass(frozen=True)
class Item:
    """One conversation to put to a model, ending on a user turn."""

    item_id: str
    qid: str
    source: str  # "pool" or "expanded"
    kind: str  # "first", "reasoned" or "neutral"
    fmt: str | None  # "snap" or "short"; None for a first answer
    target: str | None  # the option the user pushes toward, where the kind has one
    writer: str | None  # who wrote the reason, where the kind has one
    messages: tuple[dict, ...]  # role and content of each turn, the system turn first
    max_new_tokens: int


@dataclass(frozen=True)
class Reply:
    """A model's reply to one item."""

    item_id: str
    text: str
    ended: bool  # the model ended its turn inside the cap
    refused: bool  # the provider declined to answer
    detail: dict = field(default_factory=dict)  # what the backend recorded: stop reason, token counts


class Backend(Protocol):
    """A model that answers conversations. `name` identifies the model and its settings in a record."""

    name: str

    def generate(self, items: Sequence[Item]) -> Iterable[Reply]:
        """One reply for every item, in any order."""
        ...
