"""Forward batches sorted by length and sized to a token budget."""

from __future__ import annotations

import pytest

from check2.training.batches import BatchError, plan_batches
from check2.training.examples import TrainingExample


def _example(name: str, prompt: int, reply: int) -> TrainingExample:
    return TrainingExample(episode_id=name, format="short", prompt_ids=tuple(range(prompt)),
                           target_ids=tuple(range(reply)) + (99,), read_index=prompt - 9)


def _width(batch) -> int:
    return len(batch) * max(len(e.sequence_ids) for e in batch)


def test_batches_are_sorted_by_length_and_within_the_budget():
    examples = [_example(f"e{i}", 20 + 3 * (i % 5), 10 + 7 * i) for i in range(12)]
    batches = plan_batches(examples, 400)
    flat = [e for b in batches for e in b]
    assert sorted(flat, key=lambda e: e.episode_id) == sorted(examples, key=lambda e: e.episode_id)
    lengths = [len(e.sequence_ids) for e in flat]
    assert lengths == sorted(lengths)
    assert all(_width(b) <= 400 for b in batches)
    # Each batch is as full as the budget allows: the next episode would not have fitted.
    for b, following in zip(batches, batches[1:]):
        assert _width(b + following[:1]) > 400


def test_the_row_limit_and_the_form_s_count_are_applied():
    examples = [_example(f"e{i:02d}", 30, 5) for i in range(16)]
    batches = plan_batches(examples, float("inf"), max_rows=8)
    assert [len(b) for b in batches] == [8, 8]
    assert [len(b) for b in plan_batches(examples, 4 * 4, batch_tokens=lambda b: 4 * len(b))] == [4, 4, 4, 4]


def test_an_episode_above_the_budget_alone_is_refused():
    with pytest.raises(BatchError):
        plan_batches([_example("long", 300, 200)], 100)
