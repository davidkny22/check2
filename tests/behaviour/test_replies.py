"""A run folder's replies file: appended batch by batch, read back plain or compressed."""

import pytest

from check2.behaviour.backend import Item, Reply
from check2.behaviour.replies import append_replies, compress_replies, read_replies, reply_row


def _item(item_id):
    return Item(item_id=item_id, qid="q/1", source="pool", kind="first", fmt=None, target=None, writer=None,
                messages=({"role": "system", "content": "s"},), max_new_tokens=24)


def test_rows_survive_appending_and_compression(tmp_path):
    folder = tmp_path / "run"
    assert read_replies(folder) == []
    append_replies(folder, [reply_row(_item("a"), Reply("a", "B", True, False))])
    append_replies(folder, [reply_row(_item("b"), Reply("b", "C", False, False, {"tokens": 24}))])
    rows = read_replies(folder)
    assert [(r["item_id"], r["text"], r["ended"], r["detail"]) for r in rows] == [
        ("a", "B", True, {}), ("b", "C", False, {"tokens": 24})]
    compress_replies(folder)
    assert not (folder / "replies.jsonl").exists()
    assert read_replies(folder) == rows


def test_an_item_answered_twice_is_refused(tmp_path):
    folder = tmp_path / "run"
    row = reply_row(_item("a"), Reply("a", "B", True, False))
    append_replies(folder, [row, row])
    with pytest.raises(ValueError, match="answered twice"):
        read_replies(folder)


def test_a_reply_to_another_item_is_refused():
    with pytest.raises(ValueError):
        reply_row(_item("a"), Reply("b", "B", True, False))
