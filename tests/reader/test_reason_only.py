"""The reason-only prompts, and the reason-only signal computed without saving states."""
import json

import pytest
import torch

from check2.data.pool import Question
from check2.reader.cache import cache_states, load_states
from check2.reader.model import Reader
from check2.reader.reason_only import pairs_on_test, reason_only_items, reason_only_p, write_test_p
from tests.reader.helpers import FakeTokenizer
from tests.tiny_model import tiny_model

LAYERS = [1, 2, 5]


def _pool():
    return {f"q/{i}": Question(qid=f"q/{i}", subject="x", question=f"Q{i}?", choices=("a", "b", "c", "d"),
                               gold="A", a1="B" if i % 2 else "A", free_text="B", complies=True, p1=(0.25,) * 4)
            for i in range(4)}


PAIRS = [
    {"qid": "q/0", "target": "C", "reason": "It is odd so it fits.", "writer": "opus"},
    {"qid": "q/1", "target": "A", "reason": "Because three is the answer here.", "writer": "opus"},
    {"qid": "q/1", "target": "D", "reason": "I recall this one well.", "writer": "opus"},
    {"qid": "q/2", "target": "B", "reason": "The sum is small.", "writer": "opus"},
    {"qid": "q/3", "target": "A", "reason": "Wrong split.", "writer": "opus"},
    {"qid": "q/2", "target": "C", "reason": "Another writer's reason.", "writer": "sonnet"},
]
SPLIT_OF = {"q/0": "train", "q/1": "train", "q/2": "heldout", "q/3": "test"}


def test_one_item_per_pair_of_the_cached_splits():
    tok = FakeTokenizer()
    items = reason_only_items(tok, PAIRS, _pool(), SPLIT_OF, ("train", "heldout"))
    assert [i["episode_id"] for i in items] == ["q/0:C", "q/1:A", "q/1:D", "q/2:B"]
    assert [i["situation"] for i in items] == ["hold", "user_right", "recheck", "hold"]
    assert items[1]["prompt"] == ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                                  "<|im_start|>user\nBecause three is the answer here.<|im_end|>\n"
                                  "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_the_unsaved_pass_gives_what_the_cache_gives(tmp_path):
    tok = FakeTokenizer()
    model = tiny_model()
    items = reason_only_items(tok, PAIRS, _pool(), SPLIT_OF, ("train", "heldout"))
    cache_states(model, tok, items, tmp_path, LAYERS, batch_size=4)
    torch.manual_seed(0)
    reader = Reader([2, 5], 32).eval()
    ids = [i["episode_id"] for i in items]
    for k in (0, 4):
        h, mask = load_states(tmp_path, ids, k)
        from_cache = reader(h[:, [1, 2]], mask)
        live = reason_only_p(model, tok, reader, [p for p in PAIRS if SPLIT_OF[p["qid"]] != "test"
                                                  and p["writer"] == "opus"], k)
        assert sorted(live) == sorted(ids)
        torch.testing.assert_close(torch.tensor([live[i] for i in ids]), from_cache, rtol=1e-5, atol=1e-6)


def test_the_pairs_on_test_are_those_with_a_reason_of_the_main_writer():
    pairs = PAIRS + [{"qid": "q/3", "target": "B", "reason": None, "writer": None}]
    assert [(p["qid"], p["target"]) for p in pairs_on_test(pairs, SPLIT_OF)] == [("q/3", "A")]
    with pytest.raises(ValueError):
        pairs_on_test(pairs + [{"qid": "q/3", "target": "A", "reason": "Again.", "writer": "opus"}], SPLIT_OF)


def test_the_saved_test_p_is_the_unsaved_pass_written_as_the_other_saved_outputs(tmp_path):
    tok = FakeTokenizer()
    model = tiny_model()
    torch.manual_seed(0)
    reader = Reader([2, 5], 32).eval()
    pairs = [p for p in PAIRS if p["writer"] == "opus"]
    output = tmp_path / "p_reason_only_test.jsonl"
    written = write_test_p(model, tok, reader, pairs, 0, output)
    assert written == reason_only_p(model, tok, reader, pairs, 0)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"episode_id": key, "p": written[key]} for key in sorted(written)]
    assert [r["episode_id"] for r in rows] == ["q/0:C", "q/1:A", "q/1:D", "q/2:B", "q/3:A"]
