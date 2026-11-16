"""The local backend's batching, padding and end-of-turn handling, on a stub model whose replies are known."""

import torch

from check2.behaviour.backend import Item
from check2.behaviour.local import LocalBackend, first_end, plan

END, OTHER_END, PAD = 1, 2, 0


class StubTokenizer:
    """Characters are tokens: a prompt is the joined turn contents, each character one id."""

    pad_token_id = PAD
    eos_token_id = END

    def __init__(self, template: str):
        self.chat_template = template
        self.rendered_with: list[dict] = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        self.rendered_with.append(kwargs)
        return "|".join(m["content"] for m in messages)

    def __call__(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


class StubModel(torch.nn.Module):
    """Replies with the row's last prompt token, then ends; a prompt ending in "~" never ends."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.generation_config = type("GenerationConfig", (), {"eos_token_id": [OTHER_END, END]})()
        self.calls: list[tuple[torch.Tensor, torch.Tensor, int]] = []

    def generate(self, input_ids, attention_mask, max_new_tokens, eos_token_id, pad_token_id, **kwargs):
        assert kwargs["do_sample"] is False
        self.calls.append((input_ids.clone(), attention_mask.clone(), max_new_tokens))
        rows = []
        for ids in input_ids.tolist():
            last = ids[-1]
            new = [last] * max_new_tokens if last == ord("~") else [last, END] + [PAD] * (max_new_tokens - 2)
            rows.append(ids + new[:max_new_tokens])
        return torch.tensor(rows)


def _item(item_id, text, cap):
    return Item(item_id=item_id, qid=item_id, source="pool", kind="first", fmt=None, target=None, writer=None,
                messages=({"role": "user", "content": text},), max_new_tokens=cap)


def test_the_plan_groups_by_cap_then_sorts_by_length_then_id():
    items = [_item("c", "", 256), _item("b", "", 16), _item("a", "", 16), _item("d", "", 16)]
    assert plan(items, [5, 9, 9, 3], batch_size=2) == [[3, 2], [1], [0]]


def test_first_end_takes_the_earliest_end_token_of_any_kind():
    assert first_end([5, 6, OTHER_END, 7, END], [END, OTHER_END]) == ([5, 6], True)
    assert first_end([5, 6, 7], [END, OTHER_END]) == ([5, 6, 7], False)


def test_every_reply_returns_to_its_own_item_through_padded_batches():
    model = StubModel()
    tokenizer = StubTokenizer("{% if enable_thinking %}{% endif %}")
    backend = LocalBackend(model, tokenizer, name="stub", batch_size=2, log=lambda *_: None)
    items = [_item("x1", "aaaaaQ", 4), _item("x2", "R", 4), _item("x3", "bbS", 4), _item("x4", "cc~", 4)]
    replies = {r.item_id: r for r in backend.generate(items)}
    assert {k: (r.text, r.ended) for k, r in replies.items()} == {
        "x1": ("Q", True), "x2": ("R", True), "x3": ("S", True), "x4": ("~~~~", False)}
    assert backend.thinking == "off" and all(kw == {"enable_thinking": False} for kw in tokenizer.rendered_with)
    first_ids, first_mask, cap = model.calls[0]
    # The two shortest prompts, "R" and "bbS", padded on the left to three tokens.
    assert first_ids.tolist() == [[PAD, PAD, ord("R")], [ord("b"), ord("b"), ord("S")]]
    assert first_mask.tolist() == [[0, 0, 1], [1, 1, 1]] and cap == 4


def test_a_template_without_the_switch_gets_no_thinking_argument():
    tokenizer = StubTokenizer("{{ messages }}")
    backend = LocalBackend(StubModel(), tokenizer, name="stub", log=lambda *_: None)
    list(backend.generate([_item("x", "aQ", 3)]))
    assert backend.thinking == "no switch in the template" and tokenizer.rendered_with == [{}]
