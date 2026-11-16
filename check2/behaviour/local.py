"""A Hugging Face chat model on the local GPU, answering items with nothing added to it.

The model runs in bf16 with greedy decoding. Each conversation is rendered with the model's
own chat template, with thinking switched off where the template has the `enable_thinking`
switch. Items are generated in groups of one reply cap, the smallest cap first; within a
group they are sorted by prompt length, ties broken by item id, and cut into left-padded
batches. A reply ended when the model produced one of its end-of-turn tokens inside the cap.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator, Sequence

from check2.behaviour.backend import Item, Reply
from check2.data.sampling import BATCH_EPISODES, left_padded, split_reply

THINKING_SWITCH = "enable_thinking"


def has_thinking_switch(tokenizer) -> bool:
    """Whether the model's chat template reads the `enable_thinking` switch."""
    template = tokenizer.chat_template
    if isinstance(template, dict):
        template = " ".join(template.values())
    return THINKING_SWITCH in (template or "")


def end_of_turn_ids(model, tokenizer) -> list[int]:
    """The token ids that end the model's turn: its generation config's end tokens, else the tokenizer's."""
    ids = model.generation_config.eos_token_id
    if ids is None:
        ids = tokenizer.eos_token_id
    ids = [ids] if isinstance(ids, int) else list(ids)
    if not ids:
        raise ValueError("the model names no end-of-turn token")
    return sorted(set(ids))


def first_end(new_ids: list[int], end_ids: Sequence[int]) -> tuple[list[int], bool]:
    """The reply's tokens before the first end-of-turn token, and whether one was generated."""
    best: tuple[list[int], bool] = (list(new_ids), False)
    for end_id in end_ids:
        reply, ended = split_reply(new_ids, end_id)
        if ended and (not best[1] or len(reply) < len(best[0])):
            best = (reply, ended)
    return best


def plan(items: Sequence[Item], lengths: Sequence[int], batch_size: int) -> list[list[int]]:
    """Batches of item positions: by reply cap, smallest first; by prompt length, then item id."""
    order = sorted(range(len(items)), key=lambda i: (items[i].max_new_tokens, lengths[i], items[i].item_id))
    batches: list[list[int]] = []
    for i in order:
        if batches and len(batches[-1]) < batch_size and \
                items[batches[-1][0]].max_new_tokens == items[i].max_new_tokens:
            batches[-1].append(i)
        else:
            batches.append([i])
    return batches


class LocalBackend:
    """A `Backend` over a loaded causal language model and its tokenizer."""

    def __init__(self, model, tokenizer, name: str, batch_size: int = BATCH_EPISODES, log=print):
        self.model = model
        self.tokenizer = tokenizer
        self.name = name
        self.batch_size = batch_size
        self.thinking_switch = has_thinking_switch(tokenizer)
        self.end_ids = end_of_turn_ids(model, tokenizer)
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.end_ids[0]
        self.log = log

    @property
    def thinking(self) -> str:
        return "off" if self.thinking_switch else "no switch in the template"

    def render(self, item: Item) -> str:
        extra = {THINKING_SWITCH: False} if self.thinking_switch else {}
        return self.tokenizer.apply_chat_template(list(item.messages), tokenize=False,
                                                  add_generation_prompt=True, **extra)

    def encode(self, item: Item) -> list[int]:
        # The template writes every special token itself, so the tokenizer adds none.
        return self.tokenizer(self.render(item), add_special_tokens=False)["input_ids"]

    def _generate_batch(self, prompts: list[list[int]], cap: int) -> list[list[int]]:
        import torch

        ids, mask = left_padded(prompts, self.pad_id)
        device = next(self.model.parameters()).device
        with torch.inference_mode():
            out = self.model.generate(input_ids=ids.to(device), attention_mask=mask.to(device),
                                      do_sample=False, max_new_tokens=cap, repetition_penalty=1.0,
                                      temperature=None, top_p=None, top_k=None,
                                      eos_token_id=self.end_ids, pad_token_id=self.pad_id)
        return [row[ids.shape[1]:].tolist() for row in out]

    def _generate_rows(self, prompts: list[list[int]], cap: int) -> list[list[int]]:
        """Generate a batch; on running out of memory, generate its two halves instead."""
        import torch

        try:
            return self._generate_batch(prompts, cap)
        except torch.OutOfMemoryError:
            if len(prompts) == 1:
                raise
            torch.cuda.empty_cache()
            half = len(prompts) // 2
            self.log(f"out of memory at {len(prompts)} rows; splitting the batch")
            return self._generate_rows(prompts[:half], cap) + self._generate_rows(prompts[half:], cap)

    def generate_batches(self, items: Sequence[Item]) -> Iterator[list[Reply]]:
        """The replies of each batch, batch by batch, in the order of `plan`."""
        encoded = [self.encode(item) for item in items]
        batches = plan(items, [len(e) for e in encoded], self.batch_size)
        for number, batch in enumerate(batches, 1):
            cap = items[batch[0]].max_new_tokens
            start = time.monotonic()
            rows = self._generate_rows([encoded[i] for i in batch], cap)
            replies = []
            for i, new_ids in zip(batch, rows):
                reply_ids, ended = first_end(new_ids, self.end_ids)
                text = self.tokenizer.decode(reply_ids, skip_special_tokens=True)
                replies.append(Reply(item_id=items[i].item_id, text=text, ended=ended, refused=False,
                                     detail={"prompt_tokens": len(encoded[i]), "reply_tokens": len(reply_ids)}))
            self.log(f"batch {number}/{len(batches)}: {len(batch)} items, cap {cap}, "
                     f"longest prompt {max(len(encoded[i]) for i in batch)} tokens, "
                     f"{time.monotonic() - start:.1f} s")
            yield replies

    def generate(self, items: Sequence[Item]) -> Iterable[Reply]:
        for replies in self.generate_batches(items):
            yield from replies


def load(repo_id: str, revision: str, batch_size: int = BATCH_EPISODES) -> LocalBackend:
    """The model at one revision in bf16 on the GPU, with its tokenizer padding on the left."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(repo_id, revision=revision, dtype=torch.bfloat16,
                                                 device_map="cuda")
    return LocalBackend(model.eval(), tokenizer, name=f"{repo_id}@{revision}", batch_size=batch_size)
