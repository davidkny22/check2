"""Made-up inputs for the reader's fast tests: a small tokenizer with the chat format's tokens,
episodes rendered with it, and banks of random states."""
import random
import re

import torch

from check2.reader.cache import StateBank

SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
PATTERN = re.compile("|".join(re.escape(s) for s in SPECIAL) + r"|\n\n|\n|[A-Za-z]+|\d|[^\sA-Za-z\d]| ")


class FakeTokenizer:
    """Special tokens, words, digits, single characters, spaces and line breaks, each one token.

    Ids are given in order of first sight and stay below the tiny model's vocabulary, so a
    prompt in the chat format ends in the same nine-token tail as the model's own template.
    """

    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size
        self.vocab = {t: i for i, t in enumerate(SPECIAL)}
        self.pad_token_id = self.vocab["<|endoftext|>"]
        self.eos_token_id = self.vocab["<|im_end|>"]

    def _id(self, token: str) -> int:
        if token not in self.vocab:
            if len(self.vocab) >= self.vocab_size:
                raise ValueError("the made-up vocabulary is full")
            self.vocab[token] = len(self.vocab)
        return self.vocab[token]

    def _encode(self, text: str) -> list[int]:
        tokens = PATTERN.findall(text)
        if "".join(tokens) != text:
            raise ValueError(f"cannot split {text!r}")
        return [self._id(t) for t in tokens]

    def __call__(self, text, add_special_tokens: bool = True):
        if isinstance(text, str):
            return {"input_ids": self._encode(text)}
        return {"input_ids": [self._encode(t) for t in text]}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._id(token)

    def decode(self, ids) -> str:
        inverse = {i: t for t, i in self.vocab.items()}
        return "".join(inverse[int(i)] for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return text + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def conversation(tokenizer: FakeTokenizer, question: str, first_answer: str, pushback: str) -> str:
    return tokenizer.apply_chat_template([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{question}\n(A) one\n(B) two\n(C) three\n(D) four"},
        {"role": "assistant", "content": first_answer},
        {"role": "user", "content": pushback},
    ])


def made_up_episodes(tokenizer: FakeTokenizer, n: int, split: str = "train", seed: int = 0) -> list[dict]:
    """Episodes of varied length, in both formats, with the chat format's prompts."""
    rng = random.Random(seed)
    episodes = []
    for i in range(n):
        words = " ".join(rng.choice(["sure", "wrong", "because", "three", "is", "odd"])
                         for _ in range(rng.randint(1, 12)))
        fmt = ("snap", "short")[i % 2]
        episodes.append({"episode_id": f"q/{i}:reasoned:{fmt}:A", "qid": f"q/{i}", "split": split,
                         "form": "reasoned", "format": fmt, "situation": "hold",
                         "prompt": conversation(tokenizer, f"What is {i}?", "(B)", f"I think {words}.")})
    return episodes


def random_bank(lengths: dict[str, int], layers: int, d: int, seed: int = 0,
                dtype=torch.float32) -> StateBank:
    g = torch.Generator().manual_seed(seed)
    return StateBank({e: torch.randn(n, layers, d, generator=g).to(dtype) for e, n in lengths.items()})
