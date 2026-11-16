"""Loading the model and its tokenizer: the pinned full model on a GPU, or the tiny random model here.

The full model is loaded at the revision `experiments/pins.json` holds, after the runtime's
versions are checked against that file. The tiny model of `tests/tiny_model.py` has the full
model's architecture and a vocabulary of a few hundred tokens, so it is paired with the
model's own tokenizer whose ids are folded into that vocabulary: prompts keep their length,
their read index and their template tail, and mean nothing as language.
"""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from check2.config.schema import ADAPTER_LAYERS, MODEL_ID, PINS_FILE, WRITE_BANDS, WRITE_LAYERS

FAST_KERNEL_MODULE = "fla"
DEVELOPMENT_CARD_VARIABLE = "CHECK2_DEVELOPMENT_CARD"
RESERVED_IDS = 3  # the tiny model's pad, beginning and end ids sit below the folded range


class RuntimeMismatch(RuntimeError):
    """The session's libraries or GPU differ from the pinned ones, or the kernel path is not the one asked for."""


def fast_kernel_importable() -> bool:
    return importlib.util.find_spec(FAST_KERNEL_MODULE) is not None


def require_builtin_path() -> None:
    """Refuse to go on in a process in which the fast DeltaNet kernel package can be imported."""
    if fast_kernel_importable():
        raise RuntimeMismatch("the fast DeltaNet kernel package is importable; this check runs on the "
                              "library's built-in path, in a process where it cannot be imported")


@dataclass(frozen=True)
class Layout:
    """Which layers the additions use on a model: write layers, their bands, the adapter's range."""

    write_layers: tuple[int, ...]
    bands: tuple[tuple[int, ...], ...]
    adapter_first: int
    adapter_last: int


FULL_LAYOUT = Layout(tuple(WRITE_LAYERS), tuple(tuple(b) for b in WRITE_BANDS), ADAPTER_LAYERS[0], ADAPTER_LAYERS[1])
# The tiny model has 8 layers, full attention at 3 and 7. Its layout keeps the full one's shape:
# layers before the first write and adapter layer, a write into layer 3, three bands, and an
# adapter reaching the last layer.
TINY_LAYOUT = Layout((1, 2, 3, 5, 6), ((1, 2), (3,), (5, 6)), 1, 7)


class FoldedTokenizer:
    """The model's tokenizer with each id `i` mapped to `RESERVED_IDS + i % (vocab - RESERVED_IDS)`."""

    def __init__(self, base, vocab_size: int) -> None:
        self.base = base
        self.span = vocab_size - RESERVED_IDS
        self.pad_token_id = 0

    def _fold(self, ids: list[int]) -> list[int]:
        return [RESERVED_IDS + i % self.span for i in ids]

    def __call__(self, texts):
        encoded = self.base(texts)["input_ids"]
        if isinstance(texts, str):
            return {"input_ids": self._fold(encoded)}
        return {"input_ids": [self._fold(ids) for ids in encoded]}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._fold(self.base.encode(text, add_special_tokens=add_special_tokens))

    def convert_tokens_to_ids(self, token):
        if isinstance(token, str):
            return self._fold([self.base.convert_tokens_to_ids(token)])[0]
        return self._fold(self.base.convert_tokens_to_ids(token))

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return self.base.decode(list(ids), skip_special_tokens=skip_special_tokens)


def load_tiny(dtype: torch.dtype = torch.float32):
    """The tiny random model on the CPU and the folded tokenizer; `(model, tokenizer, layout)`.

    The CPU's matrix multiplies split their sums by the number of threads free when they
    run, so two identical calls can differ in the last bit on a busy machine. The tiny model
    therefore runs on one thread, where every call repeats exactly.
    """
    from check2.data.tokenizer import get_tokenizer
    from tests.tiny_model import TINY_VOCAB_SIZE, tiny_model

    torch.set_num_threads(1)

    model = tiny_model().to(dtype)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, FoldedTokenizer(get_tokenizer(), TINY_VOCAB_SIZE), TINY_LAYOUT


def read_pins(path: Path = PINS_FILE) -> dict:
    if not path.exists():
        raise RuntimeMismatch(f"{path} does not exist; the full model is loaded only at its pinned revision")
    return json.loads(path.read_text(encoding="utf-8"))


def check_runtime(pins: dict) -> dict:
    """Assert that this process's libraries and GPU are the pinned ones; returns what was compared.

    With the environment variable `CHECK2_DEVELOPMENT_CARD=1` the card's name is not
    compared, so that development work can run on another card; every version still is.
    """
    import transformers
    import triton

    now = {"transformers": transformers.__version__, "torch": torch.__version__, "triton": triton.__version__,
           "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    compared = now
    if os.environ.get(DEVELOPMENT_CARD_VARIABLE) == "1":
        compared = {k: v for k, v in now.items() if k != "gpu"}
        print(f"this process runs on a development card ({now['gpu']}), not the pinned {pins.get('gpu')}")
    differ = {k: (pins.get(k), v) for k, v in compared.items() if pins.get(k) != v}
    if differ:
        raise RuntimeMismatch(f"the runtime differs from the pinned one: {differ}")
    return now


def load_full(dtype: torch.dtype, pins_file: Path = PINS_FILE):
    """The full model at its pinned revision on the GPU, with the tokenizer; `(model, tokenizer, layout)`."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pins = read_pins(pins_file)
    check_runtime(pins)
    revision = pins["model_revision"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=revision, dtype=dtype,
                                                 attn_implementation="sdpa").to("cuda")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer, FULL_LAYOUT
