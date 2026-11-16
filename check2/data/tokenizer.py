"""The model's tokenizer, fetched alone, without the model's weights."""
from functools import lru_cache
from pathlib import Path

MODEL_ID = "Qwen/Qwen3.5-4B"
TOKENIZER_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
                   "merges.txt", "chat_template.jinja"]


@lru_cache(maxsize=1)
def tokenizer_folder() -> Path:
    """The local snapshot of the tokenizer's files; its folder name is the repository commit."""
    from huggingface_hub import snapshot_download

    try:
        folder = snapshot_download(MODEL_ID, allow_patterns=TOKENIZER_FILES)
    except Exception:
        folder = snapshot_download(MODEL_ID, allow_patterns=TOKENIZER_FILES, local_files_only=True)
    return Path(folder)


def tokenizer_revision() -> str:
    return tokenizer_folder().name


@lru_cache(maxsize=1)
def get_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(tokenizer_folder()))
