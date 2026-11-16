"""The original model and its tokenizer at the pinned revision, in bf16 with sdpa attention."""
from check2.data import paths
from check2.data.tokenizer import MODEL_ID
from check2.reader.files import read_json

PINS = paths.ROOT / "experiments" / "pins.json"


def pinned_revision() -> str:
    if not PINS.exists():
        raise FileNotFoundError(f"{PINS} is missing: the session's model revision is not pinned")
    pins = read_json(PINS)
    if pins.get("model") != MODEL_ID:
        raise ValueError(f"{PINS} pins {pins.get('model')}, not {MODEL_ID}")
    return pins["model_revision"]


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_ID, revision=pinned_revision())


def load_original(device: str = "cuda"):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=pinned_revision(), dtype=torch.bfloat16,
                                                 attn_implementation="sdpa", device_map=device)
    return model.eval()
