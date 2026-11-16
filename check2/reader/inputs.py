"""What a reader fit reads: the `train` and `heldout` records and their cached states on one device."""
from dataclasses import dataclass
from pathlib import Path

import torch

from check2.reader.cache import StateBank, read_meta


@dataclass
class Inputs:
    train: list[dict]
    heldout: list[dict]
    bank: StateBank
    bank_layers: list[int]

    @property
    def label_of(self) -> dict[str, str]:
        return {e["episode_id"]: e["situation"] for e in self.train + self.heldout}

    @property
    def train_ids(self) -> list[str]:
        return sorted(e["episode_id"] for e in self.train)

    @property
    def heldout_ids(self) -> list[str]:
        return sorted(e["episode_id"] for e in self.heldout)


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_inputs(cache_dir: str | Path, train: list[dict], heldout: list[dict],
                device: str | None = None) -> Inputs:
    """The records and a bank of their cached states, on `device` (the GPU when there is one)."""
    ids = sorted(e["episode_id"] for e in train + heldout)
    bank = StateBank.from_cache(cache_dir, ids, device or default_device())
    return Inputs(train=train, heldout=heldout, bank=bank, bank_layers=read_meta(cache_dir)["layers"])
