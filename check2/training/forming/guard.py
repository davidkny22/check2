"""The guard on saved reader outputs: they are used only with the model they were computed from.

Reader outputs computed from the formed model are saved in a folder that holds
`source.json`: the forming training's id and the SHA-256 of the update's factors. Outputs of
the original model carry no such record. Every loader of saved reader outputs, and of a
fitted reader, goes through `check_source`:

- with a forming update loaded, outputs load only from a folder whose record matches the
  loaded update's factors, and outputs with no record (the original model's) raise;
- with no forming update loaded, outputs that carry a record raise.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from check2.reader import files
from check2.training.forming.update import factors_sha256

SOURCE_FILE = "source.json"

Factors = dict[str, tuple[torch.Tensor, torch.Tensor]]


class StaleSignal(RuntimeError):
    """Saved reader outputs that do not come from the model they are about to be used with."""


def write_source(folder: str | Path, training_id: str, factors: Factors) -> Path:
    """Record in `folder` which forming update its reader outputs were computed from."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / SOURCE_FILE
    record = {"training_id": training_id, "factors_sha256": factors_sha256(factors)}
    path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return path


def read_source(folder: str | Path) -> dict | None:
    path = Path(folder) / SOURCE_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def check_source(folder: str | Path, factors: Factors | None) -> dict | None:
    """Raise `StaleSignal` unless the outputs in `folder` come from the model the factors describe.

    `factors` are the loaded forming update's, or None when no forming update is loaded.
    Returns the folder's record, or None for the original model's outputs.
    """
    source = read_source(folder)
    if factors is None:
        if source is not None:
            raise StaleSignal(f"{folder} holds reader outputs of a formed model ({source['training_id']}), "
                              "and no forming update is loaded")
        return None
    if source is None:
        raise StaleSignal(f"{folder} holds reader outputs saved from the original model, and a forming update "
                          "is loaded; compute them again from the formed model")
    if source["factors_sha256"] != factors_sha256(factors):
        raise StaleSignal(f"{folder} holds reader outputs of the formed model of {source['training_id']} "
                          f"(factors {source['factors_sha256'][:12]}), not of the loaded update")
    return source


def load_p(path: str | Path, factors: Factors | None) -> dict[str, list[float]]:
    """A saved file of `{episode_id, p}` rows, after the guard."""
    path = Path(path)
    check_source(path.parent, factors)
    return files.read_p(path)


def load_fitted_reader(path: str | Path, factors: Factors | None):
    """A saved fitted reader, on the CPU in evaluation mode, after the guard."""
    from check2.reader.crossfit import load_reader

    path = Path(path)
    check_source(path.parent, factors)
    return load_reader(path)
