"""Checkpoints: their names, saving and loading them, and the one a run loads.

A training is named by the settings that distinguish it from every other training and by
nothing else, so a training and every run that loads it compute the same name from their
configs. Connector training has no adapter, so the adapter's rank and `beta` are no part
of a connector's name; the adapter-only control has no writers and no KL term, so its
signal and `beta` are no part of its name.

An epoch's checkpoint is a folder `<root>/<training-id>/epoch<k>/` holding the writers'
state, the adapter's factors, the optimiser, the schedule, the generators and the step
count. Its last file is a small marker naming the training, the epoch and every other
file's size; a folder without it, or whose files do not match it, is never loaded. On a
session the root is `/content/checkpoints`, and each checkpoint is copied to the Drive
folder under `checkpoints/<training-id>/epoch<k>/` when its epoch ends, marker last, and
fetched from there when it is missing on the session's disk.

The epoch a run loads is the one the checkpoint rule chose, which the file
`data/trainings/<training-id>/chosen.json` names as `{"training_id": ..., "epoch": k, "record": ...}`;
a file that names another training, or none, is refused.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from check2.config.schema import WIDE_WRITER_INPUT, Config
from check2.training.drive import Rclone, local_listing

ROOT = Path(__file__).resolve().parents[2]
CHOSEN_ROOT = ROOT / "data" / "trainings"
CHOSEN_FILE = "chosen.json"
MARKER = "complete.json"
PART_FILES = ("writers", "adapter", "optimizer", "progress")
SESSION_ROOT = Path("/content/checkpoints")
SESSION_DRIVE_FOLDER = "checkpoints"
SCRATCH_ROOT = Path("/content/scratch/checkpoints")
SCRATCH_DRIVE_FOLDER = "scratch/checkpoints"
FORMING_NAMES = {"none": "", "formed_with_channel": "-formed"}


class CheckpointError(RuntimeError):
    """A checkpoint that is missing, incomplete, or belongs to another training or epoch."""


@dataclass(frozen=True)
class Storage:
    """Where checkpoints live: a folder on this disk, and the Drive folder each is copied to.

    Without a Drive copier, checkpoints stay on this disk; that is for the tiny model's runs
    on this machine, never for a session.
    """

    local_root: Path
    drive_folder: str | None = None
    drive: object | None = None

    def folder(self, training_id: str, epoch: int) -> Path:
        return self.local_root / training_id / f"epoch{epoch}"

    def drive_path(self, training_id: str, epoch: int | None = None) -> str | None:
        if self.drive is None:
            return None
        base = f"{self.drive_folder}/{training_id}"
        return base if epoch is None else f"{base}/epoch{epoch}"


def _rclone_storage(local_root: Path, drive_folder: str) -> Storage:
    drive = Rclone()
    if not drive.available():
        raise CheckpointError(f"rclone or the Drive key file is missing ({drive.binary}, {drive.key_file}); "
                              "a checkpoint on a session is always copied to the Drive folder")
    return Storage(local_root, drive_folder, drive)


def session_storage() -> Storage:
    """The checkpoints of record: `/content/checkpoints` on the session, `checkpoints/` on Drive."""
    return _rclone_storage(SESSION_ROOT, SESSION_DRIVE_FOLDER)


def scratch_storage(tag: str) -> Storage:
    """Development checkpoints, apart from those of record, under `/content/scratch/checkpoints/<tag>` and
    `scratch/checkpoints/<tag>/` on Drive; the tag, the commit a run was sent at, keeps two runs' checkpoints apart."""
    if not tag or "/" in tag or tag in (".", ".."):
        raise CheckpointError(f"{tag!r} is not a folder name")
    return _rclone_storage(SCRATCH_ROOT / tag, f"{SCRATCH_DRIVE_FOLDER}/{tag}")


# Names.

def trains_writers(config: Config) -> bool:
    """Whether the config's model carries trained writers: consumer, or no adapter with the write on."""
    kind = config.adapter.kind
    return kind == "consumer" or (kind == "none" and config.writer.alpha != 0)


def trains_adapter(config: Config) -> bool:
    return config.adapter.kind != "none"


def has_trained_part(config: Config) -> bool:
    return trains_writers(config) or trains_adapter(config)


def _signal_name(config: Config) -> str:
    # A swapped signal is an evaluation of the training made with the reader's signal. Writers that
    # also receive the model's state at the read token are a training of their own.
    name = "reader" if config.signal == "swapped" else config.signal
    return f"{name}-wide" if config.writer.input == WIDE_WRITER_INPUT else name


def _forming_name(config: Config) -> str:
    if config.signal_forming not in FORMING_NAMES:
        raise CheckpointError(f"signal_forming {config.signal_forming!r} has no trained writers or adapter")
    return FORMING_NAMES[config.signal_forming]


def connector_id(config: Config) -> str:
    """The name of the connector training a config's writers come from, or continue from."""
    return f"connector-{_signal_name(config)}-seed{config.seed}{_forming_name(config)}"


def _answer_name(config: Config) -> str:
    weight = config.training.answer_token_weight
    return f"-answer{weight:g}" if weight != 0 else ""


def training_id(config: Config) -> str:
    """The name of the training whose checkpoints a config trains or loads.

    `connector-<signal>-seed<s>`, `consumer-<signal>-rank<r>-beta<b>-seed<s>` or
    `adapter-only-rank<r>-seed<s>`, followed by `-formed` for a training on the formed model.
    The signal is `reader-wide` for writers that also receive the model's state at the read token.
    A consumer training whose answer token carries a weight of its own ends in `-answer<w>`.
    """
    if not has_trained_part(config):
        raise CheckpointError("the config has no trained part: no adapter, and the write is off")
    kind = config.adapter.kind
    if kind == "none":
        return connector_id(config)
    forming = _forming_name(config)
    if kind == "consumer":
        return (f"consumer-{_signal_name(config)}-rank{config.adapter.rank}-beta{config.adapter.beta:g}"
                f"-seed{config.seed}{forming}{_answer_name(config)}")
    if kind == "adapter_only":
        return f"adapter-only-rank{config.adapter.rank}-seed{config.seed}{forming}"
    raise CheckpointError(f"unknown adapter kind {kind!r}")


# Saving and loading one epoch.

def _marker(folder: Path) -> dict | None:
    path = folder / MARKER
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _complete(folder: Path, training_id: str, epoch: int) -> bool:
    """Whether a folder holds a complete checkpoint of this training and epoch: marker present, files as it says."""
    marker = _marker(folder)
    if marker is None or marker.get("training_id") != training_id or marker.get("epoch") != epoch:
        return False
    present = {p: s for p, s in local_listing(folder).items() if p != MARKER}
    return present == marker.get("files")


def save(training_id: str, epoch: int, state: dict, storage: Storage | None = None) -> Path:
    """Write an epoch's checkpoint, marker last, then copy it to Drive, marker last; returns its folder.

    `state` holds `writers` (a state dict or None), `adapter` (`{module: (A, B)}` or None),
    `optimizer`, `schedule`, `generators` and `step`. A complete checkpoint is never
    overwritten.
    """
    storage = storage if storage is not None else session_storage()
    missing = [k for k in ("writers", "adapter", "optimizer", "schedule", "generators", "step") if k not in state]
    if missing:
        raise CheckpointError(f"the state lacks {missing}")
    folder = storage.folder(training_id, epoch)
    if _complete(folder, training_id, epoch):
        raise CheckpointError(f"{folder} already holds a complete checkpoint; it is not overwritten")
    folder.mkdir(parents=True, exist_ok=True)
    stale = folder / MARKER
    if stale.exists():
        stale.unlink()
    # Everything but the three large parts goes in the progress file, with the name and epoch it belongs to.
    progress = {k: v for k, v in state.items() if k not in ("writers", "adapter", "optimizer")}
    parts = {"writers": state["writers"], "adapter": state["adapter"], "optimizer": state["optimizer"],
             "progress": {**progress, "step": int(state["step"]), "epoch": int(epoch), "training_id": training_id}}
    for name in PART_FILES:
        if parts[name] is not None:
            torch.save(parts[name], folder / f"{name}.pt")
    files = {p: s for p, s in local_listing(folder).items() if p != MARKER}
    marker = {"training_id": training_id, "epoch": int(epoch), "step": int(state["step"]), "files": files}
    (folder / MARKER).write_text(json.dumps(marker, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    copy_to_drive(training_id, epoch, storage)
    return folder


def copy_to_drive(training_id: str, epoch: int, storage: Storage) -> None:
    """Copy a complete local checkpoint to Drive: its files, then its marker, each checked for size."""
    if storage.drive is None:
        return
    folder = storage.folder(training_id, epoch)
    if not _complete(folder, training_id, epoch):
        raise CheckpointError(f"{folder} is not a complete checkpoint, so it is not copied to Drive")
    path = storage.drive_path(training_id, epoch)
    storage.drive.put(folder, path, exclude=(MARKER,))
    storage.drive.put(folder, path)


def _fetch(training_id: str, epoch: int, storage: Storage) -> None:
    """Copy a complete checkpoint from Drive to this disk, its marker last."""
    path = storage.drive_path(training_id, epoch)
    listing = storage.drive.listing(path)
    if listing is None or MARKER not in listing:
        raise CheckpointError(f"{training_id} epoch {epoch}: no complete checkpoint on this disk or on Drive")
    folder = storage.folder(training_id, epoch)
    stale = folder / MARKER
    if stale.exists():
        stale.unlink()
    storage.drive.get(path, folder, exclude=(MARKER,))
    storage.drive.get(path, folder)


def load(training_id: str, epoch: int, storage: Storage | None = None) -> dict:
    """An epoch's checkpoint as saved, fetched from Drive first when this disk lacks a complete one."""
    storage = storage if storage is not None else session_storage()
    folder = storage.folder(training_id, epoch)
    if not _complete(folder, training_id, epoch):
        if storage.drive is None:
            raise CheckpointError(f"{folder} holds no complete checkpoint of {training_id} epoch {epoch}")
        _fetch(training_id, epoch, storage)
        if not _complete(folder, training_id, epoch):
            raise CheckpointError(f"{folder}: the checkpoint fetched from Drive does not match its marker")
    loaded = {name: torch.load(folder / f"{name}.pt", map_location="cpu", weights_only=True)
              if (folder / f"{name}.pt").exists() else None for name in PART_FILES}
    progress = loaded.pop("progress")
    if progress is None or progress.get("training_id") != training_id or progress.get("epoch") != epoch:
        raise CheckpointError(f"{folder}: the progress file names another training or epoch")
    return {**{k: v for k, v in progress.items() if k not in ("training_id", "epoch")}, **loaded}


def complete_epochs(training_id: str, storage: Storage) -> list[int]:
    """The epochs with a complete checkpoint on this disk or on Drive, in order."""
    found = set()
    base = storage.local_root / training_id
    if base.is_dir():
        for folder in base.iterdir():
            if folder.name.startswith("epoch") and folder.name[5:].isdigit():
                if _complete(folder, training_id, int(folder.name[5:])):
                    found.add(int(folder.name[5:]))
    if storage.drive is not None:
        listing = storage.drive.listing(storage.drive_path(training_id)) or {}
        for path in listing:
            head, _, name = path.partition("/")
            if name == MARKER and head.startswith("epoch") and head[5:].isdigit():
                found.add(int(head[5:]))
    return sorted(found)


# What a run loads.

def chosen_epoch(training_id: str) -> int:
    """The epoch the checkpoint rule kept for a training; an absent file, or one written for another training or
    naming none, is an error."""
    path = CHOSEN_ROOT / training_id / CHOSEN_FILE
    if not path.is_file():
        raise CheckpointError(f"{path} does not exist; the checkpoint rule has not chosen an epoch of {training_id}")
    chosen = json.loads(path.read_text(encoding="utf-8"))
    named = chosen.get("training_id")
    if named is None:
        raise CheckpointError(f"{path} names no training, so it cannot give the epoch of {training_id!r}")
    if named != training_id:
        raise CheckpointError(f"{path} names the training {named!r}, not {training_id!r}")
    epoch = chosen.get("epoch")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1:
        raise CheckpointError(f"{path} names no epoch")
    return epoch


def load_for_run(config: Config, epoch: int | None = None,
                 storage: Storage | None = None) -> tuple[dict | None, dict | None]:
    """The writers' state and the adapter's factors a run of this config needs, at `epoch` or the chosen one.

    `(None, None)` for a config with no trained part. A part the training does not have is None.
    """
    if not has_trained_part(config):
        return None, None
    name = training_id(config)
    state = load(name, epoch if epoch is not None else chosen_epoch(name), storage)
    return state["writers"], state["adapter"]
