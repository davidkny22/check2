"""A stand-in for the Drive folder: a local folder with the copier's interface, recording each copy."""

from __future__ import annotations

import shutil
from pathlib import Path

from check2.training.drive import local_listing


class FolderDrive:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.calls: list[tuple[str, str, tuple[str, ...]]] = []

    def listing(self, drive_path: str) -> dict[str, int] | None:
        folder = self.root / drive_path
        return local_listing(folder) if folder.is_dir() else None

    def put(self, folder: Path, drive_path: str, exclude: tuple[str, ...] = ()) -> None:
        self.calls.append(("put", drive_path, tuple(exclude)))
        target = self.root / drive_path
        for relative in local_listing(folder):
            if relative in exclude:
                continue
            (target / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(folder) / relative, target / relative)

    def get(self, drive_path: str, folder: Path, exclude: tuple[str, ...] = ()) -> None:
        self.calls.append(("get", drive_path, tuple(exclude)))
        source = self.root / drive_path
        for relative in local_listing(source):
            if relative in exclude:
                continue
            (Path(folder) / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, Path(folder) / relative)
