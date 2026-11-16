"""Plain copies between a session's disk and the Drive folder, each checked file by file for size.

A session gets rclone and the campaign key file from its set-up; the key file defines the
remote `check2drive`, rooted at the Drive folder. Nothing here moves, syncs or deletes
anything on Drive. A copy is followed by a listing of both sides, and every copied file
must have the same size on each.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

# Where a session's set-up places rclone and the key file, and the remote the key file defines.
RCLONE_BIN = "/usr/local/bin/rclone"
KEY_FILE = "/root/drive-key/rclone.conf"
DRIVE_REMOTE = "check2drive"


class DriveError(RuntimeError):
    """A copy that failed, or whose sizes do not match on both sides."""


def local_listing(path: Path) -> dict[str, int]:
    """`{relative path: size}` of the files under a folder, with `/` as the separator."""
    path = Path(path)
    out = {}
    for root, _dirs, files in os.walk(path):
        for name in files:
            full = Path(root) / name
            if full.is_file() and not full.is_symlink():
                out[full.relative_to(path).as_posix()] = full.stat().st_size
    return out


def size_mismatches(copied: dict[str, int], other: dict[str, int]) -> list[str]:
    """The copied files that are missing on the other side or have another size there."""
    return [path for path, size in sorted(copied.items()) if other.get(path) != size]


def _check(copied: dict[str, int], other: dict[str, int], where: str) -> None:
    bad = size_mismatches(copied, other)
    if bad:
        detail = ", ".join(f"{p}: {copied[p]} bytes, {other.get(p)} on {where}" for p in bad[:5])
        raise DriveError(f"size check failed for {len(bad)} of {len(copied)} files ({detail})")


class Rclone:
    """The Drive folder through rclone and the key file on a session."""

    def __init__(self, binary: str = RCLONE_BIN, key_file: str = KEY_FILE, remote: str = DRIVE_REMOTE) -> None:
        self.binary, self.key_file, self.remote = binary, key_file, remote

    def available(self) -> bool:
        return os.path.exists(self.binary) and os.path.exists(self.key_file)

    def _target(self, drive_path: str) -> str:
        parts = drive_path.split("/")
        if drive_path.startswith("/") or any(p in ("", ".", "..") for p in parts):
            raise DriveError(f"the Drive path {drive_path!r} must be relative to the Drive folder")
        return f"{self.remote}:{drive_path}"

    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([self.binary, *args, "--config", self.key_file], capture_output=True, text=True)

    def listing(self, drive_path: str) -> dict[str, int] | None:
        """`{relative path: size}` of the files under a Drive folder, or None when it does not exist."""
        result = self._run("lsjson", "-R", "--files-only", self._target(drive_path))
        if result.returncode != 0:
            if "directory not found" in result.stderr.lower():
                return None
            raise DriveError(f"rclone lsjson {drive_path} failed: {result.stderr.strip()}")
        return {e["Path"]: e["Size"] for e in json.loads(result.stdout)}

    def put(self, folder: Path, drive_path: str, exclude: tuple[str, ...] = ()) -> None:
        """Copy a folder's files to a Drive folder, leaving out the named files, and check every size."""
        args = ["copy", str(folder), self._target(drive_path)]
        for name in exclude:
            args += ["--exclude", f"/{name}"]
        result = self._run(*args)
        if result.returncode != 0:
            raise DriveError(f"rclone copy to {drive_path} failed: {result.stderr.strip()}")
        copied = {p: s for p, s in local_listing(folder).items() if p not in exclude}
        _check(copied, self.listing(drive_path) or {}, "Drive")

    def get(self, drive_path: str, folder: Path, exclude: tuple[str, ...] = ()) -> None:
        """Copy a Drive folder's files into a local folder, leaving out the named files, and check every size."""
        listing = self.listing(drive_path)
        if listing is None:
            raise DriveError(f"{drive_path} does not exist on Drive")
        Path(folder).mkdir(parents=True, exist_ok=True)
        args = ["copy", self._target(drive_path), str(folder)]
        for name in exclude:
            args += ["--exclude", f"/{name}"]
        result = self._run(*args)
        if result.returncode != 0:
            raise DriveError(f"rclone copy from {drive_path} failed: {result.stderr.strip()}")
        _check({p: s for p, s in listing.items() if p not in exclude}, local_listing(folder), "this disk")
