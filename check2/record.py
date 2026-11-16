"""The record every gate, measurement and run writes, with the facts that make it traceable.

The commit comes from git when the code runs inside a repository, and otherwise from the
one-line file `.sent-commit` at the repository root, which the script that sends a commit
to a session leaves there.
"""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("gate", "measurement", "run")
RUN_STATUSES = ("running", "complete", "failed")
SESSION_VARIABLE = "CHECK2_SESSION"
SENT_COMMIT_FILE = ".sent-commit"
RESULTS = "results"
FAST_KERNEL = ("fla", "flash-linear-attention")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


STARTED = _now()


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                             check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout


def source_state() -> dict:
    """The commit the code was run from, whether the tree matched it, and where that came from."""
    head = _git("rev-parse", "HEAD")
    if head is not None:
        # Records under results/ are outputs, not tested paths, so they never make a tree unclean.
        status = _git("status", "--porcelain", "--", ".", f":(exclude){RESULTS}")
        return {"commit": head.strip(), "tree_clean": status is not None and status.strip() == "",
                "commit_source": "git"}
    sent = ROOT / SENT_COMMIT_FILE
    if sent.exists():
        return {"commit": sent.read_text(encoding="utf-8").strip(), "tree_clean": None,
                "commit_source": SENT_COMMIT_FILE}
    raise RuntimeError(f"no git repository at {ROOT} and no {SENT_COMMIT_FILE} file")


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def versions() -> dict:
    """Library versions, read from package metadata so that nothing is imported to read them."""
    module, distribution = FAST_KERNEL
    return {
        "python": platform.python_version(),
        "torch": _version("torch"),
        "transformers": _version("transformers"),
        distribution: _version(distribution) if importlib.util.find_spec(module) else None,
    }


def gpu_name() -> str | None:
    import torch

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None


def _config_fields(config) -> dict:
    if config is None:
        return {"config": None, "config_sha256": None}
    if isinstance(config, Mapping):
        text = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return {"config": dict(config), "config_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    path = Path(config)
    data = path.read_bytes()
    return {"config": str(path), "config_sha256": hashlib.sha256(data).hexdigest()}


def _common(config) -> dict:
    return {**source_state(), "session": os.environ.get(SESSION_VARIABLE), "gpu": gpu_name(),
            "versions": versions(), **_config_fields(config)}


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _write(path: Path, record: dict, replace: bool = False) -> Path:
    """Write a record as JSON; an existing record is replaced only where `replace` says so."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    with path.open("w" if replace else "x", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return path


def write_record(kind: str, name: str, values: dict, verdict: str | None = None, config=None,
                 extra: dict | None = None) -> Path:
    """Write one record and return its path.

    A gate goes to `results/gates/<name>/`, a measurement to `results/measurements/<name>/`,
    and a run's other records to `results/<name>/`, each as `<utc>-<commit>.json`.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
    record = {"kind": kind, "name": name, **_common(config), "started": STARTED, "finished": _now(),
              "values": values, "verdict": verdict, **(extra or {})}
    folder = {"gate": ROOT / RESULTS / "gates", "measurement": ROOT / RESULTS / "measurements",
              "run": ROOT / RESULTS}[kind] / name
    return _write(folder / f"{_stamp()}-{record['commit'][:12]}.json", record)


def _config_value(config, *keys):
    if not isinstance(config, Mapping):
        return None
    node = config.get("check2", config)
    for key in keys:
        if not isinstance(node, Mapping) or key not in node:
            return None
        node = node[key]
    return node


def start_run(run_id: str, config) -> Path:
    """Write `results/<run-id>/run.json` with the status `running`; returns its path."""
    record = {"run_id": run_id, "kind": "run", **_common(config),
              "split": _config_value(config, "data", "eval_split"), "seed": _config_value(config, "seed"),
              "order_sha256": None, "started": _now(), "finished": None, "episodes_expected": None,
              "episodes_written": None, "status": "running"}
    return _write(ROOT / RESULTS / run_id / "run.json", record)


def finish_run(path: Path, status: str, episodes_expected: int, episodes_written: int,
               order_sha256: str) -> Path:
    """Close a run's record. A run that wrote fewer or more episodes than expected is never complete."""
    if status not in RUN_STATUSES[1:]:
        raise ValueError(f"a run finishes as one of {RUN_STATUSES[1:]}, not {status!r}")
    if status == "complete" and episodes_written != episodes_expected:
        raise ValueError(f"a run that wrote {episodes_written} of {episodes_expected} episodes "
                         f"is not complete")
    path = Path(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    record.update(status=status, finished=_now(), episodes_expected=episodes_expected,
                  episodes_written=episodes_written, order_sha256=order_sha256)
    return _write(path, record, replace=True)
