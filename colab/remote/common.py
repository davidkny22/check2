"""Definitions shared by the helpers that run on a session.

The local scripts send each helper with this file placed in front of it, so a helper uses these
names without importing anything. Settings arrive as environment variables of the session's
kernel; `take` removes each one as it is read, so a value never carries over to a later helper.
"""

import os
import subprocess
import sys

# The pinned model and library versions.
MODEL_ID = "Qwen/Qwen3.5-4B"
TRANSFORMERS_VERSION = "5.22.1"
FLA_DISTRIBUTION = "flash-linear-attention"
FLA_VERSION = "0.6.0"
# The rclone release every session installs, from its GitHub release page.
RCLONE_VERSION = "1.72.1"

# Where things live on a session.
REPO_ON_SESSION = "/content/check-twice"
SENT_COMMIT_FILE = os.path.join(REPO_ON_SESSION, ".sent-commit")
PINS_FILE = os.path.join(REPO_ON_SESSION, "experiments", "pins.json")
EPISODES_META_FILE = os.path.join(REPO_ON_SESSION, "data", "episodes", "meta.json")
JOBS_DIR = "/content/jobs"
TRANSFER_DIR = "/content/transfer"
# The fast DeltaNet kernel package is installed here and not into the image's packages, so a
# process sees it only when this folder is on its PYTHONPATH.
FLA_DIR = "/content/fast-kernel"
RCLONE_BIN = "/usr/local/bin/rclone"
KEY_DIR = "/root/drive-key"
KEY_FILE = os.path.join(KEY_DIR, "rclone.conf")
DRIVE_REMOTE = "check2drive"

DONE_MARK = "remote step finished"


class Stop(Exception):
    """Ends a helper after its error line. The kernel shows no traceback for it."""

    def _render_traceback_(self):
        return [""]


def fail(message):
    print(f"error: {message}", flush=True)
    raise Stop(message)


def take(name):
    value = os.environ.pop(name, None)
    if value is None:
        fail(f"the setting {name} was not passed to this helper")
    return value


def finish():
    print(DONE_MARK, flush=True)


def job_env(session, builtin_path):
    """The environment of a process started on the session.

    With builtin_path, the fast kernel package's folder is absent from PYTHONPATH, so the library
    falls back to its built-in DeltaNet path. Otherwise the folder comes first, when it exists.
    """
    env = dict(os.environ)
    fla_real = os.path.realpath(FLA_DIR)
    parts = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and os.path.realpath(p) != fla_real]
    if not builtin_path and os.path.isdir(FLA_DIR):
        parts.insert(0, FLA_DIR)
    if parts:
        env["PYTHONPATH"] = os.pathsep.join(parts)
    else:
        env.pop("PYTHONPATH", None)
    env["CHECK2_SESSION"] = session
    env["PYTHONUNBUFFERED"] = "1"
    return env


def fla_importable(env):
    """Whether a new Python process with this environment can find the fast kernel package."""
    probe = "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('fla') else 3)"
    result = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True)
    if result.returncode not in (0, 3):
        fail(f"the import probe failed: {result.stdout}{result.stderr}")
    return result.returncode == 0


def _process_start(pid):
    """The start time of a live, non-zombie process in clock ticks, or None."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            stat = f.read()
    except OSError:
        return None
    fields = stat[stat.rindex(")") + 2:].split()
    if fields[0] == "Z":
        return None
    return fields[19]


def job_state(job_dir):
    """('done', exit code), ('running', None) or ('lost', None) for a job folder.

    A job is lost when its process is gone and it left no exit code, as after a restart.
    """
    exit_path = os.path.join(job_dir, "exit_code")
    if os.path.exists(exit_path):
        with open(exit_path) as f:
            return "done", int(f.read().strip())
    try:
        with open(os.path.join(job_dir, "pid")) as f:
            pid, start = f.read().split()
    except (OSError, ValueError):
        return "lost", None
    if _process_start(int(pid)) == start:
        return "running", None
    return "lost", None


def running_jobs():
    if not os.path.isdir(JOBS_DIR):
        return []
    names = sorted(os.listdir(JOBS_DIR))
    return [n for n in names if job_state(os.path.join(JOBS_DIR, n))[0] == "running"]


def tree_bytes(path):
    """(number of files, total bytes) under a file or a folder."""
    if os.path.isfile(path):
        return 1, os.path.getsize(path)
    count = total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            full = os.path.join(root, name)
            if os.path.isfile(full) and not os.path.islink(full):
                count += 1
                total += os.path.getsize(full)
    return count, total


def relay(command, env=None):
    """Runs a command, printing its output line by line as it comes; returns the exit code.

    The kernel shows only what Python prints, not what a child writes to the kernel's own
    output, so the child's output is read through a pipe and printed.
    """
    proc = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, errors="replace")
    for line in proc.stdout:
        print(line.rstrip("\n"), flush=True)
    return proc.wait()

