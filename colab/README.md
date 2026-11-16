# Session scripts

These scripts drive Colab sessions from this machine through the `colab` tool inside WSL
(`~/.local/bin/colab --auth=oauth2`). Each one runs as a script file, never inline:

```
MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash /mnt/c/<path to the checkout>/colab/<script> <args>
```

Each prints what it did in a few lines and exits non-zero on failure. Code that runs on the
session lives in `remote/`; a script sends it with `remote/common.py` in front of it.

| Script | Arguments | Example |
|---|---|---|
| `status.sh` | none | `status.sh` lists the live sessions and the balance. |
| `start.sh` | `<session> <cpu\|T4\|L4\|A100\|G4> [--high-mem] [--minutes N]` | `start.sh c2-a G4` asks once a minute for up to 20 minutes, then prints the status. |
| `setup.sh` | `<session>` | `setup.sh c2-a` installs rclone and the Drive key; on a GPU session, after `send.sh`, it also installs the fast DeltaNet kernel package in `/content/fast-kernel`, downloads the model and writes or checks `experiments/pins.json`. Safe to run twice. |
| `send.sh` | `<session> <commit>` | `send.sh c2-a master` replaces `/content/check-twice` with that commit and writes its hash to `.sent-commit`. |
| `run.sh` | `<session> <job> [--builtin-path] -- <command...>` | `run.sh c2-a tests -- python -m pytest -q` starts the job detached; output in `/content/jobs/tests/log.txt`, exit code in `exit_code`. `--builtin-path` hides the fast kernel package. |
| `poll.sh` | `<session> <job> [--lines N]` | `poll.sh c2-a cache --lines 5` prints `running` or `done <code>`, the job's age and its last lines. |
| `fetch.sh` | `<session> <remote path> <local path>` | `fetch.sh c2-a /content/check-twice/results C:/x/results` copies a file or a folder here and checks file count and bytes. |
| `drive.sh` | `<session> <to\|from> <session path> <drive path>` | `drive.sh c2-a to /content/ckpt checkpoints/run-1` copies with rclone and checks every file's size on both sides. |
| `stop.sh` | `<session>` | `stop.sh c2-a` stops the session; its disk is lost. |

A local command is cut off after ten minutes. `start.sh` keeps asking for up to its
`--minutes`, which can be longer, so start it in the background. Jobs run detached and are
polled.

## What goes wrong most

1. **The job's command is read by the WSL shell first.** `wsl -- bash script args` passes the
   arguments through a shell inside WSL, which expands `$NAME` and removes quotes before
   `run.sh` sees them. A command that needs a pipe, a redirect, a quoted string or a
   variable goes in a script file in the repository, sent with `send.sh` and run by name.
2. **The tool reports success when code on the session failed.** `colab exec` exits 0
   after a remote error, so every helper ends by printing `remote step finished`, and a
   script fails unless that is the last line. If a script stops with no clear message,
   rerun it and read the helper's output above the error.
3. **A session's disk is lost when it stops, and a lost job leaves no exit code.**
   Copy checkpoints to Drive as they are made. `poll.sh` reports `lost` when a job's
   process is gone without an exit code. `send.sh` refuses while any job is running,
   because the job runs inside the folder it replaces.
