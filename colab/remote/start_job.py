
# Starts a command on the session as a detached job with its own folder under the jobs folder.

import json
import shutil
import time

job = take("CHECK2_JOB")
argv = json.loads(take("CHECK2_ARGV"))
session = take("CHECK2_SESSION_NAME")
builtin_path = take("CHECK2_BUILTIN_PATH") == "1"

# Runs the command with the job folder as $1, then records the exit code. The command's words
# are passed as arguments, so the shell runs them as given and reads none of them as code.
WRAPPER = """d=$1
shift
"$@"
rc=$?
printf '%s\\n' "$rc" > "$d/exit_code.tmp" && mv "$d/exit_code.tmp" "$d/exit_code"
exit "$rc"
"""


def start():
    if not os.path.exists(SENT_COMMIT_FILE):
        fail(f"no commit has been sent to this session; {SENT_COMMIT_FILE} is missing")
    job_dir = os.path.join(JOBS_DIR, job)
    if os.path.isdir(job_dir):
        state, code = job_state(job_dir)
        if state == "running":
            fail(f"a job named '{job}' is already running")
        shutil.rmtree(job_dir)
        print(f"removed the earlier job '{job}' ({state}{'' if code is None else f' {code}'})")
    env = job_env(session, builtin_path)
    importable = fla_importable(env)
    if builtin_path and importable:
        fail("the fast kernel package is still importable with --builtin-path")
    os.makedirs(job_dir)
    with open(os.path.join(job_dir, "command.json"), "w") as f:
        json.dump(argv, f)
    with open(os.path.join(job_dir, "started"), "w") as f:
        f.write(f"{time.time():.3f}\n")
    with open(os.path.join(job_dir, "log.txt"), "wb") as log:
        proc = subprocess.Popen(
            ["bash", "-c", WRAPPER, "job", job_dir, *argv],
            cwd=REPO_ON_SESSION,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    with open(os.path.join(job_dir, "pid"), "w") as f:
        f.write(f"{proc.pid} {_process_start(proc.pid) or 0}\n")
    path = "built-in DeltaNet path" if builtin_path else (
        "fast kernel package importable" if importable else "fast kernel package not installed here")
    print(f"started job '{job}' (pid {proc.pid}) in {REPO_ON_SESSION}; {path}")
    print(f"command: {' '.join(argv)}")
    print(f"log: {os.path.join(job_dir, 'log.txt')}")


start()
finish()
