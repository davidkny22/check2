
# Reports a job's state, its age and the end of its log.

import time

job = take("CHECK2_JOB")
lines = int(take("CHECK2_LINES"))

# The log is read from its end; this many bytes hold far more lines than a report shows.
TAIL_BYTES = 256 * 1024


def tail(path, n):
    if n == 0 or not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        size = f.seek(0, os.SEEK_END)
        f.seek(max(0, size - TAIL_BYTES))
        text = f.read().decode("utf-8", errors="replace")
    return text.splitlines()[-n:]


def report():
    job_dir = os.path.join(JOBS_DIR, job)
    if not os.path.isdir(job_dir):
        fail(f"no job named '{job}' on this session")
    state, code = job_state(job_dir)
    with open(os.path.join(job_dir, "started")) as f:
        started = float(f.read())
    now = time.time()
    if state == "done":
        ended = os.path.getmtime(os.path.join(job_dir, "exit_code"))
        print(f"done {code}")
        print(f"started {now - started:.0f} s ago; ran {ended - started:.0f} s")
    else:
        print(state)
        print(f"started {now - started:.0f} s ago")
    log = os.path.join(job_dir, "log.txt")
    shown = tail(log, lines)
    print(f"--- last {len(shown)} lines of {log}")
    for line in shown:
        print(line)
    print("---", flush=True)
    if state == "lost":
        fail(f"the process of job '{job}' is gone and left no exit code")


report()
finish()
