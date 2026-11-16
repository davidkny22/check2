
# Copies between the session's disk and the Drive folder with rclone, then compares sizes.

import json
import posixpath

direction = take("CHECK2_DIRECTION")
session_path = take("CHECK2_SESSION_PATH")
drive_path = take("CHECK2_DRIVE_PATH")

# Progress is printed this often, which also keeps the tool from giving up on a long copy.
RCLONE_FLAGS = ["--config", KEY_FILE, "--stats", "30s", "--stats-one-line", "--stats-log-level", "NOTICE"]


def rclone(*args, capture=False):
    command = [RCLONE_BIN, *args, *RCLONE_FLAGS]
    if capture:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            fail(f"rclone {args[0]} failed: {result.stderr.strip()}")
        return result.stdout
    if relay(command) != 0:
        fail(f"rclone {args[0]} failed")
    return None


def drive_listing(target):
    """{relative path: size} of the files at target on Drive; a single file has the path ''."""
    stat = json.loads(rclone("lsjson", "--stat", target, capture=True))
    if not stat.get("IsDir"):
        return {"": stat["Size"]}
    entries = json.loads(rclone("lsjson", "-R", "--files-only", target, capture=True))
    return {e["Path"]: e["Size"] for e in entries}


def local_listing(path):
    if os.path.isfile(path):
        return {"": os.path.getsize(path)}
    out = {}
    for root, _dirs, files in os.walk(path):
        for name in files:
            full = os.path.join(root, name)
            if os.path.isfile(full) and not os.path.islink(full):
                out[os.path.relpath(full, path)] = os.path.getsize(full)
    return out


def compare(copied, other, other_name):
    """Every copied file must exist in the other listing with the same size."""
    bad = [p for p, size in sorted(copied.items()) if other.get(p) != size]
    for p in bad[:20]:
        print(f"  {p or '(the file)'}: {copied[p]} bytes here, {other.get(p)} on {other_name}")
    if bad:
        fail(f"size check failed for {len(bad)} of {len(copied)} files")
    print(f"size check: {len(copied)} files, {sum(copied.values())} bytes, the same on both sides")


def copy():
    parts = drive_path.split("/")
    if drive_path.startswith("/") or any(p in ("", ".", "..") for p in parts):
        fail(f"the Drive path '{drive_path}' must be relative to the Drive folder, without '.' or '..'")
    if not session_path.startswith("/"):
        fail(f"the session path '{session_path}' must be absolute")
    if not os.path.exists(KEY_FILE) or not os.path.exists(RCLONE_BIN):
        fail("rclone or the key file is missing; run setup.sh first")
    target = f"{DRIVE_REMOTE}:{drive_path}"
    if direction == "to":
        if not os.path.exists(session_path):
            fail(f"'{session_path}' does not exist on this session")
        verb = "copyto" if os.path.isfile(session_path) else "copy"
        rclone(verb, session_path, target)
        compare(local_listing(session_path), drive_listing(target), "Drive")
    elif direction == "from":
        listing = drive_listing(target)
        verb = "copyto" if "" in listing else "copy"
        if verb == "copyto":
            os.makedirs(posixpath.dirname(session_path) or "/", exist_ok=True)
        rclone(verb, target, session_path)
        compare(listing, local_listing(session_path), "this session")
    else:
        fail(f"unknown direction '{direction}'")
    print(f"copied {direction} Drive: {session_path} {'->' if direction == 'to' else '<-'} {target}")


copy()
finish()
