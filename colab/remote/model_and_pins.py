
# Downloads the model at one revision, then writes the pins file, or compares the session
# with it when the sent commit already has one.
#
# The revision is the pins file's, when the sent commit has that file; else the tokenizer
# revision the episodes were built with, when the sent commit has their metadata; else the
# model repository's current commit.

import json
import time

# A local command is cut off after ten minutes, so the download stops before that and reports.
# Running setup again resumes it from the files already on disk.
DOWNLOAD_LIMIT_SECONDS = 8 * 60
PROGRESS_SECONDS = 30

DOWNLOAD = """
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], revision=sys.argv[2]))
"""

RUNTIME = """
import json, subprocess, torch, transformers, triton
driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                        capture_output=True, text=True, check=True).stdout.split("\\n")[0].strip()
print(json.dumps({"transformers": transformers.__version__, "torch": torch.__version__,
                  "triton": triton.__version__, "cuda": torch.version.cuda, "driver": driver,
                  "gpu": torch.cuda.get_device_name(0)}))
"""


def read_json(path):
    with open(path) as f:
        return json.load(f)


def chosen_revision():
    if os.path.exists(PINS_FILE):
        pins = read_json(PINS_FILE)
        if "model_revision" not in pins:
            fail(f"{PINS_FILE} has no model_revision")
        return pins["model_revision"], PINS_FILE
    if os.path.exists(EPISODES_META_FILE):
        meta = read_json(EPISODES_META_FILE)
        if "tokenizer_revision" not in meta:
            fail(f"{EPISODES_META_FILE} has no tokenizer_revision")
        return meta["tokenizer_revision"], EPISODES_META_FILE
    return None, "the model repository's current commit"


def folder_bytes(path):
    return tree_bytes(path)[1] if os.path.isdir(path) else 0


def download(commit):
    from huggingface_hub import constants
    cache = os.path.join(constants.HF_HUB_CACHE, "models--" + MODEL_ID.replace("/", "--"))
    env = dict(os.environ, HF_HUB_DISABLE_PROGRESS_BARS="1")
    proc = subprocess.Popen([sys.executable, "-c", DOWNLOAD, MODEL_ID, commit], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    start = time.time()
    while True:
        try:
            out, _ = proc.communicate(timeout=PROGRESS_SECONDS)
            break
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            print(f"  downloading: {folder_bytes(cache) / 1e9:.2f} GB after {elapsed:.0f} s", flush=True)
            if elapsed > DOWNLOAD_LIMIT_SECONDS:
                proc.kill()
                proc.communicate()
                fail(f"the download was stopped after {elapsed:.0f} s; run setup.sh again to resume it")
    if proc.returncode != 0:
        fail(f"the download failed:\n{out}")
    snapshot = out.strip().splitlines()[-1]
    if os.path.basename(snapshot) != commit:
        fail(f"the download landed in {snapshot}, not at commit {commit}")
    print(f"model {MODEL_ID} at {commit}: {folder_bytes(cache) / 1e9:.2f} GB in {snapshot}")


def runtime_versions():
    result = subprocess.run([sys.executable, "-c", RUNTIME], env=job_env("setup", builtin_path=True),
                            capture_output=True, text=True)
    if result.returncode != 0:
        fail(f"could not read the runtime's versions:\n{result.stdout}{result.stderr}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def main():
    if not os.path.exists(SENT_COMMIT_FILE):
        fail("no commit has been sent to this session; run send.sh first")
    from huggingface_hub import HfApi
    revision, source = chosen_revision()
    commit = HfApi().model_info(MODEL_ID, revision=revision).sha
    print(f"(d) revision {revision or 'default branch'} from {source}; commit {commit}")
    download(commit)

    now = {"model": MODEL_ID, "model_revision": commit, **runtime_versions()}
    if not os.path.exists(PINS_FILE):
        os.makedirs(os.path.dirname(PINS_FILE), exist_ok=True)
        with open(PINS_FILE, "w") as f:
            json.dump(now, f, indent=2)
            f.write("\n")
        print(f"(e) wrote {PINS_FILE}: {json.dumps(now)}")
        return
    pinned = read_json(PINS_FILE)
    differ = {k: (pinned.get(k), now.get(k)) for k in sorted(set(pinned) | set(now)) if pinned.get(k) != now.get(k)}
    for key, (want, have) in differ.items():
        print(f"  {key}: pinned {want}, this session {have}")
    if differ:
        fail(f"this session differs from {PINS_FILE} in {', '.join(differ)}")
    print(f"(e) this session matches {PINS_FILE}: {json.dumps(now)}")


main()
finish()
