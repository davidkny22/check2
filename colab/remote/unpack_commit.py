
# Unpacks an uploaded commit archive into the repository folder on the session.

import shutil
import tarfile

archive = take("CHECK2_ARCHIVE")
expected_bytes = int(take("CHECK2_ARCHIVE_BYTES"))
commit = take("CHECK2_COMMIT")


def unpack():
    try:
        if os.path.getsize(archive) != expected_bytes:
            fail(f"the uploaded archive has {os.path.getsize(archive)} bytes, not {expected_bytes}")
        busy = running_jobs()
        if busy:
            fail(f"jobs still running in {REPO_ON_SESSION}: {', '.join(busy)}")
        if os.path.exists(REPO_ON_SESSION):
            shutil.rmtree(REPO_ON_SESSION)
        os.makedirs(REPO_ON_SESSION)
        with tarfile.open(archive, "r:gz") as tar:
            files = sum(1 for m in tar.getmembers() if m.isfile())
            tar.extractall(REPO_ON_SESSION, filter="data")
        with open(SENT_COMMIT_FILE, "w") as f:
            f.write(commit + "\n")
    finally:
        if os.path.exists(archive):
            os.remove(archive)
    print(f"unpacked {files} files to {REPO_ON_SESSION}; {SENT_COMMIT_FILE} holds {commit}")


unpack()
finish()
