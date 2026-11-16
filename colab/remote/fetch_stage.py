
# Packs a file or a folder into one archive for download ("pack"), or removes that archive
# once it has been downloaded ("clean").

import tarfile
import time

stage = take("CHECK2_STAGE")
path = take("CHECK2_PATH")


def pack():
    if not os.path.exists(path):
        fail(f"'{path}' does not exist on this session")
    kind = "file" if os.path.isfile(path) else "dir"
    files, size = tree_bytes(path)
    os.makedirs(TRANSFER_DIR, exist_ok=True)
    archive = os.path.join(TRANSFER_DIR, f"fetch-{time.time_ns()}.tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(path, arcname="payload")
    print(f"packed {kind} '{path}': {files} files, {size} bytes")
    print(f"R_KIND={kind}")
    print(f"R_ARCHIVE={archive}")
    print(f"R_FILES={files}")
    print(f"R_BYTES={size}")


def clean():
    if os.path.dirname(path) != TRANSFER_DIR:
        fail(f"'{path}' is not a transfer archive")
    os.remove(path)


if stage == "pack":
    pack()
elif stage == "clean":
    clean()
else:
    fail(f"unknown stage '{stage}'")
finish()
