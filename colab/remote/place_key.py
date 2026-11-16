
# Places the uploaded Drive key file in its own folder with mode 600. "prepare" makes the
# folder and names the path to upload to; "place" moves the upload into place, or discards it
# when the same file is already there. Nothing here prints the file's contents.

import filecmp
import stat

stage = take("CHECK2_STAGE")
INCOMING = os.path.join(KEY_DIR, "incoming")


def prepare():
    os.makedirs(KEY_DIR, exist_ok=True)
    os.chmod(KEY_DIR, 0o700)
    print(f"key folder {KEY_DIR} ready, mode 700")
    print(f"R_INCOMING={INCOMING}")


def place():
    if not os.path.exists(INCOMING):
        fail(f"nothing was uploaded to {INCOMING}")
    os.chmod(INCOMING, 0o600)
    if os.path.exists(KEY_FILE) and filecmp.cmp(INCOMING, KEY_FILE, shallow=False):
        os.remove(INCOMING)
        action = "the same key file was already in place; unchanged"
    else:
        os.replace(INCOMING, KEY_FILE)
        action = "key file placed"
    if stat.S_IMODE(os.stat(KEY_FILE).st_mode) != 0o600:
        os.chmod(KEY_FILE, 0o600)
    result = subprocess.run([RCLONE_BIN, "listremotes", "--config", KEY_FILE], capture_output=True, text=True)
    if f"{DRIVE_REMOTE}:" not in result.stdout.split():
        fail(f"the key file does not define the remote {DRIVE_REMOTE}:")
    mode = oct(stat.S_IMODE(os.stat(KEY_FILE).st_mode))[2:]
    print(f"{action}: {KEY_FILE}, mode {mode}, defines {DRIVE_REMOTE}:")


if stage == "prepare":
    prepare()
elif stage == "place":
    place()
else:
    fail(f"unknown stage '{stage}'")
finish()
