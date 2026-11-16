
# Installs the pinned rclone release from its GitHub release page, checked against the
# release's own SHA256SUMS. Does nothing when that version is already installed.

import hashlib
import platform
import tempfile
import urllib.request
import zipfile

RELEASE = f"https://github.com/rclone/rclone/releases/download/v{RCLONE_VERSION}"
ZIP_NAME = f"rclone-v{RCLONE_VERSION}-linux-amd64.zip"


def installed_version():
    if not os.path.exists(RCLONE_BIN):
        return None
    result = subprocess.run([RCLONE_BIN, "version"], capture_output=True, text=True)
    first = result.stdout.splitlines()[0] if result.stdout else ""
    return first.split()[-1] if first.startswith("rclone ") else None


def install():
    if installed_version() == f"v{RCLONE_VERSION}":
        print(f"rclone v{RCLONE_VERSION} is already at {RCLONE_BIN}; nothing to do")
        return
    if platform.machine() != "x86_64":
        fail(f"this session's machine is {platform.machine()}, and the release fetched is for x86_64")
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, ZIP_NAME)
        urllib.request.urlretrieve(f"{RELEASE}/{ZIP_NAME}", zip_path)
        with urllib.request.urlopen(f"{RELEASE}/SHA256SUMS") as response:
            sums = response.read().decode()
        expected = [line.split()[0] for line in sums.splitlines()
                    if line.strip().endswith(ZIP_NAME) and len(line.split()) == 2]
        with open(zip_path, "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        if len(expected) != 1 or expected[0] != actual:
            fail(f"{ZIP_NAME} does not match the release's SHA256SUMS")
        with zipfile.ZipFile(zip_path) as z:
            member = f"rclone-v{RCLONE_VERSION}-linux-amd64/rclone"
            with z.open(member) as src, open(RCLONE_BIN + ".new", "wb") as dst:
                dst.write(src.read())
        os.chmod(RCLONE_BIN + ".new", 0o755)
        os.replace(RCLONE_BIN + ".new", RCLONE_BIN)
    found = installed_version()
    if found != f"v{RCLONE_VERSION}":
        fail(f"{RCLONE_BIN} reports {found} after the install")
    print(f"installed rclone {found} at {RCLONE_BIN}; checksum matched the release's SHA256SUMS")


install()
finish()
