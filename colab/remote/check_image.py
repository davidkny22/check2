
# Reports the session's image and stops when its transformers is not the pinned version.

import platform
import shutil
from importlib import metadata


def check():
    try:
        transformers = metadata.version("transformers")
    except metadata.PackageNotFoundError:
        transformers = None
    gpu = None
    if shutil.which("nvidia-smi"):
        result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                capture_output=True, text=True)
        gpu = result.stdout.strip() or None
    print(f"python {platform.python_version()}, transformers {transformers}, GPU {gpu or 'none'}")
    if transformers != TRANSFORMERS_VERSION:
        fail(f"the image has transformers {transformers}, not the pinned {TRANSFORMERS_VERSION}; "
             "nothing was installed")
    print(f"R_GPU={1 if gpu else 0}")


check()
finish()
