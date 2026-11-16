
# Installs the fast DeltaNet kernel package into its own folder, so that a process started
# with that folder on its PYTHONPATH can import it and one started without cannot. Then checks
# both kinds of process: the library must report the package available in the first and
# unavailable in the second, since that report is what selects the DeltaNet kernel.

import json
import shutil
import tempfile
from importlib import metadata

# The package ships its kernels in a second distribution of the same project. Any other
# distribution the install would add means the image lacks a dependency: that is a stop.
ALLOWED = {"flash-linear-attention", "fla-core"}
SPEC = f"{FLA_DISTRIBUTION}=={FLA_VERSION}"

PROBE = """
import importlib.metadata as m, importlib.util, sys
from transformers.utils.import_utils import is_flash_linear_attention_available
found = importlib.util.find_spec('fla') is not None
version = m.version('flash-linear-attention') if found else None
print(found, version, is_flash_linear_attention_available())
"""


def canonical(name):
    return name.lower().replace("_", "-").replace(".", "-")


def installed_here():
    if not os.path.isdir(FLA_DIR):
        return {}
    return {canonical(d.metadata["Name"]): d.version for d in metadata.distributions(path=[FLA_DIR])}


def resolve():
    """The distributions pip would add to the image for the package, with their versions."""
    with tempfile.TemporaryDirectory() as tmp:
        report = os.path.join(tmp, "report.json")
        code = relay([sys.executable, "-m", "pip", "install", "--dry-run", "--quiet", "--report", report, SPEC])
        if code != 0:
            fail("pip could not resolve the package")
        with open(report) as f:
            items = json.load(f)["install"]
    return {canonical(i["metadata"]["name"]): i["metadata"]["version"] for i in items}


def probe(builtin_path):
    env = job_env("setup", builtin_path)
    result = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True)
    if result.returncode != 0:
        fail(f"the import check failed:\n{result.stdout}{result.stderr}")
    return result.stdout.strip().splitlines()[-1]


def install():
    present = installed_here()
    if present.get(FLA_DISTRIBUTION) == FLA_VERSION:
        print(f"{SPEC} is already in {FLA_DIR}: {present}")
    else:
        wanted = resolve()
        extra = sorted(set(wanted) - ALLOWED)
        if extra:
            fail(f"installing {SPEC} would also add {extra}, which the image should already have")
        if wanted.get(FLA_DISTRIBUTION) != FLA_VERSION:
            fail(f"pip resolved {wanted}, without {SPEC}")
        if os.path.exists(FLA_DIR):
            shutil.rmtree(FLA_DIR)
        pins = [f"{name}=={version}" for name, version in sorted(wanted.items())]
        code = relay([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "--target", FLA_DIR, *pins])
        if code != 0:
            fail("pip install failed")
        print(f"installed {pins} into {FLA_DIR}")
    fast = probe(builtin_path=False)
    builtin = probe(builtin_path=True)
    print(f"process with the package (found, version, library says available): {fast}")
    print(f"process on the built-in path (found, version, library says available): {builtin}")
    if fast != f"True {FLA_VERSION} True":
        fail("a process started with the package cannot use it")
    if builtin != "False None False":
        fail("a process started on the built-in path can still see the package")


install()
finish()
