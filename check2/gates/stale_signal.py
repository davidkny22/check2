"""Stale-signal guard: with the forming update present, reader outputs saved from the original model are refused.

    python -m check2.gates.stale_signal [--plant] --model tiny|full [--update <checkpoint folder>]

The model is loaded and the forming update is built on it and installed: with the factors
of a forming checkpoint's `adapter.pt` when `--update` names its folder, otherwise at
initialisation. The saved reader outputs are then loaded through the guard in five cases:

1. update present, the original model's saved `heldout` outputs: must raise;
2. update present, outputs recorded as computed from this update: must load;
3. update present, outputs recorded as computed from another update: must raise;
4. no update, outputs recorded as computed from this update: must raise;
5. no update, the original model's saved outputs: must load.

The gate is green when every case comes out as it must. The planted fault removes the
guard's check, so that case 1 loads and the gate goes red.
"""

from __future__ import annotations

import contextlib
import tempfile
import time
from pathlib import Path

import torch

from check2.config.schema import Config
from check2.gates import mechanism_setup as setup
from check2.mechanism.adapter.implementations import IMPLEMENTATIONS
from check2.reader import files
from check2.training.forming import guard
from check2.training.forming.update import FormingUpdate, layers_for

NAME = "stale_signal"
ORIGINAL_OUTPUTS = files.P_HELDOUT
FORMED_FILE = "p_heldout.jsonl"
FORMED_ROWS = 4
# The id written into the made-up records; no training of record carries it.
MADE_UP_TRAINING = "made-up-forming"


@contextlib.contextmanager
def check_removed():
    """The planted fault: the guard reads the record and checks nothing."""
    original = guard.check_source
    guard.check_source = lambda folder, factors: guard.read_source(folder)
    try:
        yield
    finally:
        guard.check_source = original


def _attempt(path: Path, factors) -> dict:
    try:
        rows = guard.load_p(path, factors)
    except guard.StaleSignal as err:
        return {"outcome": "raised", "message": str(err)}
    return {"outcome": "loaded", "rows": len(rows)}


def _other(factors: dict) -> dict:
    """The same factors with one entry of the first module's `B` changed: another update."""
    out = {name: (a.detach().clone(), b.detach().clone()) for name, (a, b) in factors.items()}
    first = sorted(out)[0]
    out[first][1].view(-1)[0] += 1.0
    return out


def check(update: FormingUpdate, original_outputs: Path, plant: bool = False) -> dict:
    """The five cases on the given update and the original model's saved outputs."""
    factors = {name: (a.detach().cpu(), b.detach().cpu()) for name, (a, b) in update.factors().items()}
    rows = files.read_p(original_outputs)
    sample = [{"episode_id": e, "p": rows[e]} for e in sorted(rows)[:FORMED_ROWS]]
    cases = {}
    with tempfile.TemporaryDirectory() as scratch:
        this = Path(scratch) / "this"
        other = Path(scratch) / "other"
        for folder, recorded in ((this, factors), (other, _other(factors))):
            files.write_jsonl(folder / FORMED_FILE, sample)
            guard.write_source(folder, MADE_UP_TRAINING, recorded)
        plan = (("update_with_original_outputs", original_outputs, factors, "raised"),
                ("update_with_its_own_outputs", this / FORMED_FILE, factors, "loaded"),
                ("update_with_another_update_s_outputs", other / FORMED_FILE, factors, "raised"),
                ("no_update_with_formed_outputs", this / FORMED_FILE, None, "raised"),
                ("no_update_with_original_outputs", original_outputs, None, "loaded"))
        with check_removed() if plant else contextlib.nullcontext():
            for name, path, given, expected in plan:
                result = _attempt(path, given)
                cases[name] = {**result, "expected": expected, "as_required": result["outcome"] == expected}
    return {"original_outputs": str(original_outputs), "update_modules": len(update.modules),
            "update_factors_sha256": guard.factors_sha256(factors), "cases": cases,
            "green": all(c["as_required"] for c in cases.values())}


def build_update(bench: setup.Bench, implementation: str, checkpoint: Path | None) -> FormingUpdate:
    """The forming update on the loaded model, installed, with the checkpoint's factors or at initialisation."""
    torch.manual_seed(Config.seed)
    first, last = layers_for(bench.layout)
    update = FormingUpdate(bench.model, implementation, first, last)
    if checkpoint is not None:
        update.load_factors(torch.load(Path(checkpoint) / "adapter.pt", map_location="cpu", weights_only=True))
    update.install()
    return update


def main(argv=None) -> int:
    parser = setup.parser(__doc__.splitlines()[0])
    parser.add_argument("--implementation", choices=sorted(IMPLEMENTATIONS), default="hooks")
    parser.add_argument("--update", type=Path, default=None, help="a forming checkpoint's epoch folder")
    parser.add_argument("--original-outputs", type=Path, default=ORIGINAL_OUTPUTS)
    args = parser.parse_args(argv)
    started = time.time()
    bench = setup.load(args.model, exact=False)
    update = build_update(bench, args.implementation, args.update)
    try:
        values = check(update, args.original_outputs, plant=args.plant)
    finally:
        update.remove()
    values.update(implementation=args.implementation, update=None if args.update is None else str(args.update))
    return setup.finish(NAME, args, bench, values, values["green"], started)


if __name__ == "__main__":
    setup.run_main(main)
