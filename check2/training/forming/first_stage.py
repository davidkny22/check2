"""The first stage's quantities, each written as a measurement record.

    python -m check2.training.forming.first_stage signal <config> [--epoch K] [--scratch-tag TAG]
    python -m check2.training.forming.first_stage behaviour [--formed <run folder>] [--original <run folder>]

`signal` reads the original model's saved reader outputs with no update loaded and the
formed model's with the forming update's factors loaded, each through the stale-signal
guard, and records the two paired differences on `heldout`, whether the signal sharpened,
and the gap between the cross-fitted `train` score and the `heldout` score before and after
forming (record `signal_sharpness`). `behaviour` records the formed model's `heldout` run
against the Original `heldout` run in wrong and right adoption (record
`formed_model_behaviour`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from check2 import record
from check2.data.episodes import FORMATS
from check2.evaluation.run_folders import load_run, require_comparable, require_one_card
from check2.reader import files
from check2.reader.model import OUTPUTS
from check2.reader.train import class_weights
from check2.training.forming import guard, refit, sharpness, training

RESULTS = Path(record.ROOT) / record.RESULTS
FORMED_RUN = "formed-model-heldout"
ORIGINAL_RUN = "original-heldout"


def _covered(episodes: list[dict], *outputs: dict) -> list[dict]:
    """The episodes the formed outputs hold; every one of them must be in every set of outputs."""
    chosen = [e for e in episodes if e["episode_id"] in outputs[0]]
    for p in outputs[1:]:
        missing = [e["episode_id"] for e in chosen if e["episode_id"] not in p]
        if missing:
            raise ValueError(f"{len(missing)} episodes have formed outputs and no original ones, e.g. {missing[:3]}")
    return chosen


def signal_values(original_train: dict, original_heldout: dict, formed_train: dict, formed_heldout: dict,
                  train: list[dict], heldout: list[dict]) -> dict:
    """Every signal quantity of the first stage, on the episodes the formed outputs hold."""
    weights = class_weights([OUTPUTS.index(e["situation"]) for e in train]).tolist()
    train = _covered(train, formed_train, original_train)
    heldout = _covered(heldout, formed_heldout, original_heldout)
    selection = sharpness.selection_difference(formed_heldout, original_heldout, heldout)
    cross_entropy = sharpness.cross_entropy_difference(formed_heldout, original_heldout, heldout, weights)
    return {"selection_score_formed_minus_original": selection,
            "cross_entropy_formed_minus_original": cross_entropy,
            "class_weights": weights,
            **sharpness.sharpened(selection, cross_entropy),
            "gap_before_forming": sharpness.gap(original_train, train, original_heldout, heldout),
            "gap_after_forming": sharpness.gap(formed_train, train, formed_heldout, heldout),
            "episodes": {"train": len(train), "heldout": len(heldout)}}


def signal(args) -> int:
    from check2.config.schema import load_config

    config = load_config(args.config)
    factors = training.frozen_factors(config, args.epoch, training.storage_for(args.scratch_tag))
    values = signal_values(guard.load_p(files.P_TRAIN, None), guard.load_p(files.P_HELDOUT, None),
                           guard.load_p(refit.P_TRAIN, factors), guard.load_p(refit.P_HELDOUT, factors),
                           files.load_split("train"), files.load_split("heldout"))
    values.update(training_id=training.training_id(config), forming_epoch=args.epoch,
                  scratch_tag=args.scratch_tag, update_factors_sha256=guard.factors_sha256(factors))
    path = record.write_record("measurement", "signal_sharpness", values)
    print(f"{values['reading']}: selection score {values['selection_score_formed_minus_original']}, "
          f"cross-entropy {values['cross_entropy_formed_minus_original']}; record {path}")
    return 0


def behaviour(args) -> int:
    formed_meta, formed = load_run(args.formed)
    original_meta, original = load_run(args.original)
    require_one_card([formed_meta, original_meta])
    require_comparable(formed_meta, original_meta)
    values = {"formed_run": formed_meta.get("run_id"), "original_run": original_meta.get("run_id"),
              "formats": {fmt: sharpness.behaviour(formed, original, fmt) for fmt in FORMATS}}
    path = record.write_record("measurement", "formed_model_behaviour", values)
    print(f"formed model against Original on heldout: record {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    on_signal = commands.add_parser("signal")
    on_signal.add_argument("config")
    on_signal.add_argument("--epoch", type=int, default=training.EPOCHS)
    on_signal.add_argument("--scratch-tag", default=None)
    on_behaviour = commands.add_parser("behaviour")
    on_behaviour.add_argument("--formed", type=Path, default=RESULTS / FORMED_RUN)
    on_behaviour.add_argument("--original", type=Path, default=RESULTS / ORIGINAL_RUN)
    args = parser.parse_args(argv)
    return signal(args) if args.command == "signal" else behaviour(args)


if __name__ == "__main__":
    raise SystemExit(main())
