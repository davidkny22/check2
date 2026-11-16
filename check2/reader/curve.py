"""The reader's learning curve: is it short of data?

In each of the five folds, the selected reader is fitted on the first quarter, half, three
quarters and all of a seeded order of that fold's training units (each duplicate group one
unit), each fit stopping by the `heldout` loss, and scores the held fold. With the folds
pooled, the paired difference in selection score between consecutive sizes is reported with
its interval from 2,000 resamples of question ids. The reader is short of data only if the
last difference, three quarters to all, has its lower end above zero.

Run as `python -m check2.reader.curve [--per-format N]` after the selection; writes a
measurement record.
"""
import argparse
import random
import sys

from check2.data.pool import duplicate_groups, load_pool
from check2.data.splits import load_splits, units_of
from check2.evaluation.intervals import interval, paired_difference
from check2.reader import files
from check2.reader.folds import fold_of
from check2.reader.inputs import Inputs, load_inputs
from check2.reader.scores import contrast_records, selection_score, user_right_scores
from check2.reader.train import fit, predict
from check2.record import write_record

FRACTIONS = (0.25, 0.5, 0.75, 1.0)
SUBSET_SEED = 2032


def nested_subsets(units: list[tuple[str, ...]], fractions=FRACTIONS,
                   seed: int = SUBSET_SEED) -> list[list[tuple[str, ...]]]:
    """For each fraction q, the first `round(q * n)` units of one seeded order of all units."""
    order = random.Random(seed).sample(sorted(units), len(units))
    return [order[:round(q * len(order))] for q in fractions]


def curve_fits(inputs: Inputs, folds: list[list[str]], groups: list[list[str]], variant: str,
               tail_position: int, seed: int = files.SEED) -> tuple[list[dict[str, list[float]]], list[dict]]:
    """For each fraction, `p` of every `train` episode from its fold's fit at that size."""
    where = fold_of(folds)
    by_qid: dict[str, list[str]] = {}
    for e in inputs.train:
        by_qid.setdefault(e["qid"], []).append(e["episode_id"])
    p_by_size: list[dict[str, list[float]]] = [{} for _ in FRACTIONS]
    fits = []
    for k in range(len(folds)):
        training_qids = sorted(q for q in by_qid if where[q] != k)
        scored = sorted(e for q in by_qid if where[q] == k for e in by_qid[q])
        for i, subset in enumerate(nested_subsets(units_of(training_qids, groups))):
            trained = sorted(e for unit in subset for q in unit for e in by_qid[q])
            result = fit(variant, inputs.bank, inputs.bank_layers, trained, inputs.label_of, tail_position,
                         seed, heldout_ids=inputs.heldout_ids)
            p_by_size[i].update(zip(scored, predict(result.reader, inputs.bank, inputs.bank_layers, scored,
                                                    tail_position).tolist()))
            fits.append({"fold": k, "fraction": FRACTIONS[i], "units": len(subset), "episodes": len(trained),
                         "best_epoch": result.best_epoch, "epochs_run": result.epochs_run})
            print(f"fold {k} fraction {FRACTIONS[i]}: {len(subset)} units, best epoch {result.best_epoch}",
                  flush=True)
    return p_by_size, fits


def summarise(p_by_size: list[dict[str, list[float]]], train: list[dict]) -> dict:
    records = [contrast_records(train, user_right_scores(p)) for p in p_by_size]
    sizes = []
    for q, by_qid in zip(FRACTIONS, records):
        est, low, high = interval(selection_score, by_qid)
        sizes.append({"fraction": q, "selection_score": est, "interval": [low, high]})
    steps = []
    for i in range(1, len(FRACTIONS)):
        d = paired_difference(selection_score, records[i], records[i - 1])
        steps.append({"from": FRACTIONS[i - 1], "to": FRACTIONS[i], "difference": d.estimate,
                      "interval": [d.low, d.high], "skipped": d.skipped})
    last = steps[-1]
    return {"sizes": sizes, "steps": steps,
            "short_of_data": last["interval"][0] is not None and last["interval"][0] > 0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", default=str(files.CONVERSATION_CACHE))
    parser.add_argument("--per-format", type=int, default=None)
    args = parser.parse_args(argv)
    selection = files.read_json(files.SELECTION)
    variant, position = selection["variant"], selection["read_position"]
    train = files.small_draw(files.load_split("train"), args.per_format)
    heldout = files.small_draw(files.load_split("heldout"), args.per_format)
    inputs = load_inputs(args.cache_dir, train, heldout)
    p_by_size, fits = curve_fits(inputs, load_splits().folds, duplicate_groups(load_pool()), variant, position)
    values = {"variant": variant, "read_position": position, "per_format": args.per_format,
              **summarise(p_by_size, train), "fits": fits}
    record = write_record("measurement", "reader_learning_curve", values)
    for s in values["sizes"]:
        print(f"fraction {s['fraction']}: {s['selection_score']} {s['interval']}")
    for s in values["steps"]:
        print(f"{s['from']} -> {s['to']}: {s['difference']} {s['interval']}")
    print(f"short of data: {values['short_of_data']}; record {record}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
