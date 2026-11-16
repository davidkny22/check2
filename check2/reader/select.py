"""The read position and the reader variant, both chosen on `heldout`.

For each candidate tail position, from `tail[0]` to `tail[K-2]`, the default reader is fitted
on `train` and scored on `heldout` by the selection score. The read position is the earliest
candidate whose score is within one standard error of the best, that standard error being the
best candidate's under 2,000 resamples of `heldout` question ids. The four variants are then
fitted at that position, and the one with the highest `heldout` selection score is kept with
its epoch count.

Run as `python -m check2.reader.select [--per-format N]`; writes `data/reader/selection.json`
and a measurement record.
"""
import argparse
import sys
from collections.abc import Mapping

from check2.data.episodes import TAIL_LENGTH
from check2.evaluation.intervals import standard_error
from check2.reader import files
from check2.reader.inputs import Inputs, load_inputs
from check2.reader.model import DEFAULT_VARIANT, VARIANTS
from check2.reader.scores import contrast_records, flatten, one_vs_rest, selection_score, user_right_scores
from check2.reader.train import MAX_EPOCHS, Fit, fit, predict
from check2.record import write_record

CANDIDATES = tuple(range(TAIL_LENGTH - 1))
# Outside this range the default reader's held-out score is far from what design measured,
# which points at a defect in the cache, the labels or the split.
PLAUSIBLE_SCORE = (0.6, 0.97)


def choose_position(scores: Mapping[int, float], best_standard_error: float) -> int:
    """The earliest position whose score is within one standard error of the best score."""
    best = max(scores.values())
    return min(k for k, s in scores.items() if s >= best - best_standard_error)


def best_position(scores: Mapping[int, float]) -> int:
    """The position with the highest score, the earliest of any tie."""
    best = max(scores.values())
    return min(k for k, s in scores.items() if s == best)


def choose_variant(scores: Mapping[str, float]) -> str:
    """The variant with the highest score; a tie goes to the one listed first in `VARIANTS`."""
    best = max(scores.values())
    return next(name for name in VARIANTS if name in scores and scores[name] == best)


def evaluate(result: Fit, inputs: Inputs, tail_position: int) -> dict:
    """A fit's `heldout` selection score, its standard error, its one-vs-rest AUROCs, its losses."""
    ids = inputs.heldout_ids
    p = dict(zip(ids, predict(result.reader, inputs.bank, inputs.bank_layers, ids, tail_position).tolist()))
    by_qid = contrast_records(inputs.heldout, user_right_scores(p))
    score = selection_score(flatten(by_qid))
    try:
        se = standard_error(selection_score, by_qid)
    except ValueError:
        se = None
    return {"score": score, "standard_error": se, "best_epoch": result.best_epoch,
            "epochs_run": result.epochs_run, "heldout_loss": result.heldout_loss,
            "train_loss": result.train_loss, "contrast_questions": len(by_qid),
            "one_vs_rest": one_vs_rest(p, inputs.label_of)}


def select(inputs: Inputs, seed: int = files.SEED) -> dict:
    """Every candidate's fit and score, the chosen position, the four variants there, the choice."""
    fits: dict[tuple[str, int], Fit] = {}
    candidates = {}
    for k in CANDIDATES:
        fits[DEFAULT_VARIANT, k] = fit(DEFAULT_VARIANT, inputs.bank, inputs.bank_layers, inputs.train_ids,
                                       inputs.label_of, k, seed, heldout_ids=inputs.heldout_ids)
        candidates[k] = evaluate(fits[DEFAULT_VARIANT, k], inputs, k)
        print(f"position tail[{k}]: score {candidates[k]['score']} se {candidates[k]['standard_error']} "
              f"best epoch {candidates[k]['best_epoch']}", flush=True)
    scores = {k: c["score"] for k, c in candidates.items() if c["score"] is not None}
    if not scores or candidates[best_position(scores)]["standard_error"] is None:
        raise ValueError("the heldout episodes hold too few of the selection contrast to score a candidate")
    best = best_position(scores)
    position = choose_position(scores, candidates[best]["standard_error"])
    variants = {DEFAULT_VARIANT: candidates[position]}
    for name in VARIANTS:
        if name != DEFAULT_VARIANT:
            result = fit(name, inputs.bank, inputs.bank_layers, inputs.train_ids, inputs.label_of, position,
                         seed, heldout_ids=inputs.heldout_ids)
            variants[name] = evaluate(result, inputs, position)
            print(f"variant {name} at tail[{position}]: score {variants[name]['score']} "
                  f"best epoch {variants[name]['best_epoch']}", flush=True)
    variant = choose_variant({n: v["score"] for n, v in variants.items() if v["score"] is not None})
    chosen = variants[variant]
    default_score = variants[DEFAULT_VARIANT]["score"]
    return {
        "candidates": {str(k): candidates[k] for k in CANDIDATES},
        "best_position": best, "best_standard_error": candidates[best]["standard_error"],
        "read_position": position,
        "variants": variants, "variant": variant, "variant_settings": VARIANTS[variant],
        "epochs": chosen["best_epoch"], "heldout_loss": chosen["heldout_loss"],
        "train_loss": chosen["train_loss"],
        "episodes": {"train": len(inputs.train), "heldout": len(inputs.heldout)},
        "seed": seed,
        "checks": {
            "default_score_plausible": PLAUSIBLE_SCORE[0] <= default_score <= PLAUSIBLE_SCORE[1],
            "heldout_loss_still_falling_at_last_epoch": chosen["best_epoch"] == MAX_EPOCHS,
            "fits_whose_best_epoch_is_the_last": sorted(
                f"{name}@tail[{k}]" for (name, k), f in fits.items() if f.best_epoch == MAX_EPOCHS)
            + sorted(f"{name}@tail[{position}]" for name, v in variants.items()
                     if name != DEFAULT_VARIANT and v["best_epoch"] == MAX_EPOCHS),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", default=str(files.CONVERSATION_CACHE))
    parser.add_argument("--per-format", type=int, default=None,
                        help="a small run on N episodes per format of each split")
    args = parser.parse_args(argv)
    train = files.small_draw(files.load_split("train"), args.per_format)
    heldout = files.small_draw(files.load_split("heldout"), args.per_format)
    result = select(load_inputs(args.cache_dir, train, heldout))
    result["per_format"] = args.per_format
    files.write_json(files.SELECTION, result)
    record = write_record("measurement", "reader_selection", result)
    checks = result["checks"]
    print(f"read position tail[{result['read_position']}] (best tail[{result['best_position']}], "
          f"se {result['best_standard_error']}); variant {result['variant']}; epochs {result['epochs']}")
    print(f"variant scores: " + ", ".join(f"{n} {v['score']}" for n, v in result["variants"].items()))
    print(f"checks: {checks}")
    print(f"wrote {files.SELECTION} and {record}")
    if not checks["default_score_plausible"]:
        print(f"STOP: the default reader's held-out score {result['variants'][DEFAULT_VARIANT]['score']} "
              f"is outside {PLAUSIBLE_SCORE}")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
