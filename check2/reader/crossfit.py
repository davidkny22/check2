"""The reader's signal for `train` and `heldout` episodes, from fits that never saw the question.

For each of the five folds of `train`, the selected variant is refitted on the other four
folds, at the selected read position for the selected number of epochs, and gives `p` for the
fold's own episodes. The fit on all of `train` gives `p` for `heldout` and is saved as the
reader every evaluation run uses.

Run as `python -m check2.reader.crossfit [--per-format N]` after the selection; writes
`data/reader/p_train.jsonl`, `p_heldout.jsonl`, `fold_fits.json`, the fitted reader, and a
measurement record.
"""
import argparse
import sys
from pathlib import Path

import torch

from check2.data.splits import load_splits
from check2.reader import files
from check2.reader.folds import fold_parts
from check2.reader.inputs import Inputs, load_inputs
from check2.reader.model import Reader
from check2.reader.scores import contrast_records, flatten, one_vs_rest, selection_score, user_right_scores
from check2.reader.train import fit, predict
from check2.record import write_record


def save_reader(reader: Reader, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"settings": reader.settings(),
                "state_dict": {k: v.detach().cpu() for k, v in reader.state_dict().items()}}, path)


def load_reader(path: str | Path) -> Reader:
    """A saved reader, on the CPU, in evaluation mode."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    reader = Reader(**saved["settings"])
    reader.load_state_dict(saved["state_dict"])
    return reader.eval()


def cross_fit(inputs: Inputs, folds: list[list[str]], variant: str, tail_position: int, epochs: int,
              seed: int = files.SEED) -> tuple[dict[str, list[float]], list[dict]]:
    """`p` for every `train` episode from the fit that left out its fold, and what each fit saw."""
    qid_of = {e["episode_id"]: e["qid"] for e in inputs.train}
    p: dict[str, list[float]] = {}
    fold_fits = []
    for k in range(len(folds)):
        trained, scored = fold_parts(inputs.train, folds, k)
        result = fit(variant, inputs.bank, inputs.bank_layers, trained, inputs.label_of, tail_position, seed,
                     epochs=epochs)
        p.update(zip(scored, predict(result.reader, inputs.bank, inputs.bank_layers, scored,
                                     tail_position).tolist()))
        fold_fits.append({"fold": k, "epochs": result.epochs_run,
                          "trained_questions": sorted({qid_of[e] for e in trained}),
                          "scored_questions": sorted({qid_of[e] for e in scored}),
                          "trained_episodes": len(trained), "scored_episodes": len(scored)})
        print(f"fold {k}: trained on {len(trained)} episodes, scored {len(scored)}", flush=True)
    return p, fold_fits


def scores_of(p: dict[str, list[float]], episodes: list[dict], label_of: dict[str, str]) -> dict:
    by_qid = contrast_records(episodes, user_right_scores(p))
    return {"selection_score": selection_score(flatten(by_qid)), "contrast_questions": len(by_qid),
            "one_vs_rest": one_vs_rest(p, label_of)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", default=str(files.CONVERSATION_CACHE))
    parser.add_argument("--reader-path", default=str(files.READER_WEIGHTS))
    parser.add_argument("--per-format", type=int, default=None)
    args = parser.parse_args(argv)
    selection = files.read_json(files.SELECTION)
    variant, position, epochs = selection["variant"], selection["read_position"], selection["epochs"]
    train = files.small_draw(files.load_split("train"), args.per_format)
    heldout = files.small_draw(files.load_split("heldout"), args.per_format)
    inputs = load_inputs(args.cache_dir, train, heldout)
    p_train, fold_fits = cross_fit(inputs, load_splits().folds, variant, position, epochs)
    full = fit(variant, inputs.bank, inputs.bank_layers, inputs.train_ids, inputs.label_of, position,
               files.SEED, epochs=epochs)
    p_heldout = dict(zip(inputs.heldout_ids, predict(full.reader, inputs.bank, inputs.bank_layers,
                                                    inputs.heldout_ids, position).tolist()))
    save_reader(full.reader, args.reader_path)
    files.write_jsonl(files.P_TRAIN, files.p_rows(p_train))
    files.write_jsonl(files.P_HELDOUT, files.p_rows(p_heldout))
    files.write_json(files.FOLD_FITS, fold_fits)
    values = {"variant": variant, "read_position": position, "epochs": epochs, "per_format": args.per_format,
              "train_cross_fitted": scores_of(p_train, train, inputs.label_of),
              "heldout": scores_of(p_heldout, heldout, inputs.label_of),
              "heldout_score_at_selection": selection["variants"][variant]["score"],
              "folds": [{k: v for k, v in f.items() if not k.endswith("_questions")} for f in fold_fits],
              "reader_path": args.reader_path}
    record = write_record("measurement", "reader_cross_fit", values)
    print(f"cross-fitted train: {values['train_cross_fitted']}")
    print(f"heldout from the fit on all of train: {values['heldout']} "
          f"(at selection: {values['heldout_score_at_selection']})")
    print(f"wrote {files.P_TRAIN}, {files.P_HELDOUT}, {files.FOLD_FITS}, {args.reader_path} and {record}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
