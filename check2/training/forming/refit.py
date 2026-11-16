"""The formed model's hidden states, and the reader fitted on them from scratch.

With the frozen forming update installed, the reader's own cache code saves the read
layers' states of every `train` and `heldout` episode. The selected reader variant is then
fitted from scratch on them at the selected read position, stopping when the `heldout` loss
has not improved for three epochs and keeping the best epoch; its `heldout` outputs come
from that fit. `train` is cross-fitted in the five folds with that epoch count, exactly as
for the original model. The outputs are saved under `data/reader/formed/`, and the fitted
reader beside the original one on the session, each folder with the record that names the
forming training and the SHA-256 of the update's factors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from check2.data.episodes import TAIL_LENGTH, check_tail_position
from check2.reader import files
from check2.reader.cache import BATCH_SIZE, READ_LAYERS, cache_states
from check2.reader.crossfit import cross_fit, save_reader, scores_of
from check2.reader.inputs import Inputs
from check2.reader.train import Fit, fit, predict
from check2.training.forming import guard
from check2.training.forming.update import FormingUpdate

OUTPUTS = files.READER_DATA / "formed"
P_TRAIN = OUTPUTS / "p_train.jsonl"
P_HELDOUT = OUTPUTS / "p_heldout.jsonl"
FOLD_FITS = OUTPUTS / "fold_fits.json"
CACHE_INDEX = OUTPUTS / "cache_index.json"
SESSION_FOLDER = files.SESSION_ROOT / "formed"
CACHE = SESSION_FOLDER / "cache"
READER_WEIGHTS = SESSION_FOLDER / "reader.pt"


class FormedForward(torch.nn.Module):
    """The model called on whole left-padded prompts with the forming update acting on tokens `0..r`.

    The read token is the template tail's token at `tail_position`, so in a left-padded
    batch of whole prompts it sits in the same column of every row. This is the callable the
    reader's cache code runs.
    """

    def __init__(self, model: torch.nn.Module, update: FormingUpdate, tail_position: int) -> None:
        super().__init__()
        check_tail_position(tail_position)
        self.model = model
        self.update = update
        self.tail_position = tail_position

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        width = attention_mask.shape[1]
        read_column = torch.full((attention_mask.shape[0],), width - TAIL_LENGTH + self.tail_position,
                                 device=attention_mask.device)
        self.update.set_call(attention_mask, read_column, input_ids.shape[1])
        try:
            return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        finally:
            self.update.clear()


def cache_formed_states(model: torch.nn.Module, tokenizer, update: FormingUpdate, episodes: list[dict],
                        out_dir: str | Path, tail_position: int, layers=READ_LAYERS,
                        batch_size: int = BATCH_SIZE) -> dict[str, list[int]]:
    """The reader's cache of the formed model's states, by the reader's own cache code; returns the index."""
    return cache_states(FormedForward(model, update, tail_position), tokenizer, episodes, out_dir, layers,
                        batch_size)


@dataclass
class FormedFit:
    """The reader fitted on formed states, and its outputs."""

    fit: Fit
    p_train: dict[str, list[float]]
    p_heldout: dict[str, list[float]]
    fold_fits: list[dict]


def refit(inputs: Inputs, folds: list[list[str]], variant: str, tail_position: int,
          seed: int = files.SEED) -> FormedFit:
    """Fit the variant from scratch with the stopping rule, then cross-fit `train` with the epoch it kept."""
    first = fit(variant, inputs.bank, inputs.bank_layers, inputs.train_ids, inputs.label_of, tail_position, seed,
                heldout_ids=inputs.heldout_ids)
    p_heldout = dict(zip(inputs.heldout_ids, predict(first.reader, inputs.bank, inputs.bank_layers,
                                                    inputs.heldout_ids, tail_position).tolist()))
    p_train, fold_fits = cross_fit(inputs, folds, variant, tail_position, first.best_epoch, seed)
    return FormedFit(fit=first, p_train=p_train, p_heldout=p_heldout, fold_fits=fold_fits)


def save_outputs(result: FormedFit, training_id: str, factors, outputs: Path = OUTPUTS,
                 reader_path: Path = READER_WEIGHTS) -> None:
    """Write the outputs and the fitted reader, each folder with its record of the forming update."""
    files.write_jsonl(outputs / P_TRAIN.name, files.p_rows(result.p_train))
    files.write_jsonl(outputs / P_HELDOUT.name, files.p_rows(result.p_heldout))
    files.write_json(outputs / FOLD_FITS.name, result.fold_fits)
    guard.write_source(outputs, training_id, factors)
    save_reader(result.fit.reader, reader_path)
    guard.write_source(Path(reader_path).parent, training_id, factors)


def summary(result: FormedFit, train: list[dict], heldout: list[dict], label_of: dict[str, str]) -> dict:
    """The formed reader's scores, for its measurement record."""
    return {"epochs": result.fit.best_epoch, "epochs_run": result.fit.epochs_run,
            "heldout_loss": result.fit.heldout_loss, "train_loss": result.fit.train_loss,
            "train_cross_fitted": scores_of(result.p_train, train, label_of),
            "heldout": scores_of(result.p_heldout, heldout, label_of),
            "folds": [{k: v for k, v in f.items() if not k.endswith("_questions")} for f in result.fold_fits]}


def main(argv: list[str] | None = None) -> int:
    """Cache the formed model's states, fit the reader on them, cross-fit, and save it all."""
    import argparse

    from check2.config.schema import load_config
    from check2.data.splits import load_splits
    from check2.mechanism.loading import FULL_LAYOUT
    from check2.reader.inputs import load_inputs
    from check2.reader.original import load_original, load_tokenizer
    from check2.record import write_record
    from check2.training.forming import training
    from check2.training.forming.update import layers_for

    parser = argparse.ArgumentParser(description="Fit the reader again on the formed model's states.")
    parser.add_argument("config", help="the formed model's config, signal_forming: formed_model_only")
    parser.add_argument("--epoch", type=int, default=training.EPOCHS, help="the forming checkpoint to load")
    parser.add_argument("--scratch-tag", default=None, help="load a development run's forming checkpoint")
    parser.add_argument("--per-format", type=int, default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    name = training.training_id(config)
    factors = training.frozen_factors(config, args.epoch, training.storage_for(args.scratch_tag))
    selection = files.read_json(files.SELECTION)
    variant, position = selection["variant"], selection["read_position"]
    if position != config.reader.read_position:
        raise ValueError(f"the selection's read position {position} is not the config's "
                         f"{config.reader.read_position}")
    model, tokenizer = load_original(), load_tokenizer()
    update = FormingUpdate(model, config.adapter.implementation, *layers_for(FULL_LAYOUT), config.adapter.rank,
                           config.adapter.alpha_lora)
    update.load_factors(factors)
    update.freeze()
    update.install()
    train = files.small_draw(files.load_split("train"), args.per_format)
    heldout = files.small_draw(files.load_split("heldout"), args.per_format)
    try:
        index = cache_formed_states(model, tokenizer, update, train + heldout, CACHE, position)
    finally:
        update.remove()
    files.write_json(CACHE_INDEX, index)
    inputs = load_inputs(CACHE, train, heldout)
    result = refit(inputs, load_splits().folds, variant, position)
    save_outputs(result, name, factors)
    values = {"training_id": name, "forming_epoch": args.epoch, "scratch_tag": args.scratch_tag,
              "update_factors_sha256": update.factors_sha256(), "variant": variant, "read_position": position,
              "per_format": args.per_format, "episodes": {"train": len(train), "heldout": len(heldout)},
              **summary(result, train, heldout, inputs.label_of), "reader_path": str(READER_WEIGHTS)}
    record = write_record("measurement", "formed_reader", values)
    print(f"formed reader: cross-fitted train {values['train_cross_fitted']}; heldout {values['heldout']}; "
          f"epochs {values['epochs']}; record {record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
