"""The forming training: the forming update and a copy of the selected reader, trained together.

The update starts at its initial values, drawn after `torch.manual_seed(seed)`; the reader
copy starts from the selected reader's weights, the reader fitted on all of `train`. Both
train at learning rate 1e-4 for 2 epochs on the reader's own training episodes, every
`train` episode of the three forms, 16 a step, drawn uniformly. The frozen model runs in
bf16 on whole prompts; the trained parameters are float32; the optimiser, the warm-up and
cosine decay, and the clipping of every trained parameter together are the training loop's.
Each step runs as forward batches within the token budget, with gradients accumulated.
When a forward batch runs out of memory, the step clears its accumulated gradients, halves
the token budget for this step only and runs again from its start with the same episodes,
as the training loop's step does; each such event is counted in the epoch's record. A
batch of one row that runs out of memory, a row that no longer fits the halved budget, or a
loss or gradient norm that is not finite stops the run before the optimiser step. Each
epoch ends with a checkpoint through the training loop's checkpoint functions, holding the
update's factors as the adapter and the reader copy's weights, and a measurement record.
A training resumes from its last complete epoch. At the end the update is frozen and the
reader copy is discarded.

The forming training of record is `train(config)`, called by the runner. A development run
at a smaller size writes only to the scratch folders, under the commit it runs at:

    python -m check2.training.forming.training <config> --scratch [--episodes N] [--epochs K]
        [--token-budget N]
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import math
import random
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch

from check2.config.schema import Config, load_config
from check2.data.episodes import END_OF_TURN
from check2.gates import mechanism_setup as setup
from check2.reader import files as reader_files
from check2.reader.crossfit import load_reader
from check2.reader.model import OUTPUTS, Reader
from check2.reader.train import class_weights
from check2.record import gpu_name, source_state, write_record
from check2.training import checkpoints, inputs
from check2.training.batches import BatchError, plan_batches
from check2.training.forming.draws import step_draws
from check2.training.forming.objective import FormingExample, batch_tokens, forming_step, make_example
from check2.training.forming.update import FormingUpdate, layers_for
from check2.training.kinds import TrainingError, load_model
from check2.training.sampling import steps_per_epoch
from check2.training.schedule import build_optimizer, build_schedule, clip
from check2.training.step import NonFiniteLoss, OutOfMemory, StepStop

LEARNING_RATE = 1e-4
EPOCHS = 2
RECORD_NAME = "forming_epoch"
# The forward batches hold whole prompts, as the whole-sequence form holds whole sequences.
BUDGET_FORM = "whole_sequence"
DEVELOPMENT_DRAW_SEED = setup.EPISODE_DRAW_SEED


@dataclass(frozen=True)
class Options:
    """Where the forming training reads and writes, and, for a development run, its size."""

    storage: checkpoints.Storage | None = None
    episodes_file: Path = inputs.EPISODES_TRAIN
    reader_path: Path = reader_files.READER_WEIGHTS
    episode_limit: int | None = None
    epochs: int | None = None
    token_budget: float | None = None

    def development(self) -> dict | None:
        size = {"episode_limit": self.episode_limit, "epochs": self.epochs}
        return size if any(v is not None for v in size.values()) else None


@dataclass
class StepLosses:
    """One step's losses, each the mean over its episodes, the gradient norm before clipping,
    and each out-of-memory event: the budget and the rows of the batch that ran out."""

    total: float
    cross_entropy: float
    kl: float
    gradient_norm: float
    out_of_memory: list[dict] = field(default_factory=list)


def training_id(config: Config) -> str:
    """The forming training's name, from the settings that distinguish it."""
    if config.signal_forming != "formed_model_only":
        raise TrainingError(f"signal_forming {config.signal_forming!r} names no forming training; it is "
                            "formed_model_only")
    return f"forming-rank{config.adapter.rank}-seed{config.seed}"


def forming_config(config: Config) -> Config:
    """The formed model's settings for a run or training of either stage: the same seed and rank, no other part."""
    return replace(config, signal_forming="formed_model_only", writer=replace(config.writer, alpha=0.0),
                   adapter=replace(config.adapter, kind="none"))


def training_episodes(config: Config, episodes_file: Path) -> list[dict]:
    """The reader's own training episodes, every `train` episode, sorted by id, at the config's read position."""
    position = config.reader.read_position
    if not isinstance(position, int) or isinstance(position, bool):
        raise TrainingError(f"the read position {position!r} is not resolved; load the config for a run")
    chosen = [e for e in inputs.read_jsonl(episodes_file) if e["split"] == "train"]
    return sorted((inputs.at_read_position(e, position) for e in chosen), key=lambda e: e["episode_id"])


def _attempt(model: torch.nn.Module, update: FormingUpdate, reader: Reader, examples: list[FormingExample],
             weights: torch.Tensor, pad_id: int, token_budget: float, failed: dict) -> dict:
    """Run the step's forward batches, accumulating gradients; returns the summed episode losses."""
    sums = {"total": 0.0, "cross_entropy": 0.0, "kl": 0.0}
    for part in plan_batches(examples, token_budget, batch_tokens):
        failed["rows"] = len(part)
        losses = forming_step(model, update, reader, part, weights, len(examples), pad_id)
        for key in sums:
            sums[key] += float(getattr(losses, key).double().sum())
    return sums


def _free(update: FormingUpdate, optimizer: torch.optim.Optimizer) -> None:
    """Clear the step's accumulated gradients and the update's state, and return the freed memory to the card."""
    optimizer.zero_grad(set_to_none=True)
    update.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_step(model: torch.nn.Module, update: FormingUpdate, reader: Reader, examples: list[FormingExample],
             weights: torch.Tensor, pad_id: int, token_budget: float, optimizer, schedule) -> StepLosses:
    """One optimiser step over a step's episodes, in forward batches within the budget.

    The budget is halved for this step each time a forward batch runs out of memory.
    """
    size = len(examples)
    budget = token_budget
    events: list[dict] = []
    optimizer.zero_grad(set_to_none=True)
    while True:
        failed: dict = {}
        out_of_memory = False
        try:
            sums = _attempt(model, update, reader, examples, weights, pad_id, budget, failed)
        except torch.OutOfMemoryError:
            out_of_memory = True
        # Memory is freed outside the handler, so that the error's frames no longer hold it.
        if not out_of_memory:
            break
        _free(update, optimizer)
        events.append({"token_budget": budget, "rows": failed.get("rows")})
        if failed.get("rows") == 1:
            raise OutOfMemory(f"a forward batch of one row ran out of memory at a token budget of {budget}")
        budget = budget / 2
        try:
            plan_batches(examples, budget, batch_tokens)
        except BatchError as err:
            raise OutOfMemory(f"after running out of memory, a row does not fit half the budget: {err}") from err
    out = StepLosses(total=sums["total"] / size, cross_entropy=sums["cross_entropy"] / size, kl=sums["kl"] / size,
                     gradient_norm=math.nan, out_of_memory=events)
    if not math.isfinite(out.total):
        raise NonFiniteLoss(f"the step's loss is {out.total}")
    out.gradient_norm = float(clip(optimizer))
    if not math.isfinite(out.gradient_norm):
        raise NonFiniteLoss(f"the step's gradient norm is {out.gradient_norm}")
    optimizer.step()
    schedule.step()
    return out


def _generators(rng: random.Random) -> dict:
    return {"sampling": rng.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_generators(rng: random.Random, saved: dict) -> None:
    rng.setstate(saved["sampling"])
    torch.set_rng_state(saved["torch"])
    if saved["cuda"]:
        torch.cuda.set_rng_state_all(saved["cuda"])


def _factors(update: FormingUpdate) -> dict:
    return {name: (a.detach().clone(), b.detach().clone()) for name, (a, b) in update.factors().items()}


def _plan(config: Config, episodes: list[dict], options: Options, epochs: int, per_epoch: int,
          reader_sha: str) -> dict:
    """What makes two runs of the forming training the same; a checkpoint resumes only a run with an equal plan."""
    ids = "\n".join(e["episode_id"] for e in episodes)
    return {"training": "forming", "seed": config.seed, "read_position": config.reader.read_position,
            "episodes": len(episodes), "episode_ids_sha256": hashlib.sha256(ids.encode("utf-8")).hexdigest(),
            "episodes_per_step": config.training.episodes_per_step, "steps_per_epoch": per_epoch, "epochs": epochs,
            "reader_sha256": reader_sha, "development": options.development()}


def _keep_record(record: Path, name: str, storage: checkpoints.Storage) -> None:
    """Copy an epoch's record beside the checkpoints, outside the folder a send replaces, and to Drive."""
    folder = storage.local_root / name / "records"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / record.name).write_bytes(Path(record).read_bytes())
    if storage.drive is not None:
        storage.drive.put(folder, f"{storage.drive_path(name)}/records")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def train(config: Config, options: Options | None = None, bench: setup.Bench | None = None) -> Path:
    """Run, or resume, the forming training; returns the folder of its checkpoints on this disk."""
    options = options if options is not None else Options()
    name = training_id(config)
    storage = options.storage if options.storage is not None else checkpoints.session_storage()
    episodes = training_episodes(config, options.episodes_file)
    weights = class_weights([OUTPUTS.index(e["situation"]) for e in episodes])
    if options.episode_limit is not None:
        chosen = set(random.Random(DEVELOPMENT_DRAW_SEED).sample([e["episode_id"] for e in episodes],
                                                                 options.episode_limit))
        episodes = [e for e in episodes if e["episode_id"] in chosen]
    per_step = config.training.episodes_per_step
    per_epoch = steps_per_epoch(len(episodes), per_step)
    epochs = options.epochs if options.epochs is not None else EPOCHS
    total = per_epoch * epochs
    bench = bench if bench is not None else load_model(config)
    device = next(bench.model.parameters()).device
    pad_id = bench.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = bench.tokenizer.convert_tokens_to_ids(END_OF_TURN)
    examples = {e["episode_id"]: make_example(e, bench.tokenizer) for e in episodes}
    # The copy starts from the reader fitted on all of train, on the original model's states.
    reader = load_reader(options.reader_path).to(device).train()
    reader_sha = hashlib.sha256(Path(options.reader_path).read_bytes()).hexdigest()
    torch.manual_seed(config.seed)
    update = FormingUpdate(bench.model, config.adapter.implementation, *layers_for(bench.layout),
                           config.adapter.rank, config.adapter.alpha_lora)
    update.install()
    try:
        budget = {"token_budget": options.token_budget, "record": None}
        if options.token_budget is None:
            budget = inputs.measured_token_budget(BUDGET_FORM, gpu_name(), bench.builtin_path)
        optimizer = build_optimizer([(update.parameters(), LEARNING_RATE), (list(reader.parameters()), LEARNING_RATE)])
        schedule = build_schedule(optimizer, total)
        rng = random.Random(config.seed)
        plan = _plan(config, episodes, options, epochs, per_epoch, reader_sha)
        done = checkpoints.complete_epochs(name, storage)
        last = done[-1] if done else 0
        if last:
            state = checkpoints.load(name, last, storage)
            if state.get("plan") != plan or state["step"] != last * per_epoch:
                raise checkpoints.CheckpointError(f"{name} epoch {last} was saved by a run with another plan: "
                                                  f"{state.get('plan')} against {plan}")
            update.load_factors(state["adapter"])
            reader.load_state_dict(state["reader"])
            optimizer.load_state_dict(state["optimizer"])
            schedule.load_state_dict(state["schedule"])
            _restore_generators(rng, state["generators"])
            checkpoints.copy_to_drive(name, last, storage)
            print(f"{name}: resuming after epoch {last} of {epochs}")
        draws = step_draws(episodes, config.seed, max(0, total - last * per_epoch), per_step, generator=rng)
        for epoch in range(last + 1, epochs + 1):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            losses = []
            for index in range(per_epoch):
                step = (epoch - 1) * per_epoch + index
                batch = [examples[e["episode_id"]] for e in next(draws)]
                try:
                    losses.append(run_step(bench.model, update, reader, batch, weights, pad_id,
                                           budget["token_budget"], optimizer, schedule))
                except StepStop as err:
                    kept = f"epoch {epoch - 1}" if epoch > 1 else "none"
                    raise type(err)(f"{name}, step {step}: {err}; the run stops; last checkpoint kept: "
                                    f"{kept}") from err
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            state = {"writers": None, "adapter": _factors(update), "optimizer": optimizer.state_dict(),
                     "schedule": schedule.state_dict(), "generators": _generators(rng), "step": epoch * per_epoch,
                     "reader": {k: v.detach().cpu() for k, v in reader.state_dict().items()}, "plan": plan}
            checkpoints.save(name, epoch, state, storage)
            values = {"training_id": name, "training": "forming", "epoch": epoch, "steps": per_epoch,
                      "mean_loss": {key: _mean([getattr(s, key) for s in losses])
                                    for key in ("total", "cross_entropy", "kl")},
                      "first_step": asdict(losses[0]), "last_step": asdict(losses[-1]),
                      "seconds_per_step": seconds / per_epoch,
                      "peak_memory_gib": (torch.cuda.max_memory_allocated() / 2**30
                                          if torch.cuda.is_available() else None),
                      "update_factors_sha256": update.factors_sha256(), "token_budget": budget,
                      "out_of_memory_events": [{"step": (epoch - 1) * per_epoch + i, **event}
                                               for i, s in enumerate(losses) for event in s.out_of_memory],
                      "step_losses": [{k: v for k, v in asdict(s).items() if k != "out_of_memory"} for s in losses],
                      "plan": plan}
            values["out_of_memory_count"] = len(values["out_of_memory_events"])
            record = write_record("measurement", RECORD_NAME, values, config=asdict(config),
                                  extra={"device": str(device), "builtin_path": bench.builtin_path,
                                         "dtype": str(bench.dtype).replace("torch.", "")})
            _keep_record(record, name, storage)
            print(f"{name}: epoch {epoch} of {epochs}, mean loss {values['mean_loss']}, "
                  f"{values['seconds_per_step']:.3f} s a step, record {record.name}")
        update.freeze()
        return storage.local_root / name
    finally:
        update.remove()


def frozen_factors(config: Config, epoch: int = EPOCHS, storage: checkpoints.Storage | None = None) -> dict:
    """The update's factors as the forming training left them at `epoch`, its last by default."""
    return checkpoints.load(training_id(config), epoch, storage)["adapter"]


def storage_for(scratch_tag: str | None) -> checkpoints.Storage:
    """The checkpoints of record, or a development run's under the commit tag it was sent at."""
    return checkpoints.scratch_storage(scratch_tag) if scratch_tag else checkpoints.session_storage()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--scratch", action="store_true", help="write checkpoints to the scratch folders")
    parser.add_argument("--episodes", type=int, default=None, help="train on this many of the training episodes")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--token-budget", type=int, default=None)
    args = parser.parse_args(argv)
    if not args.scratch:
        parser.error("a training of record runs through experiments/run.py; this command needs --scratch")
    storage = checkpoints.scratch_storage(source_state()["commit"][:12])
    options = Options(storage=storage, episode_limit=args.episodes, epochs=args.epochs,
                      token_budget=args.token_budget)
    print(train(load_config(args.config), options))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
