"""The training loop: connector, consumer, perfect-signal and adapter-only training, each from its config alone.

Which training a config names, and what it starts from, is in `kinds`; what one step
does, and when it stops the run, is in `step`. Each step takes the episodes
`step_batches` draws and runs them in the config's training form. A step that stops the
run keeps the last checkpoint and raises. Each epoch ends with a checkpoint, copied to
Drive, and a measurement record of the mean of each loss term, the seconds per step, the
peak memory and every step that ran out of memory and was run again. A training resumes from its
last complete epoch, with its parts, optimiser, schedule and generators as they were.

A training of record runs through `experiments/run.py train <config>`. A development
run at a smaller size writes only to the scratch folders, under the commit it runs at:

    python -m check2.training.loop <config> --scratch [--episodes N] [--epochs K]
        [--connector-epoch K] [--token-budget N]
"""

from __future__ import annotations

import argparse
import hashlib
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from check2.config.schema import Config, load_config
from check2.data import paths
from check2.gates import mechanism_setup as setup
from check2.mechanism.generate import Pipeline
from check2.record import gpu_name, source_state, write_record
from check2.training import answer_token, checkpoints, inputs, wide_inputs
from check2.training.forms import FORMS
from check2.training.examples import read_targets
from check2.training.kinds import (Kind, build_pipeline, form_name, forming_factors, kind_of, load_model,
                                   optimizer_groups)
from check2.training.sampling import step_batches, steps_per_epoch
from check2.training.schedule import build_optimizer, build_schedule
from check2.training.step import StepStop, run_step

RECORD_NAME = "training_epoch"
# A development run's smaller set of episodes is drawn as a gate draws from a split.
DEVELOPMENT_DRAW_SEED = setup.EPISODE_DRAW_SEED


@dataclass(frozen=True)
class Options:
    """Where a training reads and writes, and, for a development run, its size."""

    storage: checkpoints.Storage | None = None
    episodes_file: Path = inputs.EPISODES_TRAIN
    targets_file: Path = paths.TARGETS
    p_file: Path = inputs.P_TRAIN
    states_file: Path = wide_inputs.STATES_TRAIN
    episode_limit: int | None = None
    epochs: int | None = None
    token_budget: float | None = None
    connector_epoch: int | None = None

    def development(self) -> dict | None:
        size = {"episode_limit": self.episode_limit, "epochs": self.epochs, "connector_epoch": self.connector_epoch}
        return size if any(v is not None for v in size.values()) else None


def _generators(rng: random.Random) -> dict:
    return {"sampling": rng.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_generators(rng: random.Random, saved: dict) -> None:
    rng.setstate(saved["sampling"])
    torch.set_rng_state(saved["torch"])
    if saved["cuda"]:
        torch.cuda.set_rng_state_all(saved["cuda"])


def _state(pipeline: Pipeline, optimizer, schedule, rng: random.Random, step: int, plan: dict) -> dict:
    adapter = None
    if pipeline.adapter is not None:
        adapter = {name: (a.detach().clone(), b.detach().clone())
                   for name, (a, b) in pipeline.adapter.factors().items()}
    return {"writers": None if pipeline.writers is None else pipeline.writers.state_dict(), "adapter": adapter,
            "optimizer": optimizer.state_dict(), "schedule": schedule.state_dict(), "generators": _generators(rng),
            "step": step, "plan": plan}


def _restore(state: dict, pipeline: Pipeline, optimizer, schedule, rng: random.Random) -> None:
    if pipeline.writers is not None:
        pipeline.writers.load_state_dict(state["writers"])
    if pipeline.adapter is not None:
        pipeline.adapter.load_factors(state["adapter"])
    optimizer.load_state_dict(state["optimizer"])
    schedule.load_state_dict(state["schedule"])
    _restore_generators(rng, state["generators"])


def _plan(config: Config, kind: Kind, episodes: list[dict], options: Options, epochs: int,
          per_epoch: int, source: dict | None) -> dict:
    """What makes two runs of a training the same training; a checkpoint resumes only a run with an equal plan."""
    ids = "\n".join(e["episode_id"] for e in episodes)
    return {"training": kind.name, "form": form_name(config), "seed": config.seed,
            "read_position": config.reader.read_position, "episodes": len(episodes),
            "episode_ids_sha256": hashlib.sha256(ids.encode("utf-8")).hexdigest(),
            "episodes_per_step": config.training.episodes_per_step, "steps_per_epoch": per_epoch, "epochs": epochs,
            "targets_sha256": inputs.file_fact(options.targets_file)["sha256"],
            "p_sha256": (inputs.file_fact(options.p_file)["sha256"]
                         if kind.writers_rate is not None and config.signal == "reader" else None),
            "writers_from": source, "development": options.development(),
            **({"answer_token_weight": config.training.answer_token_weight}
               if config.training.answer_token_weight != 0 else {}),
            **({"states_sha256": inputs.file_fact(options.states_file)["sha256"]} if wide_inputs.is_wide(config) else {})}


def _keep_record(record: Path, training_id: str, storage: checkpoints.Storage) -> None:
    """Copy an epoch's record beside the training's checkpoints, outside the folder a send replaces, and to Drive."""
    folder = storage.local_root / training_id / "records"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / record.name).write_bytes(Path(record).read_bytes())
    if storage.drive is not None:
        storage.drive.put(folder, f"{storage.drive_path(training_id)}/records")


def _recorded(config: Config) -> dict:
    """The config as an epoch's record carries it: the answer token's weight is written only when it is not zero,
    so that a record of a training without it reads as it did before the setting existed."""
    out = asdict(config)
    if config.training.answer_token_weight == 0:
        del out["training"]["answer_token_weight"]
    return out


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def train(config: Config, options: Options | None = None, bench: setup.Bench | None = None) -> Path:
    """Run, or resume, the training the config names; returns the folder of its checkpoints on this disk."""
    options = options if options is not None else Options(**inputs.files_for(config))
    kind = kind_of(config)
    name = checkpoints.training_id(config)
    storage = options.storage if options.storage is not None else checkpoints.session_storage()
    episodes = inputs.episodes_for(config, options.episodes_file, options.targets_file)
    if options.episode_limit is not None:
        chosen = set(random.Random(DEVELOPMENT_DRAW_SEED).sample([e["episode_id"] for e in episodes],
                                                                 options.episode_limit))
        episodes = [e for e in episodes if e["episode_id"] in chosen]
    per_step = config.training.episodes_per_step
    per_epoch = steps_per_epoch(len(episodes), per_step)
    epochs = options.epochs if options.epochs is not None else kind.epochs
    total = per_epoch * epochs
    forming = forming_factors(config, storage)
    signals = (inputs.signals_for(config, episodes, options.p_file, forming)
               if kind.writers_rate is not None else None)
    if signals is not None and wide_inputs.is_wide(config):
        signals = wide_inputs.signals_for(config, episodes, options.p_file, options.states_file, forming)
    bench = bench if bench is not None else load_model(config)
    examples = inputs.examples_for(episodes, options.targets_file, bench.tokenizer)
    answer = None
    if config.training.answer_token_weight != 0:
        replies = {i: target["reply"] for i, target in read_targets(options.targets_file).items()}
        answer = answer_token.weighting_for(config.training.answer_token_weight, examples.values(), replies,
                                            bench.tokenizer)
    pipeline, source = build_pipeline(config, kind, bench, storage, options.connector_epoch, forming)
    try:
        form = FORMS[form_name(config)](pipeline)
        budget = {"token_budget": options.token_budget, "record": None}
        if options.token_budget is None:
            budget = inputs.measured_token_budget(form.name, gpu_name(), bench.builtin_path)
        optimizer = build_optimizer(optimizer_groups(kind, pipeline))
        schedule = build_schedule(optimizer, total)
        rng = random.Random(config.seed)
        plan = _plan(config, kind, episodes, options, epochs, per_epoch, source)
        done = checkpoints.complete_epochs(name, storage)
        last = done[-1] if done else 0
        if last:
            state = checkpoints.load(name, last, storage)
            if state.get("plan") != plan or state["step"] != last * per_epoch:
                raise checkpoints.CheckpointError(f"{name} epoch {last} was saved by a run with another plan: "
                                                  f"{state.get('plan')} against {plan}")
            _restore(state, pipeline, optimizer, schedule, rng)
            checkpoints.copy_to_drive(name, last, storage)
            print(f"{name}: resuming after epoch {last} of {epochs}")
        draws = step_batches(episodes, config.seed, max(0, total - last * per_epoch), per_step, generator=rng)
        for epoch in range(last + 1, epochs + 1):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            losses = []
            for index in range(per_epoch):
                step = (epoch - 1) * per_epoch + index
                batch = [examples[e["episode_id"]] for e in next(draws)]
                try:
                    losses.append(run_step(kind, form, batch, signals, config.adapter.beta, budget["token_budget"],
                                           optimizer, schedule, answer))
                except StepStop as err:
                    kept = f"epoch {epoch - 1}" if epoch > 1 else "none"
                    raise type(err)(f"{name}, step {step}: {err}; the run stops; last checkpoint kept: "
                                    f"{kept}") from err
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            checkpoints.save(name, epoch, _state(pipeline, optimizer, schedule, rng, epoch * per_epoch, plan), storage)
            values = {"training_id": name, "training": kind.name, "epoch": epoch, "steps": per_epoch,
                      "mean_loss": {"total": _mean([s.total for s in losses]),
                                    "cross_entropy": _mean([s.cross_entropy for s in losses]),
                                    "kl": _mean([s.kl for s in losses])},
                      "seconds_per_step": seconds / per_epoch,
                      "peak_memory_gib": (torch.cuda.max_memory_allocated() / 2**30
                                          if torch.cuda.is_available() else None),
                      "token_budget": budget,
                      "out_of_memory_events": [{"step": (epoch - 1) * per_epoch + i, **event}
                                               for i, s in enumerate(losses) for event in s.out_of_memory],
                      "step_losses": [{k: v for k, v in asdict(s).items() if k != "out_of_memory"} for s in losses],
                      "plan": plan}
            values["out_of_memory_count"] = len(values["out_of_memory_events"])
            record = write_record("measurement", RECORD_NAME, values, config=_recorded(config),
                                  extra={"device": str(pipeline.device), "builtin_path": bench.builtin_path,
                                         "dtype": str(bench.dtype).replace("torch.", "")})
            _keep_record(record, name, storage)
            print(f"{name}: epoch {epoch} of {epochs}, mean loss {values['mean_loss']}, "
                  f"{values['seconds_per_step']:.3f} s a step, record {record.name}")
        return storage.local_root / name
    finally:
        pipeline.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run a development training at a smaller size.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--scratch", action="store_true", help="write checkpoints to the scratch folders")
    parser.add_argument("--episodes", type=int, default=None, help="train on this many of the training episodes")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--connector-epoch", type=int, default=None,
                        help="consumer training: the connector epoch to continue from, in place of the chosen one")
    parser.add_argument("--token-budget", type=int, default=None)
    args = parser.parse_args(argv)
    if not args.scratch:
        parser.error("a training of record runs through experiments/run.py; this command needs --scratch")
    storage = checkpoints.scratch_storage(source_state()["commit"][:12])
    options = Options(storage=storage, episode_limit=args.episodes, epochs=args.epochs,
                      token_budget=args.token_budget, connector_epoch=args.connector_epoch)
    print(train(load_config(args.config), options))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
