"""The kinds of training, and the parts each starts from.

- `adapter.kind: none`: connector training. The writers alone, at learning rate 1e-3, for
  3 epochs; the model is frozen and has no adapter.
- `adapter.kind: consumer`: consumer training. The adapter and the writers, each at 1e-4,
  for 2 epochs; the writers continue from the chosen epoch of the connector training with
  the same signal and seed.
- `adapter.kind: adapter_only`: the adapter alone, at 1e-4, for 2 epochs; no writers.
- `signal: perfect`: the same trainings with `p` the one-hot vector of the situation.

The writers and the adapter start from values drawn after `torch.manual_seed(seed)`, so
two trainings with one seed start alike.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from check2.config.schema import WIDE_WRITER_INPUT, Config
from check2.gates import mechanism_setup as setup
from check2.mechanism.adapter.implementations import IMPLEMENTATIONS
from check2.mechanism import wide_signal
from check2.mechanism.control import ControlState
from check2.mechanism.generate import Pipeline
from check2.mechanism.loading import fast_kernel_importable, require_builtin_path
from check2.mechanism.write import Writers
from check2.training import checkpoints


class TrainingError(RuntimeError):
    """A config that names no training, or a training that cannot go on."""


@dataclass(frozen=True)
class Kind:
    """One kind of training: which parts it trains, at which learning rates, for how many epochs."""

    name: str
    epochs: int
    writers_rate: float | None
    adapter_rate: float | None


KINDS = {
    "none": Kind("connector", 3, 1e-3, None),
    "consumer": Kind("consumer", 2, 1e-4, 1e-4),
    "adapter_only": Kind("adapter_only", 2, None, 1e-4),
}


def kind_of(config: Config) -> Kind:
    """The training a config names, or an error for a setting no training has."""
    if config.signal_forming == "formed_model_only":
        raise TrainingError("signal_forming formed_model_only names the forming training, which "
                            "check2.training.forming.training runs")
    if config.signal not in ("reader", "perfect"):
        raise TrainingError(f"signal {config.signal!r} is an evaluation setting; a training takes reader or perfect")
    if not checkpoints.has_trained_part(config):
        raise TrainingError("the config has no adapter and the write off: nothing is trained")
    return KINDS[config.adapter.kind]


def form_name(config: Config) -> str:
    return "shared_prefix" if config.training.shared_prefix else "whole_sequence"


def load_model(config: Config) -> setup.Bench:
    """The full model in bf16, on the DeltaNet path the config names."""
    if config.training.delta_kernel == "fast_bf16" and not fast_kernel_importable():
        raise TrainingError("delta_kernel fast_bf16, but the fast DeltaNet kernel package cannot be imported")
    if config.training.delta_kernel == "builtin_bf16":
        require_builtin_path()
    return setup.load("full", exact=False)


def forming_factors(config: Config, storage: checkpoints.Storage) -> dict | None:
    """The frozen forming update's factors a training on the formed model carries, or None for any other."""
    if config.signal_forming == "none":
        return None
    from check2.training.forming import training

    return training.frozen_factors(training.forming_config(config), storage=storage)


def build_pipeline(config: Config, kind: Kind, bench: setup.Bench, storage: checkpoints.Storage,
                   connector_epoch: int | None = None, forming: dict | None = None) -> tuple[Pipeline, dict | None]:
    """The pipeline with the parts the training trains at their starting values; and where the writers came from.

    With `forming`, the frozen forming update's factors, the pipeline is the formed model's:
    the update is installed and acts in every pass at the tokens up to the read token.
    """
    torch.manual_seed(config.seed)
    device = next(bench.model.parameters()).device
    control = ControlState()
    writers = adapter = None
    source = None
    if kind.writers_rate is not None:
        writers = (wide_signal.wide_writers(bench.layout, bench.model.config.hidden_size, config.writer.hidden)
                   if config.writer.input == WIDE_WRITER_INPUT else
                   Writers(bench.layout.write_layers, bench.model.config.hidden_size, config.writer.hidden)).to(device)
        if kind.name == "consumer":
            name = checkpoints.connector_id(config)
            epoch = connector_epoch if connector_epoch is not None else checkpoints.chosen_epoch(name)
            writers.load_state_dict(checkpoints.load(name, epoch, storage)["writers"])
            source = {"training_id": name, "epoch": epoch}
    if kind.adapter_rate is not None:
        implementation = config.adapter.implementation
        if implementation not in IMPLEMENTATIONS:
            raise TrainingError(f"adapter implementation {implementation!r}: the config names one of "
                                f"{sorted(IMPLEMENTATIONS)}")
        adapter = IMPLEMENTATIONS[implementation](bench.model, control, config.adapter.rank, config.adapter.alpha_lora,
                                                  bench.layout.adapter_first, bench.layout.adapter_last)
    if forming is not None:
        from check2.training.forming.attach import FormedPipeline, follow_consumer_state, load_frozen

        update = load_frozen(bench.model, bench.layout, config.adapter.implementation, forming)
        pipeline = FormedPipeline(bench.model, bench.tokenizer, control, writers=writers, adapter=adapter,
                                  forming=update)
        pipeline.attached = [follow_consumer_state(update, bench.model, control)]
        return pipeline, source
    return Pipeline(bench.model, bench.tokenizer, control, writers=writers, adapter=adapter), source


def optimizer_groups(kind: Kind, pipeline: Pipeline) -> list[tuple[list[torch.nn.Parameter], float]]:
    groups = []
    if pipeline.adapter is not None:
        groups.append((pipeline.adapter.parameters(), kind.adapter_rate))
    if pipeline.writers is not None:
        groups.append((list(pipeline.writers.parameters()), kind.writers_rate))
    return groups
