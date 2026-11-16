"""What a run's config asks the model to carry: the trained writers, the trained adapter, the reader.

The original model, and the runs built on it (System prompt, Steering), are the setting with
no adapter and `writer.alpha: 0`; they load no trained part. Any other setting with no
adapter is the Connector run, which loads writers. A consumer adapter comes with its writers;
the adapter-only control has no writers, so its config sets `writer.alpha: 0`.

A run of signal forming also carries the frozen forming update and computes `p` live with the
reader fitted on the formed model's states. `formed_model_only` loads nothing else; the
second stage, `formed_with_channel`, also loads its own trained writers and adapter.
"""

from __future__ import annotations

import torch

from check2.config.schema import WIDE_WRITER_INPUT, WRITE_BANDS, Config
from check2.mechanism import wide_signal
from check2.mechanism.adapter.implementations import IMPLEMENTATIONS, build
from check2.mechanism.control import ControlState
from check2.mechanism.generate import Pipeline
from check2.mechanism.loading import Layout
from check2.mechanism.write import Writers


class RunError(ValueError):
    """A config the runner cannot execute as one setting, or a run it refuses to start or resume."""


def needs_writers(config: Config) -> bool:
    kind = config.adapter.kind
    return kind == "consumer" or (kind == "none" and config.writer.alpha != 0)


def needs_adapter(config: Config) -> bool:
    return config.adapter.kind != "none"


def check_executable(config: Config, checkpoint_epoch: int | None) -> None:
    """Refuse a setting this runner does not execute, before anything is loaded or written."""
    if config.adapter.kind == "adapter_only" and config.writer.alpha != 0:
        raise RunError("the adapter-only control has no writers; its config sets writer.alpha: 0")
    if config.signal_forming == "formed_model_only" and (needs_writers(config) or needs_adapter(config)):
        raise RunError("the formed model alone carries the forming update and no writers or adapter")
    if config.signal_forming == "formed_with_channel" and not (needs_writers(config) or needs_adapter(config)):
        raise RunError("the second stage of signal forming carries its own trained writers or adapter")
    if config.signal_forming != "none" and (config.signal != "reader" or config.baseline != type(config.baseline)()):
        raise RunError("a run of signal forming reads the formed reader's p and has no baseline setting")
    if config.reader.input != "conversation":
        raise RunError("a run reads the conversation; the reason-only reader is scored on cached states")
    if needs_adapter(config) and config.adapter.implementation not in IMPLEMENTATIONS:
        raise RunError(f"adapter implementation {config.adapter.implementation!r}: the config writes the chosen "
                       f"implementation out, one of {sorted(IMPLEMENTATIONS)}")
    if config.data.eval_split == "train" and (needs_writers(config) or needs_adapter(config)
                                              or config.baseline != type(config.baseline)()):
        raise RunError("a run over train is the one greedy Original pass for the steering direction")
    if config.signal == "swapped" and not needs_writers(config):
        raise RunError("a swapped signal needs writers to carry it")
    if config.baseline.steering_strength is not None and (needs_writers(config) or needs_adapter(config)):
        raise RunError("Steering is the original model plus the steering vector; it takes no trained part")
    if config.baseline.system_prompt_suffix is not None and (needs_writers(config) or needs_adapter(config)):
        raise RunError("System prompt is the original model with a longer system turn; it takes no trained part")
    if checkpoint_epoch is not None:
        if config.data.eval_split != "heldout":
            raise RunError("a checkpoint epoch is named only for a run over heldout")
        if not (needs_writers(config) or needs_adapter(config)):
            raise RunError("a run with no trained part has no epoch to load")


def trained_parts(config: Config, checkpoint_epoch: int | None) -> tuple[dict | None, dict | None]:
    """The writers' state and the adapter's factors the config names, from their checkpoint."""
    if not (needs_writers(config) or needs_adapter(config)):
        return None, None
    from check2.training.checkpoints import load_for_run

    writers_state, factors = load_for_run(config, checkpoint_epoch)
    if needs_writers(config) and writers_state is None:
        raise RunError("the config names trained writers, and their checkpoint does not exist")
    if needs_adapter(config) and factors is None:
        raise RunError("the config names a trained adapter, and its checkpoint does not exist")
    return (writers_state if needs_writers(config) else None), (factors if needs_adapter(config) else None)


def disabled_layers(config: Config, layout: Layout) -> tuple[int, ...]:
    """The write layers of the switched-off band, on the model's own layout."""
    band = list(config.writer.disabled_layers)
    if not band:
        return ()
    return tuple(layout.bands[[list(b) for b in WRITE_BANDS].index(band)])


def forming_factors(config: Config) -> dict | None:
    """The frozen forming update's factors a run of signal forming carries, or None for any other run."""
    if config.signal_forming == "none":
        return None
    from check2.training.forming import training

    return training.frozen_factors(training.forming_config(config))


def run_reader(config: Config, factors: dict | None):
    """The fitted reader a run computes `p` with: the formed model's for a run of signal forming."""
    from check2.reader import files as reader_files
    from check2.training.forming import guard, refit

    path = refit.READER_WEIGHTS if factors is not None else reader_files.READER_WEIGHTS
    return guard.load_fitted_reader(path, factors)


def build_pipeline(config: Config, model: torch.nn.Module, tokenizer, layout: Layout, reader,
                   writers_state: dict | None, factors: dict | None) -> Pipeline:
    """The pipeline the config describes, with the reader always present so that `p` is computed live."""
    forming = forming_factors(config)
    device = next(model.parameters()).device
    control = ControlState()
    writers = adapter = None
    if writers_state is not None:
        writers = (wide_signal.wide_writers(layout, model.config.hidden_size, config.writer.hidden)
                   if config.writer.input == WIDE_WRITER_INPUT else
                   Writers(layout.write_layers, model.config.hidden_size, config.writer.hidden)).to(device)
        writers.load_state_dict(writers_state)
        writers.eval()
    if factors is not None:
        adapter = build(config.adapter.implementation, model, control, config.adapter.rank, config.adapter.alpha_lora,
                        layout.adapter_first, layout.adapter_last)
        adapter.load_factors(factors)
    if forming is not None:
        from check2.training.forming.attach import FormedPipeline, load_frozen

        update = load_frozen(model, layout, config.adapter.implementation, forming)
        return FormedPipeline(model, tokenizer, control, writers=writers, adapter=adapter, reader=reader,
                              forming=update)
    return Pipeline(model, tokenizer, control, writers=writers, adapter=adapter, reader=reader)
