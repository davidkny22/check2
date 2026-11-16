"""Executing one run from its config: the fixed batches, through the split at the read token.

For each batch: the call over tokens `0..r`; `p` from the reader, computed live and recorded
whether or not the run uses it; the signal the config names; generation to the format's cap;
the stated answer; and the appended read, one call over every reply of the batch, which gives
`a2` where the reply stated none and the check where it stated one. Each batch's records are
on disk before the next batch starts, and a run that was stopped resumes after its last
complete batch.

A run whose writers take the wide signal also saves each episode's state at its read token
on the session's disk, under `/content/wide/<run_id>/`, one file per batch; Swap for such a
run gives an episode its donor's state from the states the variant's Check² run saved.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import fields, is_dataclass
from pathlib import Path

import torch

from check2.config.schema import WIDE_WRITER_INPUT, Config, load_config
from check2.evaluation import steering, swap
from check2.evaluation.runs import decoding
from check2.evaluation.runs import episodes as run_episodes
from check2.evaluation.runs.output import RunOutput
from check2.evaluation.runs.parts import (RunError, build_pipeline, check_executable, disabled_layers,
                                          trained_parts)
from check2.evaluation.scoring import score_stated
from check2.evaluation.tables import load_run
from check2.mechanism.generate import REPLY_CAPS, Pipeline
from check2.mechanism.loading import Layout

WIDE_STATES_ROOT = Path("/content/wide")  # the session's own disk; never committed


def wide_states_folder(run_id: str) -> Path:
    return WIDE_STATES_ROOT / run_id


def save_wide_states(run_id: str, index: int, episode_ids: list[str], states: torch.Tensor) -> Path:
    """One batch's read states, as the run computed them live, in the model's number format."""
    folder = wide_states_folder(run_id)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"batch-{index:05d}.pt"
    torch.save({"episode_ids": list(episode_ids), "states": states.detach().cpu().clone()}, path)
    return path


def load_wide_states(run_id: str) -> dict[str, torch.Tensor]:
    """Every episode's read state a wide run saved, by episode id."""
    out: dict[str, torch.Tensor] = {}
    for path in sorted(wide_states_folder(run_id).glob("batch-*.pt")):
        saved = torch.load(path, weights_only=True)
        for i, eid in enumerate(saved["episode_ids"]):
            out[eid] = saved["states"][i]
    return out


def run_id_for(config_path: Path, config: Config, checkpoint_epoch: int | None) -> str:
    """The config file's name, with `-heldout` for a run over heldout and `-epoch<k>` for one epoch's checkpoint."""
    run_id = Path(config_path).stem
    if config.data.eval_split == "heldout":
        run_id += "-heldout"
    if checkpoint_epoch is not None:
        run_id += f"-epoch{checkpoint_epoch}"
    return run_id


def settings_of(config: Config) -> dict:
    """The resolved config as the file format writes it, under its one top-level key.

    A setting marked to be omitted at its default is written only when it differs from it.
    """
    def plain(node):
        if is_dataclass(node):
            return {f.metadata.get("key", f.name): plain(getattr(node, f.name)) for f in fields(node)
                    if not (f.metadata.get("omit_default") and getattr(node, f.name) == f.default)}
        if isinstance(node, tuple):
            return [plain(v) for v in node]
        return node
    return {"check2": plain(config)}


def _row(episode: dict, reply, stated: str | None, appended: str, p_reader, p_used, part: str | None,
         donor: str | None) -> dict:
    return {
        "episode_id": episode["episode_id"], "qid": episode["qid"], "form": episode["form"],
        "format": episode["format"], "situation": episode["situation"], "a1": episode["a1"],
        "gold": episode["gold"], "target": episode["target"],
        "a2": stated if stated is not None else appended,
        "scored_by": "stated" if stated is not None else "appended",
        "ended": reply.ended, "reply": reply.text, "reply_tokens": len(reply.token_ids),
        "appended_read": appended,
        "p": [float(x) for x in p_reader],
        "p_used": None if p_used is None else [float(x) for x in p_used],
        "part": part,
        "donor": donor,
    }


def run_batch(pipeline: Pipeline, config: Config, batch: list[dict], disabled: tuple[int, ...],
              part_of: dict[str, str], source, donors: dict | None = None,
              room=None, donor_states: dict | None = None,
              kept_states: list | None = None) -> tuple[list[dict], float]:
    """One batch through the split at the read token; its records and the seconds its replies took.

    With `donors`, the Swap run's draw, each listed episode carries its donor's `p`. With
    `room`, the preallocated cache's, the appended read continues from a call over `0..r`
    of its own, sized for its texts. With wide writers, the batch's own read states are
    appended to `kept_states`, and with `donor_states` each listed episode carries its
    donor's state as well as its donor's `p`.
    """
    fmt = batch[0]["format"]
    alpha = config.writer.alpha
    started = time.perf_counter()
    prefilled = pipeline.prefill(batch)
    p, p_reader = pipeline.signal(prefilled, source)
    if p_reader is None:
        raise RunError("every run computes p live; the pipeline has no reader")
    donor_ids: list[str | None] = [None] * len(batch)
    if pipeline.wide and kept_states is not None:
        kept_states.append(prefilled.read_state.detach().cpu().clone())
    if donors is not None:
        p, donor_ids = swap.swapped(p, batch, donors)
        if pipeline.wide:
            prefilled.read_state = swap.swapped_states(prefilled.read_state, batch, donors, donor_states)
    used = p if pipeline.writers is not None else None
    replies = pipeline.generate(prefilled, used, REPLY_CAPS[fmt], alpha, disabled)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    if room is not None:
        prefilled = decoding.prefill_for_appended_read(pipeline, batch, replies, room)
        if donors is not None and pipeline.wide:
            prefilled.read_state = swap.swapped_states(prefilled.read_state, batch, donors, donor_states)
    appended = pipeline.appended_read(prefilled, used, replies, alpha, disabled)
    p_reader, used = p_reader.cpu(), None if used is None else used.cpu()
    rows = [_row(e, reply, score_stated(reply.text, fmt), appended[b], p_reader[b],
                 None if used is None else used[b], part_of.get(e["qid"]), donor_ids[b])
            for b, (e, reply) in enumerate(zip(batch, replies))]
    return rows, seconds


def _signal_source(config: Config):
    """The reader's `p`, the one-hot situation, or, for Swap, the reader's `p` with donors put in."""
    return "perfect" if config.signal == "perfect" else "reader"


def _donors(config: Config, batches: list[list[dict]]) -> tuple[dict | None, dict | None]:
    """The Swap draw, and for the wide-signal variant the donors' saved read states."""
    if config.signal != "swapped":
        return None, None
    folder = swap.check2_run(config.data.eval_split, writer_input=config.writer.input)
    _, check2_records = load_run(folder)
    donors = swap.draw_donors([e for batch in batches for e in batch], check2_records)
    return donors, (load_wide_states(folder.name) if config.writer.input == WIDE_WRITER_INPUT else None)


def run(config: Config, run_id: str, model: torch.nn.Module, tokenizer, layout: Layout, reader,
        checkpoint_epoch: int | None = None, episode_folder: Path | None = None) -> Path:
    """Execute the run on a loaded model and reader; returns the run's folder once it is complete."""
    check_executable(config, checkpoint_epoch)
    writers_state, factors = trained_parts(config, checkpoint_epoch)
    source = _signal_source(config)
    batches = run_episodes.order(run_episodes.select(config, episode_folder))
    donors, donor_states = _donors(config, batches)
    expected_ids = [e["episode_id"] for batch in batches for e in batch]
    order_hash = run_episodes.order_sha256(batches)
    suffix = config.baseline.system_prompt_suffix
    output = RunOutput(run_id, settings_of(config), order_hash, len(expected_ids))
    part_of = run_episodes.parts()
    disabled = disabled_layers(config, layout)
    pipeline = build_pipeline(config, model, tokenizer, layout, reader, writers_state, factors)
    try:
        strength = config.baseline.steering_strength
        vectors = None
        if strength is not None:
            vectors, direction_sha = steering.load_direction()
            output.note("steering_direction_sha256", direction_sha)
        with contextlib.ExitStack() as stack:
            kept = None
            if config.data.eval_split == "train":
                kept = stack.enter_context(steering.read_token_states(model, pipeline.control))
            for index, batch in enumerate(batches):
                if index < output.batches_done:
                    continue
                if suffix is not None:
                    batch = run_episodes.with_system_suffix(batch, suffix)
                # Steering adds the direction of this batch's own format, for every call the batch makes.
                steer = (contextlib.nullcontext() if vectors is None
                         else steering.steered(model, pipeline.control, vectors, batch, float(strength)))
                with steer, decoding.settings_on(config, pipeline, batch, REPLY_CAPS[batch[0]["format"]]) as room:
                    kept_states: list = []
                    rows, seconds = run_batch(pipeline, config, batch, disabled, part_of, source, donors, room,
                                              donor_states, kept_states)
                if pipeline.wide:
                    save_wide_states(run_id, index, [e["episode_id"] for e in batch], kept_states[0])
                if kept is not None:
                    # Under the preallocated cache the appended read makes the second call over 0..r.
                    calls = 1 if room is None else 2
                    if len(kept) != calls:
                        raise RunError(f"expected {calls} call(s) over tokens 0..r in the batch, saw {len(kept)}")
                    steering.save_batch_states(run_id, index, [e["episode_id"] for e in batch], kept[0])
                    kept.clear()
                output.append(rows, batch[0]["format"], seconds)
    except BaseException:
        output.fail(sum(len(b) for b in batches[:output.batches_done]))
        raise
    finally:
        pipeline.close()
    return output.finish(expected_ids)


def execute(config, checkpoint_epoch: int | None = None) -> Path:
    """Run the config file `config` on the full model at its pinned revision; returns the run's folder.

    The config is loaded for a run, so its model revision and read position are resolved, and
    the runtime's versions are asserted against `experiments/pins.json` before the model loads.
    A checkpoint epoch is named only for a run over heldout.
    """
    path = Path(config)
    settings = load_config(path)
    check_executable(settings, checkpoint_epoch)
    run_id = run_id_for(path, settings, checkpoint_epoch)
    from check2.mechanism.loading import load_full
    from check2.evaluation.runs.parts import forming_factors, run_reader

    model, tokenizer, layout = load_full(torch.bfloat16)
    if settings.blas == "cublaslt":
        torch.backends.cuda.preferred_blas_library("cublaslt")
    reader = run_reader(settings, forming_factors(settings)).to(next(model.parameters()).device)
    reader.eval()
    if tuple(reader.layers) != tuple(settings.reader.layers):
        raise RunError(f"the reader reads layers {list(reader.layers)}, the config names "
                       f"{list(settings.reader.layers)}")
    return run(settings, run_id, model, tokenizer, layout, reader, checkpoint_epoch)
