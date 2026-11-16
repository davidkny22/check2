"""The training loop on the tiny model: what it trains, how it resumes, and how it stops."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest
import torch

from check2 import record
from check2.config.schema import Config
from check2.mechanism.generate import one_hot_situations
from check2.training import checkpoints, inputs, loop, step
from check2.training.kinds import TrainingError
from check2.training.objectives import EpisodeLosses
from check2.training.sampling import steps_per_epoch
from tests.evaluation import faults
from tests.training.fake_drive import FolderDrive

EPISODES = 32  # two steps an epoch
EPOCHS = 2

# The specification's sentences on each training, written out: its epochs and the rate of each trained part.
# Connector training: "Learning rate 1e-3, 3 epochs." Consumer training: "the adapter (learning rate 1e-4) and
# the writers, continued from connector training (learning rate 1e-4). 2 epochs." Adapter-only control: "same
# gating, targets, sampling, steps and learning rate" as the consumer's adapter.
STATED = {
    "none": {"epochs": 3, "rates": {"writers": 1e-3}},
    "consumer": {"epochs": 2, "rates": {"adapter": 1e-4, "writers": 1e-4}},
    "adapter_only": {"epochs": 2, "rates": {"adapter": 1e-4}},
}
# "a linear warm-up over the first 50 steps, then cosine decay"; "gradient clipping at norm 1.0".
WARMUP = 50
CLIP_NORM = 1.0
# A training that leaves the warm-up: two epochs of 30 steps of 16 episodes.
LONG_STEPS_PER_EPOCH = 30
LONG_EPOCHS = 2

# The faults the two reviews of the training loop planted, each the change one reviewer's patch makes,
# planted through `tests.evaluation.faults`. Only `kind_of` reads the table of kinds, so a change to the
# table is planted in `kind_of`.
_KIND_LOOKUP = "    return KINDS[config.adapter.kind]\n"


def _kinds_changed(changes: str) -> str:
    return f"    return {{**KINDS, {changes}}}[config.adapter.kind]\n"


_PLAN_CHECK = ('            if state.get("plan") != plan or state["step"] != last * per_epoch:\n'
               '                raise checkpoints.CheckpointError(f"{name} epoch {last} was saved by a run with '
               'another plan: "\n'
               '                                                  f"{state.get(\'plan\')} against {plan}")\n')
_STEP_CALL_END = "optimizer, schedule, answer))\n"
LOOP_FAULTS = {
    "connector-rate-a-tenth": ("check2.training.kinds", "kind_of", _KIND_LOOKUP,
                               _kinds_changed('"none": Kind("connector", 3, 1e-4, None)')),
    "consumer-writers-rate-ten-times": ("check2.training.kinds", "kind_of", _KIND_LOOKUP,
                                        _kinds_changed('"consumer": Kind("consumer", 2, 1e-3, 1e-4)')),
    "connector-and-consumer-epochs-swapped": (
        "check2.training.kinds", "kind_of", _KIND_LOOKUP,
        _kinds_changed('"none": Kind("connector", 2, 1e-3, None), "consumer": Kind("consumer", 3, 1e-4, 1e-4)')),
    "rates-and-epochs-swapped": (
        "check2.training.kinds", "kind_of", _KIND_LOOKUP,
        _kinds_changed('"none": Kind("connector", 2, 1e-4, None), "consumer": Kind("consumer", 3, 1e-4, 1e-3)')),
    "schedule-over-one-epoch": ("check2.training.loop", "train", "schedule = build_schedule(optimizer, total)",
                                "schedule = build_schedule(optimizer, per_epoch)"),
    "no-clipping": ("check2.training.step", "run_step", "out.gradient_norm = float(clip(optimizer))",
                    "out.gradient_norm = float(torch.nn.utils.clip_grad_norm_("
                    '[p for g in optimizer.param_groups for p in g["params"]], float("inf")))'),
    "clipping-each-group-alone": ("check2.training.step", "run_step", "out.gradient_norm = float(clip(optimizer))",
                                  'out.gradient_norm = float(max(torch.nn.utils.clip_grad_norm_(g["params"], 1.0) '
                                  "for g in optimizer.param_groups))"),
    "no-plan-check": ("check2.training.loop", "train", _PLAN_CHECK, ""),
    "resume-checks-only-the-step": ("check2.training.loop", "train",
                                    'if state.get("plan") != plan or state["step"] != last * per_epoch:',
                                    'if state["step"] != last * per_epoch:'),
    "budget-stays-halved": ("check2.training.loop", "train", _STEP_CALL_END,
                            _STEP_CALL_END
                            + "                    if losses[-1].out_of_memory:\n"
                            '                        budget = {**budget, "token_budget": '
                            'losses[-1].out_of_memory[-1]["token_budget"] / 2}\n'),
}


def _fails_under(monkeypatch, name: str, check) -> None:
    monkeypatch.setitem(faults.FAULTS, name, LOOP_FAULTS[name])
    faults.fails_under(name, check)


def _config(kind: str = "none", signal: str = "reader") -> Config:
    base = Config()
    alpha = 0.0 if kind == "adapter_only" else 1.0
    return replace(base, signal=signal, reader=replace(base.reader, read_position=0),
                   writer=replace(base.writer, alpha=alpha),
                   adapter=replace(base.adapter, kind=kind, implementation="hooks"))


@pytest.fixture(autouse=True)
def results(tmp_path, monkeypatch):
    """Records go to a folder of the test's own, and the chosen epochs are read from another."""
    monkeypatch.setattr(record, "RESULTS", str(tmp_path / "results"))
    monkeypatch.setattr(checkpoints, "CHOSEN_ROOT", tmp_path / "chosen")
    return tmp_path / "results"


def _options(storage, targets_file, **more) -> loop.Options:
    settings = {"episode_limit": EPISODES, "epochs": EPOCHS, "token_budget": float("inf"), **more}
    return loop.Options(storage=storage, targets_file=targets_file, **settings)


def _choose(name: str, epoch: int) -> None:
    folder = checkpoints.CHOSEN_ROOT / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / checkpoints.CHOSEN_FILE).write_text(json.dumps({"training_id": name, "epoch": epoch,
                                                              "record": "made-up.json"}))


def _files(folder) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(folder.iterdir())}


@pytest.fixture
def connector(bench, targets_file, tmp_path):
    """A connector training of two epochs, its first epoch chosen."""
    storage = checkpoints.Storage(tmp_path / "session", "checkpoints", FolderDrive(tmp_path / "drive"))
    config = _config("none")
    loop.train(config, _options(storage, targets_file), bench=bench)
    _choose(checkpoints.training_id(config), 1)
    return storage


@pytest.mark.parametrize("kind", ["none", "consumer", "adapter_only"])
def test_only_trained_parts_receive_gradient_and_the_model_is_unchanged_after_a_step(bench, targets_file, tmp_path,
                                                                                      connector, kind):
    config = _config(kind)
    loop_kind = loop.kind_of(config)
    pipeline, _ = loop.build_pipeline(config, loop_kind, bench, connector)
    try:
        before = {n: p.detach().clone() for n, p in bench.model.named_parameters()}
        episodes = inputs.episodes_for(config, inputs.EPISODES_TRAIN, targets_file)[:16]
        examples = inputs.examples_for(episodes, targets_file, bench.tokenizer)
        signals = {e["episode_id"]: row for e, row in zip(episodes, one_hot_situations(episodes))}
        form = loop.FORMS[loop.form_name(config)](pipeline)
        optimizer = loop.build_optimizer(loop.optimizer_groups(loop_kind, pipeline))
        schedule = loop.build_schedule(optimizer, 100)
        schedule.step()  # past the first step, whose learning rate is zero
        trained = [p for g in optimizer.param_groups for p in g["params"]]
        step.run_step(loop_kind, form, [examples[e["episode_id"]] for e in episodes], signals,
                      config.adapter.beta, float("inf"), optimizer, schedule)
        assert all(p.grad is None and not p.requires_grad for p in bench.model.parameters())
        assert all(p.grad is not None for p in trained)
        assert any(float(p.grad.abs().sum()) > 0 for p in trained)
        assert {id(p) for p in trained} == {id(p) for p in (
            (pipeline.adapter.parameters() if pipeline.adapter else [])
            + (list(pipeline.writers.parameters()) if pipeline.writers else []))}
        for n, p in bench.model.named_parameters():
            assert torch.equal(p, before[n]), n
    finally:
        pipeline.close()


class _Seen(Exception):
    """Raised by a spy once it has seen what it was put in place to see."""


def _stated_epochs_and_rates(bench, targets_file, connector, monkeypatch) -> None:
    """For each kind, from its config: the epochs `kind_of` gives, and the trained part and rate of each group of
    the optimiser `loop.train` builds, against the specification's numbers."""
    seen: dict = {}
    real_build_pipeline, real_build_optimizer = loop.build_pipeline, loop.build_optimizer

    def build_pipeline(*args, **kwargs):
        pipeline, source = real_build_pipeline(*args, **kwargs)
        parts = {"writers": list(pipeline.writers.parameters()) if pipeline.writers is not None else None,
                 "adapter": pipeline.adapter.parameters() if pipeline.adapter is not None else None}
        seen["parts"] = {part: {id(p) for p in parameters} for part, parameters in parts.items()
                         if parameters is not None}
        return pipeline, source

    def build_optimizer(groups):
        optimizer = real_build_optimizer(groups)
        seen["groups"] = [({id(p) for p in g["params"]}, g["lr"]) for g in optimizer.param_groups]
        raise _Seen

    monkeypatch.setattr(loop, "build_pipeline", build_pipeline)
    monkeypatch.setattr(loop, "build_optimizer", build_optimizer)
    for kind, stated in STATED.items():
        config = _config(kind)
        assert loop.kind_of(config).epochs == stated["epochs"], kind
        seen.clear()
        try:
            loop.train(config, _options(connector, targets_file), bench=bench)
        except _Seen:
            pass
        rates = {}
        for ids, rate in seen["groups"]:
            parts = [part for part, part_ids in seen["parts"].items() if part_ids == ids]
            assert len(parts) == 1, f"{kind}: a group of the optimiser is not one trained part"
            rates[parts[0]] = rate
        assert rates == stated["rates"], kind
        assert set(seen["parts"]) == set(stated["rates"]), kind


def test_each_training_has_the_stated_epochs_and_the_stated_rate_for_each_trained_part(bench, targets_file,
                                                                                       connector, monkeypatch):
    _stated_epochs_and_rates(bench, targets_file, connector, monkeypatch)


@pytest.mark.parametrize("fault", ["connector-rate-a-tenth", "consumer-writers-rate-ten-times",
                                   "connector-and-consumer-epochs-swapped", "rates-and-epochs-swapped"])
def test_a_rate_or_an_epoch_count_other_than_the_stated_one_turns_the_check_red(bench, targets_file, connector,
                                                                                monkeypatch, fault):
    _fails_under(monkeypatch, fault, lambda: _stated_epochs_and_rates(bench, targets_file, connector, monkeypatch))


def test_a_training_without_epochs_in_its_options_runs_its_kind_s_epochs(bench, targets_file, tmp_path, connector,
                                                                        results):
    per_epoch = steps_per_epoch(EPISODES, Config().training.episodes_per_step)
    folder = results / "measurements" / loop.RECORD_NAME
    for kind, stated in STATED.items():
        config = _config(kind)
        name = checkpoints.training_id(config)
        # Consumer training continues from the connector's chosen writers, on the connector's storage.
        storage = connector if kind == "consumer" else checkpoints.Storage(tmp_path / f"default-epochs-{kind}")
        before = set(folder.glob("*.json"))
        loop.train(config, _options(storage, targets_file, epochs=None), bench=bench)
        epochs = [json.loads(p.read_text())["values"] for p in sorted(set(folder.glob("*.json")) - before)]
        assert {v["training_id"] for v in epochs} == {name}, kind
        assert [v["epoch"] for v in epochs] == list(range(1, stated["epochs"] + 1)), kind
        assert sum(v["steps"] for v in epochs) == stated["epochs"] * per_epoch, kind


def _stated_rate(step_index: int, total: int, base: float) -> float:
    """Written from the sentence: optimiser step `s`, counted from zero, runs at `s / 50` of its rate while
    `s < 50`; the cosine decay then reaches zero at the end of the training of `total` steps."""
    if step_index < WARMUP:
        return base * step_index / WARMUP
    return base * 0.5 * (1 + math.cos(math.pi * (step_index - WARMUP) / (total - WARMUP)))


def _every_step_at_the_stated_rate(bench, targets_file, folder, monkeypatch) -> None:
    """A connector training of two epochs of 30 steps: the rate of every optimiser step, as the step is taken."""
    rates: list[list[float]] = []
    real_build_optimizer = loop.build_optimizer

    def build_optimizer(groups):
        optimizer = real_build_optimizer(groups)
        optimizer.register_step_pre_hook(lambda opt, args, kwargs: rates.append([g["lr"] for g in opt.param_groups]))
        return optimizer

    monkeypatch.setattr(loop, "build_optimizer", build_optimizer)
    config = _config("none")
    episodes = LONG_STEPS_PER_EPOCH * config.training.episodes_per_step
    loop.train(config, _options(checkpoints.Storage(folder), targets_file, episode_limit=episodes,
                                epochs=LONG_EPOCHS), bench=bench)
    total = LONG_STEPS_PER_EPOCH * LONG_EPOCHS
    assert total > WARMUP, "the training must leave the warm-up for the check to bite"
    assert len(rates) == total
    base = STATED["none"]["rates"]["writers"]
    for index, rate in enumerate(rates):
        assert rate == [pytest.approx(_stated_rate(index, total, base), rel=1e-12, abs=1e-18)], index


def test_every_optimiser_step_runs_at_the_rate_of_the_schedule_over_the_whole_training(bench, targets_file, tmp_path,
                                                                                       monkeypatch):
    _every_step_at_the_stated_rate(bench, targets_file, tmp_path / "long", monkeypatch)


def test_a_schedule_built_over_one_epoch_turns_the_rate_check_red(bench, targets_file, tmp_path, monkeypatch):
    _fails_under(monkeypatch, "schedule-over-one-epoch",
                 lambda: _every_step_at_the_stated_rate(bench, targets_file, tmp_path / "long", monkeypatch))


def _one_consumer_step_clipped_together(bench, targets_file, connector) -> None:
    """One consumer step whose raw gradient norm is above 1.0: the gradients it leaves on all trained parameters
    together have norm 1.0, and the norm it reports is the joint norm before clipping."""
    config = _config("consumer")
    kind = loop.kind_of(config)
    pipeline, _ = loop.build_pipeline(config, kind, bench, connector)
    try:
        episodes = inputs.episodes_for(config, inputs.EPISODES_TRAIN, targets_file)[:config.training.episodes_per_step]
        examples = inputs.examples_for(episodes, targets_file, bench.tokenizer)
        signals = {e["episode_id"]: row for e, row in zip(episodes, one_hot_situations(episodes))}
        form = loop.FORMS[loop.form_name(config)](pipeline)
        optimizer = loop.build_optimizer(loop.optimizer_groups(kind, pipeline))
        schedule = loop.build_schedule(optimizer, 100)
        trained = [p for g in optimizer.param_groups for p in g["params"]]
        # Each parameter's gradient as last accumulated, before anything clips it in place.
        raw: dict[int, torch.Tensor] = {}
        hooks = [p.register_post_accumulate_grad_hook(lambda p: raw.__setitem__(id(p), p.grad.detach().clone()))
                 for p in trained]
        try:
            out = step.run_step(kind, form, [examples[e["episode_id"]] for e in episodes], signals,
                                config.adapter.beta, float("inf"), optimizer, schedule)
        finally:
            for hook in hooks:
                hook.remove()
        assert len(raw) == len(trained), "every trained parameter must have received a gradient"
        raw_norm = math.sqrt(sum(float(raw[id(p)].double().pow(2).sum()) for p in trained))
        clipped_norm = math.sqrt(sum(float(p.grad.double().pow(2).sum()) for p in trained))
        assert raw_norm > CLIP_NORM, f"the raw norm {raw_norm} must be above the limit for the check to bite"
        assert out.gradient_norm == pytest.approx(raw_norm, rel=1e-5)
        assert clipped_norm == pytest.approx(CLIP_NORM, abs=1e-5)
    finally:
        pipeline.close()


def test_a_step_clips_all_trained_gradients_together_to_norm_one(bench, targets_file, connector):
    _one_consumer_step_clipped_together(bench, targets_file, connector)


@pytest.mark.parametrize("fault", ["no-clipping", "clipping-each-group-alone"])
def test_a_step_that_does_not_clip_all_gradients_together_turns_the_clipping_check_red(bench, targets_file, connector,
                                                                                     monkeypatch, fault):
    _fails_under(monkeypatch, fault, lambda: _one_consumer_step_clipped_together(bench, targets_file, connector))


def test_a_training_stopped_after_its_first_epoch_and_resumed_gives_the_same_checkpoint(bench, targets_file, tmp_path,
                                                                                       connector, monkeypatch):
    config = _config("consumer")
    name = checkpoints.training_id(config)
    through = checkpoints.Storage(tmp_path / "through", "checkpoints", FolderDrive(tmp_path / "drive-through"))
    # The connector's checkpoints are on the other storage; this one reads its chosen writers from there.
    monkeypatch.setattr(checkpoints, "load", _loader(connector, checkpoints.load))
    loop.train(config, _options(through, targets_file), bench=bench)

    drive = FolderDrive(tmp_path / "drive-stopped")
    first_session = checkpoints.Storage(tmp_path / "first-session", "checkpoints", drive)
    real_save = checkpoints.save

    class Stopped(Exception):
        pass

    def save_then_stop(training_id, epoch, state, storage=None):
        out = real_save(training_id, epoch, state, storage)
        if epoch == 1:
            raise Stopped
        return out

    monkeypatch.setattr(checkpoints, "save", save_then_stop)
    with pytest.raises(Stopped):
        loop.train(config, _options(first_session, targets_file), bench=bench)
    monkeypatch.setattr(checkpoints, "save", real_save)
    assert checkpoints.complete_epochs(name, first_session) == [1]

    # The session is lost: a new one starts with an empty disk and the same Drive folder.
    second_session = checkpoints.Storage(tmp_path / "second-session", "checkpoints", drive)
    loop.train(config, _options(second_session, targets_file), bench=bench)
    assert ("get", f"checkpoints/{name}/epoch1", ()) in drive.calls
    assert _files(second_session.folder(name, EPOCHS)) == _files(through.folder(name, EPOCHS))


def _refused_for_another_plan(start) -> bool:
    try:
        start()
    except checkpoints.CheckpointError as err:
        return "another plan" in str(err)
    return False


def _second_starts_with_another_plan_are_refused(bench, targets_file, connector) -> None:
    """After one epoch of consumer training, a second start with another episode limit, and one with another
    chosen connector epoch, on the same storage, are refused; a second start with the same plan is not."""
    config = _config("consumer")
    options = _options(connector, targets_file, epochs=1)
    loop.train(config, options, bench=bench)
    assert not _refused_for_another_plan(lambda: loop.train(config, options, bench=bench))
    # As many steps an epoch as the first start, so that only the plan tells the two apart.
    other_limit = EPISODES + 8
    per_step = config.training.episodes_per_step
    assert steps_per_epoch(other_limit, per_step) == steps_per_epoch(EPISODES, per_step)
    assert _refused_for_another_plan(
        lambda: loop.train(config, replace(options, episode_limit=other_limit), bench=bench))
    _choose(checkpoints.connector_id(config), 2)
    assert _refused_for_another_plan(lambda: loop.train(config, options, bench=bench))


def test_a_second_start_with_another_plan_is_refused(bench, targets_file, connector):
    _second_starts_with_another_plan_are_refused(bench, targets_file, connector)


@pytest.mark.parametrize("fault", ["no-plan-check", "resume-checks-only-the-step"])
def test_a_resume_that_ignores_the_plan_turns_the_plan_check_red(bench, targets_file, connector, monkeypatch, fault):
    _fails_under(monkeypatch, fault, lambda: _second_starts_with_another_plan_are_refused(bench, targets_file,
                                                                                         connector))


def _loader(connector_storage, real_load):
    def load(training_id, epoch, storage=None):
        source = connector_storage if training_id.startswith("connector-") else storage
        return real_load(training_id, epoch, source)
    return load


def test_consumer_training_starts_from_the_connector_s_chosen_writers(bench, connector):
    config = _config("consumer")
    connector_name = checkpoints.connector_id(config)
    chosen = checkpoints.load(connector_name, 1, connector)["writers"]
    last = checkpoints.load(connector_name, 2, connector)["writers"]
    assert any(not torch.equal(chosen[k], last[k]) for k in chosen), "the two epochs must differ for the check to bite"
    pipeline, source = loop.build_pipeline(config, loop.kind_of(config), bench, connector)
    try:
        state = pipeline.writers.state_dict()
        assert source == {"training_id": connector_name, "epoch": 1}
        assert all(torch.equal(state[k], chosen[k]) for k in chosen)
    finally:
        pipeline.close()


def test_the_perfect_signal_training_differs_from_the_main_one_only_in_p(bench, targets_file, tmp_path, monkeypatch):
    seen: dict[str, list] = {"reader": [], "perfect": []}
    starts: dict[str, dict] = {}
    real_step, real_build = step.connector_step, loop.build_pipeline

    for signal in ("reader", "perfect"):
        def fake_step(form, batch, p, size, signal=signal):
            seen[signal].append(([e.episode_id for e in batch], p.detach().cpu().clone()))
            return real_step(form, batch, p, size)

        def build(*args, signal=signal, **kwargs):
            pipeline, source = real_build(*args, **kwargs)
            starts[signal] = {k: v.detach().clone() for k, v in pipeline.writers.state_dict().items()}
            return pipeline, source

        monkeypatch.setattr(step, "connector_step", fake_step)
        monkeypatch.setattr(loop, "build_pipeline", build)
        storage = checkpoints.Storage(tmp_path / signal)
        loop.train(_config("none", signal), _options(storage, targets_file, epochs=1), bench=bench)

    assert [ids for ids, _ in seen["reader"]] == [ids for ids, _ in seen["perfect"]]
    assert all(torch.equal(starts["reader"][k], starts["perfect"][k]) for k in starts["reader"])
    episodes = {e["episode_id"]: e for e in inputs.episodes_for(_config(), inputs.EPISODES_TRAIN, targets_file)}
    saved = {r["episode_id"]: r["p"] for r in inputs.read_jsonl(inputs.P_TRAIN)}
    for (ids, p_reader), (_, p_perfect) in zip(seen["reader"], seen["perfect"]):
        assert torch.equal(p_perfect, one_hot_situations([episodes[i] for i in ids]))
        assert torch.equal(p_reader, torch.tensor([saved[i] for i in ids], dtype=torch.float32))
        assert not torch.equal(p_reader, p_perfect)


def test_a_non_finite_loss_stops_the_run_and_keeps_the_checkpoint(bench, targets_file, tmp_path, monkeypatch):
    config = _config("none")
    name = checkpoints.training_id(config)
    storage = checkpoints.Storage(tmp_path / "session", "checkpoints", FolderDrive(tmp_path / "drive"))
    per_epoch = steps_per_epoch(EPISODES, config.training.episodes_per_step)
    calls = {"n": 0}
    real_step = step.connector_step

    def fake_step(form, batch, p, size):
        calls["n"] += 1
        out = real_step(form, batch, p, size)
        if calls["n"] > per_epoch:  # one forward batch a step, at an unbounded budget
            nan = torch.full_like(out.total, math.nan)
            return EpisodeLosses(total=nan, cross_entropy=nan)
        return out

    monkeypatch.setattr(step, "connector_step", fake_step)
    with pytest.raises(step.NonFiniteLoss, match="last checkpoint kept: epoch 1"):
        loop.train(config, _options(storage, targets_file), bench=bench)
    assert checkpoints.complete_epochs(name, storage) == [1]
    assert not storage.folder(name, 2).exists()
    kept = checkpoints.load(name, 1, storage)
    assert all(torch.isfinite(v).all() for v in kept["writers"].values())


def test_an_episode_with_no_saved_p_is_an_error(bench, targets_file, tmp_path):
    p_file = tmp_path / "p.jsonl"
    rows = inputs.read_jsonl(inputs.P_TRAIN)
    episodes = inputs.episodes_for(_config(), inputs.EPISODES_TRAIN, targets_file)
    dropped = episodes[0]["episode_id"]
    p_file.write_text("".join(json.dumps(r) + "\n" for r in rows if r["episode_id"] != dropped))
    with pytest.raises(inputs.InputError, match="no saved p"):
        inputs.signals_for(_config(), episodes, p_file)


# The reviewers' input: two rows of the saved signal for one episode, with different `p`.
REPEATED_EPISODE = "abstract_algebra/11:named:short:A"
REPEATED_ROWS = (f'{{"episode_id": "{REPEATED_EPISODE}", "p": [0.9, 0.05, 0.05]}}\n'
                 f'{{"episode_id": "{REPEATED_EPISODE}", "p": [0.05, 0.05, 0.9]}}\n')


def test_a_saved_signal_with_two_rows_for_one_episode_is_an_error(tmp_path):
    p_file = tmp_path / "p.jsonl"
    p_file.write_text(REPEATED_ROWS, encoding="utf-8")
    with pytest.raises(inputs.InputError, match=f"more than one saved p.*{REPEATED_EPISODE}"):
        inputs.signals_for(_config(), [{"episode_id": REPEATED_EPISODE}], p_file)


def test_a_setting_no_training_has_is_refused():
    with pytest.raises(TrainingError):
        loop.kind_of(replace(_config("none"), writer=replace(Config().writer, alpha=0.0)))
    with pytest.raises(TrainingError):
        loop.kind_of(_config("consumer", "swapped"))
    with pytest.raises(TrainingError):
        loop.kind_of(replace(_config("none"), signal_forming="formed_model_only"))
    assert loop.kind_of(replace(_config("consumer"), signal_forming="formed_with_channel")).name == "consumer"
    assert loop.kind_of(_config("none")).name == "connector"
    assert loop.kind_of(_config("consumer", "perfect")).name == "consumer"
    assert loop.kind_of(_config("adapter_only")).name == "adapter_only"


def test_each_epoch_writes_its_record_and_keeps_a_copy_beside_the_checkpoints(bench, targets_file, tmp_path, results):
    config = _config("adapter_only")
    storage = checkpoints.Storage(tmp_path / "session", "checkpoints", FolderDrive(tmp_path / "drive"))
    loop.train(config, _options(storage, targets_file), bench=bench)
    name = checkpoints.training_id(config)
    records = sorted((results / "measurements" / loop.RECORD_NAME).glob("*.json"))
    assert len(records) == EPOCHS
    for epoch, path in enumerate(records, 1):
        values = json.loads(path.read_text())["values"]
        assert values["training_id"] == name and values["epoch"] == epoch
        assert set(values["mean_loss"]) == {"total", "cross_entropy", "kl"} and values["mean_loss"]["kl"] is None
        assert values["seconds_per_step"] > 0 and "peak_memory_gib" in values
        assert (storage.local_root / name / "records" / path.name).read_bytes() == path.read_bytes()
    assert sorted(storage.drive.listing(f"checkpoints/{name}/records")) == [p.name for p in records]


def _budget_of_rows(bench, targets_file, rows: int) -> int:
    """A token budget that holds `rows` of the longest training continuation in one forward batch."""
    episodes = inputs.episodes_for(_config(), inputs.EPISODES_TRAIN, targets_file)
    examples = inputs.examples_for(episodes, targets_file, bench.tokenizer).values()
    return rows * max(len(e.continuation_ids) for e in examples)


def _out_of_memory_after(monkeypatch, failing_calls) -> dict:
    """Make chosen forward batches of connector training run out of memory before their backward pass,
    as the KL term does, with the gradients of the step's earlier batches already accumulated."""
    calls = {"n": 0, "rows": []}
    real_step = step.connector_step

    def fake_step(form, batch, p, size):
        calls["n"] += 1
        calls["rows"].append(len(batch))
        if failing_calls(calls["n"]):
            raise torch.OutOfMemoryError("CUDA out of memory (made up)")
        return real_step(form, batch, p, size)

    monkeypatch.setattr(step, "connector_step", fake_step)
    return calls


def _budgets_planned(monkeypatch) -> list[float]:
    """The token budget of every planning of a step's forward batches, in order."""
    budgets: list[float] = []
    real_plan = step.plan_batches

    def plan_batches(examples, token_budget, *args, **kwargs):
        budgets.append(token_budget)
        return real_plan(examples, token_budget, *args, **kwargs)

    monkeypatch.setattr(step, "plan_batches", plan_batches)
    return budgets


def _by_step(rows: list[int], per_step: int) -> list[list[int]]:
    """The rows of consecutive forward batches, grouped into the steps they make up."""
    steps, current = [], []
    for count in rows:
        current.append(count)
        if sum(current) == per_step:
            steps.append(current)
            current = []
    assert not current, rows
    return steps


def _retried_step(bench, targets_file, folder, results, monkeypatch) -> None:
    config = _config("none")
    name = checkpoints.training_id(config)
    per_step = config.training.episodes_per_step
    budget = _budget_of_rows(bench, targets_file, 8)
    plain_calls = _out_of_memory_after(monkeypatch, lambda n: False)
    plain = checkpoints.Storage(folder / "plain")
    loop.train(config, _options(plain, targets_file, epochs=1, token_budget=budget), bench=bench)
    plain_steps = _by_step(plain_calls["rows"], per_step)
    assert len(plain_steps) == 2

    # The first step's second forward batch runs out of memory, after the first batch's backward pass.
    calls = _out_of_memory_after(monkeypatch, lambda n: n == 2)
    budgets = _budgets_planned(monkeypatch)
    retried = checkpoints.Storage(folder / "retried")
    loop.train(config, _options(retried, targets_file, epochs=1, token_budget=budget), bench=bench)
    first_attempt = calls["rows"][:2]
    assert first_attempt[0] < 16, "the step must have run a forward batch before the one that ran out"

    expected = checkpoints.load(name, 1, plain)["writers"]
    got = checkpoints.load(name, 1, retried)["writers"]
    for key, value in expected.items():
        assert float(value.norm()) > 0, key
        relative = float((got[key] - value).norm()) / float(value.norm())
        assert relative <= 1e-6, (key, relative)

    record = max((results / "measurements" / loop.RECORD_NAME).glob("*.json"))
    values = json.loads(record.read_text())["values"]
    assert values["out_of_memory_count"] == 1
    assert values["out_of_memory_events"] == [{"step": 0, "token_budget": budget, "rows": first_attempt[1]}]

    # The first step is planned at the budget, checked and run again at half of it; the next step is planned at the
    # full budget, and runs in the forward batches of a step that never ran out of memory.
    assert budgets == [budget, budget / 2, budget / 2, budget]
    retried_steps = _by_step(calls["rows"][2:], per_step)
    assert len(retried_steps) == 2
    assert retried_steps[1] == plain_steps[1]


def test_a_step_that_runs_out_of_memory_is_run_again_and_gives_the_same_weights(bench, targets_file, tmp_path,
                                                                                  results, monkeypatch):
    _retried_step(bench, targets_file, tmp_path, results, monkeypatch)


def test_a_budget_that_stays_halved_after_the_retried_step_turns_the_retry_check_red(bench, targets_file, tmp_path,
                                                                                     results, monkeypatch):
    _fails_under(monkeypatch, "budget-stays-halved",
                 lambda: _retried_step(bench, targets_file, tmp_path, results, monkeypatch))


def test_a_step_that_runs_out_of_memory_at_one_row_stops_the_run_and_keeps_the_checkpoint(bench, targets_file,
                                                                                          tmp_path, monkeypatch):
    config = _config("none")
    name = checkpoints.training_id(config)
    storage = checkpoints.Storage(tmp_path / "session")
    per_epoch = steps_per_epoch(EPISODES, config.training.episodes_per_step)
    calls = _out_of_memory_after(monkeypatch, lambda n: n > per_epoch)  # every batch after the first epoch
    budget = _budget_of_rows(bench, targets_file, 16)
    with pytest.raises(step.OutOfMemory, match="last checkpoint kept: epoch 1"):
        loop.train(config, _options(storage, targets_file, token_budget=budget), bench=bench)
    assert calls["n"] > per_epoch + 1, "the step must have been run again before the run stopped"
    assert checkpoints.complete_epochs(name, storage) == [1]
    assert not storage.folder(name, 2).exists()


def test_a_forward_batch_of_one_row_that_runs_out_of_memory_stops_at_once(bench, targets_file, monkeypatch):
    config = _config("none")
    kind = loop.kind_of(config)
    pipeline, _ = loop.build_pipeline(config, kind, bench, None)
    try:
        episodes = inputs.episodes_for(config, inputs.EPISODES_TRAIN, targets_file)[:1]
        examples = list(inputs.examples_for(episodes, targets_file, bench.tokenizer).values())
        signals = {e["episode_id"]: row for e, row in zip(episodes, one_hot_situations(episodes))}
        form = loop.FORMS[loop.form_name(config)](pipeline)
        optimizer = loop.build_optimizer(loop.optimizer_groups(kind, pipeline))
        schedule = loop.build_schedule(optimizer, 10)
        before = {k: v.detach().clone() for k, v in pipeline.writers.state_dict().items()}
        calls = _out_of_memory_after(monkeypatch, lambda n: True)
        with pytest.raises(step.OutOfMemory, match="one row"):
            step.run_step(kind, form, examples, signals, 1.0, _budget_of_rows(bench, targets_file, 4), optimizer,
                          schedule)
        assert calls["n"] == 1
        assert all(p.grad is None for p in pipeline.writers.parameters())
        assert all(torch.equal(v, before[k]) for k, v in pipeline.writers.state_dict().items())
    finally:
        pipeline.close()
