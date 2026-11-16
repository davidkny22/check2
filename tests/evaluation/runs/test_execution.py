"""Executing a run on the tiny model: repeatable, resumable, complete or refused."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from check2.evaluation.runs import RunError, execution, output, run, run_id_for
from check2.evaluation.runs import episodes as run_episodes
from check2.evaluation.scoring import score_stated
from check2.mechanism.control import ControlState
from check2.mechanism.generate import Pipeline, Reply

from .conftest import config


def _run(bench, reader, cfg, run_id, folder, epoch=None):
    return run(cfg, run_id, bench.model, bench.tokenizer, bench.layout, reader, epoch, episode_folder=folder)


def _meta(folder):
    return json.loads((folder / "run.json").read_text(encoding="utf-8"))


def _replies(records):
    return [(r["episode_id"], r["reply"], r["a2"]) for r in records]


class LostSession(RuntimeError):
    pass


def test_two_executions_of_one_config_give_identical_outputs_and_order(bench, reader, results, episode_folder):
    first = _run(bench, reader, config(), "original-a", episode_folder)
    second = _run(bench, reader, config(), "original-b", episode_folder)
    a, b = output.read_records(first), output.read_records(second)
    assert a == b and len(a) == 20
    assert _meta(first)["order_sha256"] == _meta(second)["order_sha256"]
    assert _meta(first)["status"] == "complete" and _meta(first)["episodes_written"] == 20
    numbers = json.loads((first / "numbers.json").read_text(encoding="utf-8"))
    assert numbers["rates"] and numbers["every_reply_has_a2"]


def test_the_order_depends_on_the_episodes_alone(results, episode_folder):
    plain = run_episodes.order(run_episodes.select(config(), episode_folder))
    other = run_episodes.order(run_episodes.select(config(**{"adapter.kind": "consumer", "writer.alpha": 1.0}),
                                                   episode_folder))
    assert run_episodes.order_sha256(plain) == run_episodes.order_sha256(other)
    sizes = [len(b) for b in plain]
    assert sizes == [2] * 10
    for batch in plain:
        assert len({e["format"] for e in batch}) == 1
    flat = [e for b in plain for e in b]
    for fmt in ("snap", "short"):
        mine = [(e["n_prompt_tokens"], e["episode_id"]) for e in flat if e["format"] == fmt]
        assert mine == sorted(mine)
    assert [e["format"] for e in flat] == ["snap"] * 10 + ["short"] * 10


def test_a_run_stopped_after_three_batches_and_resumed_equals_one_run_through(bench, reader, results,
                                                                               episode_folder, monkeypatch):
    whole = _run(bench, reader, config(), "through", episode_folder)
    calls = []
    real = execution.run_batch

    def stops(*args, **kwargs):
        if len(calls) == 3:
            raise LostSession("the session was lost")
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(execution, "run_batch", stops)
    with pytest.raises(LostSession):
        _run(bench, reader, config(), "resumed", episode_folder)
    stopped = results / "resumed"
    assert _meta(stopped)["status"] == "failed" and not (stopped / "numbers.json").exists()
    assert len(output.read_records(stopped)) == 6
    # Half a batch on disk after the last complete one, as a lost session can leave it.
    with (stopped / output.EPISODES_FILE).open("ab") as f:
        f.write(b"\x1f\x8b\x08\x00partial")
    monkeypatch.setattr(execution, "run_batch", real)
    resumed_calls = []

    def counts(*args, **kwargs):
        resumed_calls.append(1)
        return real(*args, **kwargs)
    monkeypatch.setattr(execution, "run_batch", counts)
    _run(bench, reader, config(), "resumed", episode_folder)
    assert len(resumed_calls) == 7
    assert output.read_records(stopped) == output.read_records(whole)
    assert _meta(stopped)["status"] == "complete"
    assert _meta(stopped)["order_sha256"] == _meta(whole)["order_sha256"]


def test_a_run_with_a_missing_batch_is_not_complete_and_has_no_numbers(bench, reader, results, episode_folder,
                                                                         monkeypatch):
    real = output.RunOutput.append
    seen = []

    def drops_second(self, rows, fmt, seconds):
        seen.append(1)
        if len(seen) == 2:
            return None
        return real(self, rows, fmt, seconds)

    monkeypatch.setattr(output.RunOutput, "append", drops_second)
    with pytest.raises(RunError, match="not complete"):
        _run(bench, reader, config(), "gap", episode_folder)
    folder = results / "gap"
    meta = _meta(folder)
    assert meta["status"] != "complete" and meta["episodes_written"] == 18 and meta["episodes_expected"] == 20
    assert not (folder / "numbers.json").exists()


def test_a_completed_run_is_not_run_again(bench, reader, results, episode_folder):
    _run(bench, reader, config(), "once", episode_folder)
    with pytest.raises(RunError, match="already complete"):
        _run(bench, reader, config(), "once", episode_folder)


def test_a_run_is_resumed_only_with_its_own_settings(bench, reader, results, episode_folder, monkeypatch):
    def lost(*args, **kwargs):
        raise LostSession("lost")
    monkeypatch.setattr(execution, "run_batch", lost)
    with pytest.raises(LostSession):
        _run(bench, reader, config(), "settings", episode_folder)
    with pytest.raises(RunError, match="other settings"):
        _run(bench, reader, config(**{"data.formats": ("snap",)}), "settings", episode_folder)


SNAP_TEXTS = ["B", "(C).", "I think so", "d", "A"]
SHORT_TEXTS = ["Because. Final answer: (D)", "No final line here", "Final answer: **B**", "final answer: (A)"]


def test_the_appended_read_gives_a2_exactly_where_no_answer_is_stated(bench, reader, results, episode_folder,
                                                                      monkeypatch):
    def scripted(self, prefilled, p, max_new_tokens, alpha=1.0, disabled_layers=(), sampling=None,
                 keep_scores=False):
        texts = SNAP_TEXTS if prefilled.episodes[0]["format"] == "snap" else SHORT_TEXTS
        out = []
        for e in prefilled.episodes:
            text = texts[sum(map(ord, e["episode_id"])) % len(texts)]
            ids = self.tokenizer.encode(text)
            out.append(Reply(text=text, token_ids=ids, ended=False))
        return out

    monkeypatch.setattr(Pipeline, "generate", scripted)
    folder = _run(bench, reader, config(), "scripted", episode_folder)
    records = output.read_records(folder)
    kinds = set()
    for r in records:
        stated = score_stated(r["reply"], r["format"])
        kinds.add(stated is None)
        if stated is None:
            assert r["scored_by"] == "appended" and r["a2"] == r["appended_read"]
        else:
            assert r["scored_by"] == "stated" and r["a2"] == stated
    assert kinds == {True, False}
    # The appended letter is the one the pipeline's own appended read gives on the same batch.
    batch = [e for e in run_episodes.select(config(), episode_folder) if e["episode_id"] == records[0]["episode_id"]]
    with Pipeline(bench.model, bench.tokenizer, ControlState(),
                  reader=reader) as pipeline:
        prefilled = pipeline.prefill(batch)
        reply = Reply(text=records[0]["reply"], token_ids=[], ended=False)
        assert pipeline.appended_read(prefilled, None, [reply]) == [records[0]["appended_read"]]


def test_with_alpha_zero_the_replies_equal_a_run_with_no_writers(bench, reader, results, episode_folder,
                                                                 checkpoints):
    off = _run(bench, reader, config(**{"adapter.kind": "consumer", "writer.alpha": 0.0}), "off", episode_folder)
    alone = _run(bench, reader, config(**{"adapter.kind": "adapter_only", "writer.alpha": 0.0}), "alone",
                 episode_folder)
    on = _run(bench, reader, config(**{"adapter.kind": "consumer", "writer.alpha": 1.0}), "on", episode_folder)
    assert _replies(output.read_records(off)) == _replies(output.read_records(alone))
    assert _replies(output.read_records(on)) != _replies(output.read_records(alone))
    assert all(r["p_used"] is None for r in output.read_records(alone))
    assert all(r["p_used"] == r["p"] for r in output.read_records(on))


def test_a_config_naming_a_missing_checkpoint_is_refused(bench, reader, results, episode_folder, checkpoints):
    checkpoints.answer = (None, None)
    for changes in ({"adapter.kind": "consumer", "writer.alpha": 1.0}, {"writer.alpha": 1.0},
                    {"adapter.kind": "adapter_only"}):
        with pytest.raises(RunError, match="checkpoint does not exist"):
            _run(bench, reader, config(**changes), "missing", episode_folder)
    assert not (results / "missing").exists()


def test_the_original_model_loads_no_trained_part(bench, reader, results, episode_folder, checkpoints):
    _run(bench, reader, config(), "original", episode_folder)
    assert checkpoints.calls == []


def test_an_epoch_is_named_only_for_heldout(bench, reader, results, episode_folder, checkpoints):
    trained = config(**{"adapter.kind": "consumer", "writer.alpha": 1.0})
    with pytest.raises(RunError, match="only for a run over heldout"):
        _run(bench, reader, config(**{"adapter.kind": "consumer", "writer.alpha": 1.0, "data.eval_split": "test"}),
             "epoch-test", episode_folder, epoch=1)
    with pytest.raises(RunError, match="no epoch"):
        _run(bench, reader, config(), "epoch-original", episode_folder, epoch=1)
    _run(bench, reader, trained, "epoch-heldout", episode_folder, epoch=2)
    assert checkpoints.calls[-1][1] == 2


def test_the_run_id_is_the_config_name_with_split_and_epoch(tmp_path):
    path = tmp_path / "check2.yaml"
    assert run_id_for(path, config(**{"data.eval_split": "test"}), None) == "check2"
    assert run_id_for(path, config(), None) == "check2-heldout"
    assert run_id_for(path, config(), 3) == "check2-heldout-epoch3"


def test_execute_loads_the_config_file_the_model_and_the_saved_reader(bench, reader, tmp_path, monkeypatch):
    import torch

    from check2.mechanism import loading
    from check2.reader import files as reader_files
    from check2.reader.crossfit import save_reader
    from check2.reader.model import Reader

    configs = Path(__file__).resolve().parents[3] / "experiments" / "configs"
    loads = []
    monkeypatch.setattr(loading, "load_full", lambda dtype: loads.append(dtype) or (bench.model, bench.tokenizer,
                                                                                   bench.layout))
    saved = tmp_path / "reader.pt"
    monkeypatch.setattr(reader_files, "READER_WEIGHTS", saved)
    save_reader(reader, saved)
    with pytest.raises(RunError, match="reader reads layers"):
        execution.execute(configs / "heldout" / "original.yaml")
    assert loads == [torch.bfloat16]
    save_reader(Reader([14, 15, 18, 19, 22, 23], bench.model.config.hidden_size), saved)
    calls = []
    monkeypatch.setattr(execution, "run", lambda *args: calls.append(args) or tmp_path)
    assert execution.execute(configs / "heldout" / "original.yaml") == tmp_path
    config, run_id = calls[0][0], calls[0][1]
    assert run_id == "original-heldout" and config.data.eval_split == "heldout"
    assert isinstance(config.reader.read_position, int) and config.model_revision != "pinned"
    execution.execute(configs / "heldout" / "check2.yaml", checkpoint_epoch=2)
    assert calls[-1][1] == "check2-heldout-epoch2" and calls[-1][-1] == 2
    with pytest.raises(RunError, match="only for a run over heldout"):
        execution.execute(configs / "test" / "check2.yaml", checkpoint_epoch=2)
