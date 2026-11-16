"""The training loop with the answer token's weight, on the tiny model: what reaches the step and the record."""

from __future__ import annotations

import json
from dataclasses import replace

from check2.training import checkpoints, inputs, loop, step
from tests.training.test_loop import _config, _loader, _options, connector, results  # noqa: F401 - fixtures

WEIGHT = 0.5


def _train(config, bench, targets_file, tmp_path, connector, monkeypatch, tag: str):  # noqa: F811
    seen = []
    real_step = step.consumer_step

    def spying_step(form, batch, p, beta, size, answer=None):
        seen.append((batch, answer))
        return real_step(form, batch, p, beta, size, answer)

    monkeypatch.setattr(step, "consumer_step", spying_step)
    monkeypatch.setattr(checkpoints, "load", _loader(connector, checkpoints.load))
    storage = checkpoints.Storage(tmp_path / tag)
    folder = loop.train(config, _options(storage, targets_file, epochs=1), bench=bench)
    return seen, folder


def _weighted(config, weight: float):
    return replace(config, training=replace(config.training, answer_token_weight=weight))


def _epoch_record(results) -> dict:  # noqa: F811
    files = sorted((results / "measurements" / loop.RECORD_NAME).glob("*.json"))
    return json.loads(files[-1].read_text(encoding="utf-8"))


def test_the_weighted_training_hands_each_step_its_weight_and_the_short_episodes_positions(
        bench, targets_file, tmp_path, connector, results, monkeypatch):  # noqa: F811
    config = _weighted(_config("consumer"), WEIGHT)
    seen, folder = _train(config, bench, targets_file, tmp_path, connector, monkeypatch, "weighted")
    assert folder.name == "consumer-reader-rank32-beta1-seed0-answer0.5"
    assert seen and all(answer is not None and answer.weight == WEIGHT for _, answer in seen)
    answer = seen[0][1]
    assert all(other is answer for _, other in seen)
    episodes = inputs.episodes_for(config, inputs.EPISODES_TRAIN, targets_file)
    drawn = {e.episode_id: e for batch, _ in seen for e in batch}
    by_id = {e["episode_id"]: e for e in episodes}
    short = {i for i in drawn if by_id[i]["format"] == "short"}
    assert short and short < set(drawn) and short <= set(answer.positions)
    assert all(by_id[i]["format"] == "short" for i in answer.positions)
    written = _epoch_record(results)
    assert written["config"]["training"]["answer_token_weight"] == WEIGHT
    assert written["values"]["plan"]["answer_token_weight"] == WEIGHT


def test_a_training_without_the_weight_hands_the_step_none_and_writes_no_key(
        bench, targets_file, tmp_path, connector, results, monkeypatch):  # noqa: F811
    config = _config("consumer")
    seen, folder = _train(config, bench, targets_file, tmp_path, connector, monkeypatch, "plain")
    assert folder.name == "consumer-reader-rank32-beta1-seed0"
    assert seen and all(answer is None for _, answer in seen)
    written = _epoch_record(results)
    assert "answer_token_weight" not in written["config"]["training"]
    assert "answer_token_weight" not in written["values"]["plan"]
