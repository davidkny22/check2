"""Training with the wide signal on the tiny model: the saved states, each conversation's S, and two steps."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from check2.config.schema import Config
from check2.data.episodes import TAIL_LENGTH
from check2.mechanism import wide_signal
from check2.reader import cache, wide_states
from check2.training import checkpoints, inputs, loop, step, wide_inputs

WIDE = "probabilities_and_state"
LAYERS = (2, wide_signal.TINY_STATE_LAYER, 6)
READ_POSITION = 0


def _config(kind: str = "none") -> Config:
    base = Config()
    return replace(base, reader=replace(base.reader, read_position=READ_POSITION),
                   writer=replace(base.writer, input=WIDE),
                   adapter=replace(base.adapter, kind=kind, implementation="hooks"))


class _CacheTokenizer:
    """The tiny model's folded tokenizer, taking the keyword the cache's window code passes."""

    def __init__(self, base) -> None:
        self.base = base
        self.pad_token_id = base.pad_token_id

    def __call__(self, texts, add_special_tokens: bool = True):
        if isinstance(texts, str):
            return {"input_ids": self.base.encode(texts, add_special_tokens=add_special_tokens)}
        return self.base(texts)

    def convert_tokens_to_ids(self, token):
        return self.base.convert_tokens_to_ids(token)


def _state_alone(bench, episode: dict) -> torch.Tensor:
    """The tiny state layer's output at the read token, from a pass of this conversation alone."""
    ids = bench.tokenizer([episode["prompt"]])["input_ids"][0]
    read_index = episode["n_prompt_tokens"] - TAIL_LENGTH + READ_POSITION
    with torch.no_grad():
        out = bench.model(input_ids=torch.tensor([ids[:read_index + 1]]), output_hidden_states=True)
    return out.hidden_states[wide_signal.TINY_STATE_LAYER + 1][0, -1]


@pytest.fixture(scope="module")
def training(bench, targets_file, tmp_path_factory):
    """Sixteen training episodes, a tiny-model cache of them, and their saved states file."""
    folder = tmp_path_factory.mktemp("wide")
    episodes = inputs.episodes_for(_config(), inputs.EPISODES_TRAIN, targets_file)[:16]
    cache.cache_states(bench.model, _CacheTokenizer(bench.tokenizer), [{**e, "split": "train"} for e in episodes],
                       folder / "cache",
                       LAYERS, batch_size=4)
    path = wide_states.write_states("train", [e["episode_id"] for e in episodes], READ_POSITION, folder / "cache",
                                    folder / "wide", layer=wide_signal.TINY_STATE_LAYER)
    return episodes, folder, path


def test_the_saved_state_is_the_read_tokens_state_from_a_pass_alone(bench, training):
    episodes, _, path = training
    saved = wide_states.load_states(path)
    assert sorted(saved) == sorted(e["episode_id"] for e in episodes)
    for e in episodes:
        assert torch.allclose(saved[e["episode_id"]].to(torch.float32), _state_alone(bench, e),
                              rtol=1e-4, atol=1e-5), e["episode_id"]


def test_a_cache_that_lacks_an_episode_is_refused_before_anything_is_written(training, tmp_path):
    episodes, folder, _ = training
    with pytest.raises(wide_states.StatesError):
        wide_states.write_states("train", [e["episode_id"] for e in episodes] + ["not-in-the-cache"], READ_POSITION,
                                 folder / "cache", tmp_path, layer=wide_signal.TINY_STATE_LAYER)
    assert not any(tmp_path.iterdir())


def test_each_conversation_gets_its_saved_p_beside_its_saved_state(training):
    episodes, _, path = training
    signals = wide_inputs.signals_for(_config(), episodes, inputs.P_TRAIN, path)
    saved_p = {r["episode_id"]: r["p"] for r in inputs.read_jsonl(inputs.P_TRAIN)}
    states = wide_states.load_states(path)
    for e in episodes:
        h = states[e["episode_id"]].to(torch.float32)
        n = (h - h.mean()) / torch.sqrt(((h - h.mean()) ** 2).mean() + 1e-5)
        expected = torch.cat([torch.tensor(saved_p[e["episode_id"]]), n])
        assert torch.allclose(signals[e["episode_id"]], expected, rtol=1e-5, atol=1e-6)


def test_a_training_conversation_with_no_saved_state_is_refused(bench, targets_file, training):
    episodes, folder, _ = training
    path = wide_states.write_states("train", [e["episode_id"] for e in episodes[1:]], READ_POSITION,
                                    folder / "cache", folder / "fewer", layer=wide_signal.TINY_STATE_LAYER)
    with pytest.raises(inputs.InputError, match="no saved state"):
        wide_inputs.signals_for(_config(), episodes, inputs.P_TRAIN, path)


def test_two_connector_steps_change_the_weights_that_read_the_state_and_leave_the_model_alone(bench, targets_file,
                                                                                               training, tmp_path):
    episodes, _, path = training
    config = _config("none")
    kind = loop.kind_of(config)
    storage = checkpoints.Storage(tmp_path / "session")
    pipeline, _ = loop.build_pipeline(config, kind, bench, storage)
    try:
        assert pipeline.writers.wide and pipeline.writers.input_size == wide_signal.size(bench.model.config.hidden_size)
        model_before = {n: p.detach().clone() for n, p in bench.model.named_parameters()}
        examples = inputs.examples_for(episodes, targets_file, bench.tokenizer)
        signals = wide_inputs.signals_for(config, episodes, inputs.P_TRAIN, path)
        form = loop.FORMS[loop.form_name(config)](pipeline)
        optimizer = loop.build_optimizer(loop.optimizer_groups(kind, pipeline))
        schedule = loop.build_schedule(optimizer, 100)
        schedule.step()  # past the first step, whose learning rate is zero
        batch = [examples[e["episode_id"]] for e in episodes]
        state_columns = {j: pipeline.writers.first[str(j)].weight for j in pipeline.writers.layers}
        before = {j: w.detach()[:, 3:].clone() for j, w in state_columns.items()}
        for _ in range(2):
            step.run_step(kind, form, batch, signals, config.adapter.beta, float("inf"), optimizer, schedule)
        # The second step's gradient reaches the first layer through the second, which the first step moved.
        assert all(float(w.grad[:, 3:].abs().sum()) > 0 for w in state_columns.values())
        # Weight decay alone would move a weight by at most two steps' rate times the decay times its size.
        group = optimizer.param_groups[0]
        for j, w in state_columns.items():
            decay_only = 2 * group["lr"] * group["weight_decay"] * float(before[j].abs().max())
            assert float((w.detach()[:, 3:] - before[j]).abs().max()) > 10 * decay_only, j
        for n, p in bench.model.named_parameters():
            assert torch.equal(p, model_before[n]), n
    finally:
        pipeline.close()
