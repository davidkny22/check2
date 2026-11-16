"""The forming update attached to the library's own generate, as the formed model's target sampling runs it."""

from __future__ import annotations

import gzip
import json

import torch

from check2.data import sampling
from check2.data.episodes import TAIL_LENGTH
from check2.training.forming.attach import follow_prompt_tail
from check2.training.forming.targets import sample_formed
from tests.training.forming.helpers import additions_seen, random_update


def _hooks(bench) -> int:
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in bench.model.modules())


def test_in_the_library_s_generate_the_update_adds_only_at_real_prompt_tokens_up_to_the_read_token(bench, episodes):
    batch = episodes["short"]
    prompts = [bench.tokenizer(e["prompt"])["input_ids"] for e in batch]
    ids, attention = sampling.left_padded(prompts, 7)
    update = random_update(bench)
    update.install()
    hooks = follow_prompt_tail(update, bench.model, 0)
    try:
        with additions_seen(update) as seen, torch.inference_mode():
            bench.model.generate(input_ids=ids, attention_mask=attention, do_sample=False, max_new_tokens=3,
                                 pad_token_id=7, eos_token_id=2)
    finally:
        hooks.remove()
        update.remove()
    width = ids.shape[1]
    region = (torch.arange(width).unsqueeze(0) <= width - TAIL_LENGTH) & attention.bool()
    for e, row in zip(batch, region):
        assert int(row.sum()) == e["read_index"] + 1
    for name, calls in seen.items():
        prompt_call, *decoding = calls
        assert float(prompt_call[~region].abs().max()) == 0.0, name
        assert bool((prompt_call[region].norm(dim=-1) > 0).all()), name
        assert len(decoding) == 2 and all(float(c.abs().max()) == 0.0 for c in decoding), name
    assert update.control.mask is None


def test_formed_sampling_runs_the_sampler_with_the_update_and_leaves_no_hook(bench, episodes, tmp_path):
    update = random_update(bench)
    update.install()
    hooks = _hooks(bench)
    batches = sampling.plan_batches(episodes["snap"] + episodes["short"], 4)
    out = tmp_path / "samples.jsonl.gz"
    try:
        per_format = sample_formed(bench.model, bench.tokenizer, update, 0, batches, 2, 0, out)
    finally:
        update.remove()
    assert _hooks(bench) == hooks - len(update.modules)
    with gzip.open(out, "rt", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 8 and all(len(x["samples"]) == 2 for x in lines)
    assert set(per_format) == {"snap", "short"}
