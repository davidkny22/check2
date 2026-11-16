"""The Swap run's donors, and the Steering baseline's train pass, direction command and hook in a run."""

from __future__ import annotations

import gzip
import json
import random

import pytest
import torch

from check2.evaluation import steering, swap
from check2.evaluation.runs import execution, output, run
from check2.evaluation.runs import episodes as run_episodes
from check2.mechanism.model_parts import decoder_layers
from tests.evaluation.faults import fails_under

from .conftest import config


def _episode(i, form, fmt, situation):
    return {"episode_id": f"q{i:03d}:{form}:{fmt}", "qid": f"q{i:03d}", "form": form, "format": fmt,
            "situation": situation, "p": [i / 1000, 0.5, 0.5 - i / 1000]}


def test_swap_donors_share_form_and_format_and_come_from_the_other_situation():
    situations = ("user_right", "hold", "recheck")
    records = [_episode(i, form, fmt, situations[i % 3]) for i, (form, fmt) in
               enumerate((f, m) for f in ("bare", "named", "reasoned") for m in ("snap", "short") for _ in range(9))]
    records = [r for r in records if not (r["form"] == "bare" and r["situation"] == "recheck")]
    rng_order = list(records)
    random.Random(1).shuffle(rng_order)
    donors = swap.draw_donors(rng_order, records)
    by_id = {r["episode_id"]: r for r in records}
    assert set(donors) == {r["episode_id"] for r in records if r["situation"] != "recheck"}
    for eid, (donor, p) in donors.items():
        mine, theirs = by_id[eid], by_id[donor]
        assert (theirs["form"], theirs["format"]) == (mine["form"], mine["format"])
        assert theirs["situation"] == swap.OTHER[mine["situation"]]
        assert p == theirs["p"]
    # One generator, one choice per episode in the run's order, from that episode's sorted donors.
    rng = random.Random(2030)
    for e in rng_order:
        if e["situation"] == "recheck":
            continue
        pool = sorted(r["episode_id"] for r in records if (r["form"], r["format"]) == (e["form"], e["format"])
                      and r["situation"] == swap.OTHER[e["situation"]])
        assert donors[e["episode_id"]][0] == rng.choice(pool)
    p_reader = torch.tensor([r["p"] for r in rng_order[:6]])
    swapped, ids = swap.swapped(p_reader, rng_order[:6], donors)
    for b, e in enumerate(rng_order[:6]):
        if e["situation"] == "recheck":
            assert ids[b] is None and torch.equal(swapped[b], p_reader[b])
        else:
            assert ids[b] == donors[e["episode_id"]][0]


def test_a_swap_run_carries_donor_p_and_keeps_recheck_p(bench, reader, results, episode_folder, checkpoints):
    def go(cfg, name):
        return run(cfg, name, bench.model, bench.tokenizer, bench.layout, reader, episode_folder=episode_folder)

    with pytest.raises(swap.SwapError):
        go(config(**{"adapter.kind": "consumer", "writer.alpha": 1.0, "signal": "swapped"}), "swap-early")
    check2 = output.read_records(go(config(**{"adapter.kind": "consumer", "writer.alpha": 1.0}), "check2-heldout"))
    swapped = output.read_records(go(config(**{"adapter.kind": "consumer", "writer.alpha": 1.0, "signal": "swapped"}),
                                     "swap-heldout"))
    p_of = {r["episode_id"]: r["p"] for r in check2}
    by_id = {r["episode_id"]: r for r in check2}
    assert [r["episode_id"] for r in swapped] == [r["episode_id"] for r in check2]
    for r in swapped:
        assert r["p"] == p_of[r["episode_id"]]
        if r["situation"] == "recheck":
            assert r["donor"] is None and r["p_used"] == r["p"]
        else:
            donor = by_id[r["donor"]]
            assert (donor["form"], donor["format"]) == (r["form"], r["format"])
            assert donor["situation"] == swap.OTHER[r["situation"]]
            assert r["p_used"] == pytest.approx(p_of[r["donor"]], rel=0, abs=0)


TINY_STEERING_LAYER = 4
TRAIN_PASS = {"data.eval_split": "train", "data.forms": ("named", "reasoned")}


def test_the_train_pass_saves_read_states_and_the_command_writes_each_format_s_direction(
        bench, reader, results, episode_folder, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(steering, "STEERING_LAYER", TINY_STEERING_LAYER)
    monkeypatch.setattr(steering, "STATES_ROOT", tmp_path / "states")
    monkeypatch.setattr(steering, "DIRECTION_FILE", tmp_path / "steering" / "direction.json")
    (episode_folder / "train.jsonl").write_text((episode_folder / "heldout.jsonl").read_text(encoding="utf-8"),
                                                encoding="utf-8")
    train = run(config(**TRAIN_PASS), "steering-pass", bench.model, bench.tokenizer, bench.layout, reader,
                episode_folder=episode_folder)
    saved = steering.load_states("steering-pass")
    records = output.read_records(train)
    assert set(saved) == {r["episode_id"] for r in records}
    # The saved state is the layer's output at the read token, computed here on one episode alone.
    first = run_episodes.select(config(**TRAIN_PASS), episode_folder)[0]
    from check2.gates.mechanism_setup import layer_outputs, plain_pipeline

    with plain_pipeline(bench) as pipeline, layer_outputs(bench.model, [TINY_STEERING_LAYER]) as seen:
        pipeline.prefill([first])
    assert torch.allclose(saved[first["episode_id"]], seen[TINY_STEERING_LAYER][0][0, -1].float(), atol=1e-5)
    # A made-up outcome for the tiny model's run: in each format, every other episode with a wrong target adopts.
    groups = {fmt: {"not_adopted": [], "adopted": []} for fmt in ("snap", "short")}
    patched = []
    for r in records:
        if r["target"] != r["gold"]:
            group = groups[r["format"]]
            adopts = len(group["adopted"]) <= len(group["not_adopted"])
            group["adopted" if adopts else "not_adopted"].append(r["episode_id"])
            r = {**r, "a2": r["target"] if adopts else r["gold"]}
        patched.append(r)
    assert all(ids for group in groups.values() for ids in group.values())
    with gzip.open(train / output.EPISODES_FILE, "wt", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in patched)
    assert steering.main(["--run", str(train)]) == 0
    printed = capsys.readouterr().out
    written = json.loads(steering.DIRECTION_FILE.read_text(encoding="utf-8"))
    for fmt, group in groups.items():
        counts = {name: len(ids) for name, ids in group.items()}
        assert written["counts"][fmt] == counts
        held = torch.stack([saved[e].double() for e in group["not_adopted"]]).mean(0)
        adopted = torch.stack([saved[e].double() for e in group["adopted"]]).mean(0)
        expected = (held - adopted).float()
        torch.testing.assert_close(torch.tensor(written["vectors"][fmt]), expected, rtol=0, atol=1e-6)
        assert written["norms"][fmt] == pytest.approx(float(expected.norm()), abs=1e-5)
        assert (f"{fmt} norm {written['norms'][fmt]:.4f}, {counts['not_adopted']} not adopted against "
                f"{counts['adopted']} adopted") in printed


def test_a_steering_run_adds_each_batch_s_own_format_vector_after_the_read_and_nothing_at_strength_zero(
        bench, reader, results, episode_folder, tmp_path, monkeypatch):
    monkeypatch.setattr(steering, "STEERING_LAYER", TINY_STEERING_LAYER)
    monkeypatch.setattr(steering, "DIRECTION_FILE", tmp_path / "steering" / "direction.json")
    hidden = bench.model.config.hidden_size
    vectors = {"snap": torch.linspace(-1.0, 1.0, hidden), "short": torch.linspace(2.0, -0.5, hidden)}
    steering.DIRECTION_FILE.parent.mkdir(parents=True)
    steering.DIRECTION_FILE.write_text(json.dumps({"vectors": {f: v.tolist() for f, v in vectors.items()}}),
                                       encoding="utf-8")

    def go(cfg, name):
        return run(cfg, name, bench.model, bench.tokenizer, bench.layout, reader, episode_folder=episode_folder)

    # Strength 0 is the original model, token for token.
    original = output.read_records(go(config(), "original-heldout"))
    assert output.read_records(go(config(**{"baseline.steering_strength": 0}), "steering-g0-heldout")) == original
    # The layer's output before the steering hook (a hook registered first) and after it (a hook registered
    # inside each batch, after the run has put steering on for it), with the batch's format and the mask.
    layer = decoder_layers(bench.model)[TINY_STEERING_LAYER]
    before, after = [], []
    real_run_batch = execution.run_batch

    def observed_batch(pipeline, cfg, batch, *args, **kwargs):
        fmt = batch[0]["format"]
        assert all(e["format"] == fmt for e in batch)

        def keep(module, inputs, out):
            after.append((fmt, batch, pipeline.control.mask.detach().clone(), out.detach().clone()))
        handle = layer.register_forward_hook(keep)
        try:
            return real_run_batch(pipeline, cfg, batch, *args, **kwargs)
        finally:
            handle.remove()

    monkeypatch.setattr(execution, "run_batch", observed_batch)
    first = layer.register_forward_hook(lambda module, inputs, out: before.append(out.detach().clone()))
    strength = 2
    try:
        folder = go(config(**{"baseline.steering_strength": strength}), "steering-g2-heldout")
    finally:
        first.remove()
    assert len(before) == len(after) > 0
    steered_formats, prompt_passes = set(), {}
    for plain, (fmt, batch, mask, out) in zip(before, after):
        added = out - plain
        assert torch.equal(added[mask == 0], torch.zeros_like(added[mask == 0])), fmt
        if not mask.any():
            continue
        steered_formats.add(fmt)
        expected = (strength * vectors[fmt]).expand_as(added[mask == 1])
        torch.testing.assert_close(added[mask == 1], expected, rtol=0, atol=1e-5)
        prompt_passes.setdefault(id(batch), (batch, mask))
    assert steered_formats == {"snap", "short"}
    # In each batch's first call after the read, every prompt token after `r` is steered.
    for batch, mask in prompt_passes.values():
        assert mask.sum(1).tolist() == [e["n_prompt_tokens"] - e["read_index"] - 1 for e in batch]
    numbers = json.loads((folder / "numbers.json").read_text(encoding="utf-8"))
    assert numbers["inputs"]["steering_direction_sha256"]


def check_steering_scales_with_its_strength(bench, episode_folder):
    """The steering layer's output at strengths 1 and 8, minus its output with no steering: zero over tokens 0..r,
    and 1 to 8 at every prompt token after the read."""
    from check2.gates.mechanism_setup import layer_outputs, plain_pipeline

    batch = [e for e in run_episodes.select(config(), episode_folder) if e["format"] == "snap"][:2]
    torch.manual_seed(0)
    vector = torch.randn(bench.model.config.hidden_size)
    outputs = {}
    for strength in (None, 1, 8):
        with plain_pipeline(bench) as pipeline:
            handles = [] if strength is None else [steering.install_steering(bench.model, pipeline.control, vector,
                                                                             strength, layer=TINY_STEERING_LAYER)]
            try:
                with layer_outputs(bench.model, [TINY_STEERING_LAYER]) as seen:
                    pipeline.prompt_pass(pipeline.prefill(batch), None, 0.0)
            finally:
                for handle in handles:
                    handle.remove()
        outputs[strength] = seen[TINY_STEERING_LAYER]
    plain_prefix, plain_rest = outputs[None]
    for strength in (1, 8):
        assert torch.equal(outputs[strength][0], plain_prefix)
    once, eight = outputs[1][1] - plain_rest, outputs[8][1] - plain_rest
    torch.testing.assert_close(once, vector.expand_as(once), rtol=0, atol=1e-5)
    torch.testing.assert_close(eight, 8 * once, rtol=1e-5, atol=1e-5)


def test_steering_adds_its_strength_times_the_direction_after_the_read(bench, episode_folder):
    check_steering_scales_with_its_strength(bench, episode_folder)


def test_ignoring_the_strength_turns_the_steering_check_red(bench, episode_folder):
    fails_under("steering-strength-ignored", lambda: check_steering_scales_with_its_strength(bench, episode_folder))
