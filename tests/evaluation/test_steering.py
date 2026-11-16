"""The Steering baseline's direction, one per reply format, on made-up states whose answer is known by construction."""

from __future__ import annotations

import json

import pytest
import torch

from check2.evaluation import steering

# Each format's held-minus-adopted difference, chosen before any code runs.
KNOWN = {"snap": [1.0, -2.0, 0.5, 0.0], "short": [-3.0, 0.0, 0.25, 4.0]}
BASE = [0.5, 1.5, -1.0, 2.0]
# Noise that cancels inside each group, so each group's mean is its centre exactly.
NOISE = ([1.0, 2.0, -1.0, 0.5], [0.25, -0.5, 3.0, 1.0])
# Group sizes that differ between the formats, as on the model: snap mostly adopts, short mostly holds.
SIZES = {"snap": {"not_adopted": 2, "adopted": 4}, "short": {"not_adopted": 4, "adopted": 2}}


def _record(eid: str, fmt: str, form: str, adopted: bool, target: str = "B", gold: str = "A") -> dict:
    return {"episode_id": eid, "format": fmt, "form": form, "target": target, "gold": gold,
            "a2": target if adopted else "C"}


def made_up(offset: float, sizes: dict = SIZES) -> tuple[list[dict], dict[str, torch.Tensor]]:
    """Records and states where format `f`'s held states centre on `BASE + KNOWN[f]` and its adopted states on `BASE`.

    Every state of the short format is moved by `offset` in every coordinate. Records that must
    not count (bare, and `target == gold`) carry states far from both centres.
    """
    records, states = [], {}
    for fmt in ("snap", "short"):
        shift = torch.full((4,), offset if fmt == "short" else 0.0, dtype=torch.float64)
        for group, adopted in (("not_adopted", False), ("adopted", True)):
            centre = torch.tensor(BASE, dtype=torch.float64) + shift
            if not adopted:
                centre = centre + torch.tensor(KNOWN[fmt], dtype=torch.float64)
            for i in range(sizes[fmt][group]):
                noise = torch.tensor(NOISE[(i // 2) % 2], dtype=torch.float64) * (1 if i % 2 == 0 else -1)
                eid = f"{fmt}:{group}:{i}"
                records.append(_record(eid, fmt, "named" if i % 2 else "reasoned", adopted))
                states[eid] = (centre + noise).float()
        records.append(_record(f"{fmt}:bare", fmt, "bare", True))
        states[f"{fmt}:bare"] = torch.full((4,), 1e6)
        for adopted in (False, True):
            eid = f"{fmt}:target-is-gold:{adopted}"
            records.append(_record(eid, fmt, "reasoned", adopted, target="A", gold="A"))
            states[eid] = torch.full((4,), -1e6)
    return records, states


def test_each_format_s_direction_is_its_own_known_difference_whatever_the_format_offset():
    for offset in (0.0, 1000.0, -250.0):
        vectors, _ = steering.direction(*made_up(offset))
        assert sorted(vectors) == ["short", "snap"]
        for fmt, known in KNOWN.items():
            assert torch.equal(vectors[fmt], torch.tensor(known)), (offset, fmt, vectors[fmt])


def test_the_counts_are_those_of_each_format_s_records():
    _, counts = steering.direction(*made_up(1000.0))
    assert counts == SIZES


@pytest.mark.parametrize("empty", ["adopted", "not_adopted"])
@pytest.mark.parametrize("fmt", ["snap", "short"])
def test_a_format_with_an_empty_group_is_refused(fmt, empty):
    sizes = {f: dict(s) for f, s in SIZES.items()}
    sizes[fmt][empty] = 0
    with pytest.raises(steering.SteeringError) as raised:
        steering.direction(*made_up(1000.0, sizes))
    message = str(raised.value)
    assert fmt in message
    assert f"{sizes[fmt]['not_adopted']} not adopted" in message and f"{sizes[fmt]['adopted']} adopted" in message


def test_a_batch_takes_the_vector_of_its_one_format_and_a_batch_of_two_formats_is_refused():
    vectors = {"snap": torch.ones(4), "short": -torch.ones(4)}
    assert torch.equal(steering.batch_vector(vectors, [{"format": "short"}, {"format": "short"}]), -torch.ones(4))
    assert torch.equal(steering.batch_vector(vectors, [{"format": "snap"}]), torch.ones(4))
    with pytest.raises(steering.SteeringError):
        steering.batch_vector(vectors, [{"format": "snap"}, {"format": "short"}])


def test_loading_gives_each_format_s_vector_and_refuses_a_file_with_one_vector_for_both(tmp_path):
    path = tmp_path / "direction.json"
    path.write_text(json.dumps({"vectors": {"snap": [1.0, 2.0], "short": [3.0, 4.0]}}), encoding="utf-8")
    vectors, sha = steering.load_direction(path)
    assert torch.equal(vectors["snap"], torch.tensor([1.0, 2.0]))
    assert torch.equal(vectors["short"], torch.tensor([3.0, 4.0]))
    assert len(sha) == 64
    path.write_text(json.dumps({"vector": [1.0, 2.0]}), encoding="utf-8")
    with pytest.raises(steering.SteeringError):
        steering.load_direction(path)
