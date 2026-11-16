"""The Swap run's signal: each `user_right` or `hold` episode carries the `p` of a donor of the other situation.

A donor is an episode of the same form and format whose situation is the other of the two,
taken from the Check² run over the same split, with its `p` read from that run's records.
One `random.Random(2030)` draws one donor per episode with `rng.choice` over its sorted
donors, the episodes taken in the run's batch order. A `recheck` episode keeps its own `p`
and is left out at scoring.

For the wide-signal variant the donors come from the variant's own Check² run, and an
episode receives its donor's whole `S`: the donor's `p` from that run's records and the
donor's state at its read token from the states that run saved.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from check2 import record
from check2.evaluation.tables import IncompleteRun, load_run, run_role

DONOR_SEED = 2030
OTHER = {"user_right": "hold", "hold": "user_right"}
SWAP_ROLE = "check2"


class SwapError(ValueError):
    """The donors cannot be drawn: no single complete Check² run over the run's episodes."""


def check2_run(split: str, results: Path | None = None, writer_input: str = "probabilities") -> Path:
    """The folder of the one complete Check² run over `split`, on the main episodes, with the given writer input."""
    results = record.ROOT / record.RESULTS if results is None else Path(results)
    donor_role = SWAP_ROLE if writer_input == "probabilities" else f"{SWAP_ROLE} input={writer_input}"
    found = []
    for folder in sorted(p for p in results.iterdir() if (p / "run.json").is_file()) if results.exists() else []:
        try:
            meta, _ = load_run(folder)
        except IncompleteRun:
            continue
        if meta.get("split") == split and run_role(meta["config"]) == (donor_role, ""):
            found.append(folder)
    if len(found) != 1:
        raise SwapError(f"Swap reads its donors' p from one complete Check² run over {split}; found {len(found)}: "
                        f"{[f.name for f in found]}")
    return found[0]


def draw_donors(episodes_in_order: Sequence[Mapping[str, Any]],
                check2_records: Sequence[Mapping[str, Any]]) -> dict[str, tuple[str, list[float]]]:
    """For each `user_right` or `hold` episode, its donor's id and the donor's `p` from the Check² run."""
    ids = [e["episode_id"] for e in episodes_in_order]
    if sorted(ids) != sorted(r["episode_id"] for r in check2_records):
        raise SwapError("the Check² run does not cover exactly the Swap run's episodes")
    p_of = {r["episode_id"]: r["p"] for r in check2_records}
    pools: dict[tuple[str, str, str], list[str]] = {}
    for r in check2_records:
        pools.setdefault((r["form"], r["format"], r["situation"]), []).append(r["episode_id"])
    for key in pools:
        pools[key].sort()
    rng = random.Random(DONOR_SEED)
    out = {}
    for e in episodes_in_order:
        other = OTHER.get(e["situation"])
        if other is None:
            continue
        donors = pools.get((e["form"], e["format"], other))
        if not donors:
            raise SwapError(f"{e['episode_id']}: no {other} episode of form {e['form']} in the {e['format']} format")
        donor = rng.choice(donors)
        out[e["episode_id"]] = (donor, p_of[donor])
    return out


def swapped_states(states: torch.Tensor, batch: Sequence[Mapping[str, Any]],
                   donors: Mapping[str, tuple[str, list[float]]],
                   donor_states: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """The batch's read states `[B, d]` with each donor's saved state in place of a `user_right` or `hold` episode's own."""
    out = states.clone()
    for b, e in enumerate(batch):
        if e["episode_id"] in donors:
            donor = donors[e["episode_id"]][0]
            if donor not in donor_states:
                raise SwapError(f"{e['episode_id']}: its donor {donor} has no saved state")
            out[b] = donor_states[donor].to(device=out.device, dtype=out.dtype)
    return out


def swapped(p_reader: torch.Tensor, batch: Sequence[Mapping[str, Any]],
            donors: Mapping[str, tuple[str, list[float]]]) -> tuple[torch.Tensor, list[str | None]]:
    """The batch's `p` with each donor's in place of a `user_right` or `hold` episode's own, and the donor ids."""
    p = p_reader.clone()
    ids: list[str | None] = []
    for b, e in enumerate(batch):
        if e["episode_id"] in donors:
            donor, values = donors[e["episode_id"]]
            p[b] = torch.tensor(values, dtype=p.dtype, device=p.device)
            ids.append(donor)
        else:
            ids.append(None)
    return p, ids
