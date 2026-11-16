"""The check on the appended read: its agreement with the answer a reply stated, on a seeded sample.

Every reply of a run carries `appended_read`, the letter the appended read gives. For a reply
that stated no answer that letter is its `a2`. For a reply that stated one, it is the check.
The checked sample is 200 stated replies per situation in each format, or all of them where
a group has fewer, each group drawn by a fresh `random.Random(2034)` from its sorted ids.
The appended read is trusted in a run when it agrees with the stated answer on at least 97%
of the sample; below that, the run's rates on stated answers only stand beside its full rates.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from typing import Any

from check2.evaluation.rates import rate

SAMPLE_SEED = 2034
SAMPLE_PER_GROUP = 200
LEAST_AGREEMENT = 0.97


def _groups(records: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str | None], list[str]]:
    groups: dict[tuple[str, str | None], list[str]] = {}
    for r in records:
        if r["scored_by"] == "stated":
            groups.setdefault((r["format"], r["situation"]), []).append(r["episode_id"])
    return groups


def sample(records: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str | None], list[str]]:
    """Per (format, situation), the ids of the stated replies in the checked sample."""
    out = {}
    for key, ids in sorted(_groups(records).items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
        ids = sorted(ids)
        out[key] = random.Random(SAMPLE_SEED).sample(ids, min(SAMPLE_PER_GROUP, len(ids)))
    return out


def agreement(records: Iterable[Mapping[str, Any]]) -> dict:
    """The appended read's agreement with the stated answer on the sample, per group and in all."""
    records = list(records)
    by_id = {r["episode_id"]: r for r in records}
    stated = _groups(records)
    groups, agree_all, n_all = [], 0, 0
    for (fmt, situation), ids in sample(records).items():
        agree = sum(by_id[i]["appended_read"] == by_id[i]["a2"] for i in ids)
        groups.append({"format": fmt, "situation": situation, "agreement": rate(agree, len(ids)),
                       "stated_replies": len(stated[(fmt, situation)])})
        agree_all, n_all = agree_all + agree, n_all + len(ids)
    return {"groups": groups, "all": rate(agree_all, n_all)}


def below_least_agreement(check: Mapping[str, Any]) -> bool:
    """Whether an agreement from `agreement` is below the least the appended read is trusted at."""
    return check["all"]["rate"] is not None and check["all"]["rate"] < LEAST_AGREEMENT
