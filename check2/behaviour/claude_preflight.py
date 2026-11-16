"""What putting the list to each Claude model would cost, estimated with no billed call.

Input tokens are counted by the provider's token-counting endpoint, which is not billed, on
a seeded sample of each kind and format of item, and scaled to the list. The follow-up
items depend on each model's own first answers, which do not exist before the run, so
they are built here on a placeholder first answer of one letter. The worst case prices
every call at its `max_tokens`. The expected cost prices a short-format reply at the mean
length of the short-format replies measured in the design, and every other reply at its
cap; for a model that thinks, thinking tokens are in the worst case only.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

from check2.behaviour import claude_list
from check2.behaviour.backend import Item
from check2.behaviour.claude import SETTINGS, count_tokens, max_tokens_for
from check2.behaviour.spend import MILLION, Price

PLACEHOLDER_FIRST_ANSWER = "A"
SAMPLE_PER_GROUP = 200
SAMPLE_SEED = 2054
REPLY_LENGTHS = Path(__file__).resolve().parents[2] / "measurements" / "protocol" / "reply_format_compare.json"


def expected_reply_tokens() -> dict[str, float]:
    """Mean reply length by format where the design measured one: the short format's replies."""
    data = json.loads(REPLY_LENGTHS.read_text(encoding="utf-8"))
    return {"short": float(data["short"]["tokens_mean"])}


def placeholder_items(model: str) -> list[Item]:
    """The model's list, with every follow-up built on the placeholder first answer."""
    first = claude_list.first_items(model)
    return first + claude_list.follow_ups({i.qid: PLACEHOLDER_FIRST_ANSWER for i in first})


def _group(item: Item) -> str:
    return item.kind if item.fmt is None else f"{item.kind}/{item.fmt}"


def estimate(client, model: str, price: Price, workers: int = 8) -> dict:
    """Calls, input tokens, worst-case and expected cost of one model's list."""
    groups: dict[str, list[Item]] = defaultdict(list)
    for it in placeholder_items(model):
        groups[_group(it)].append(it)
    rng = random.Random(SAMPLE_SEED)
    lengths = expected_reply_tokens()
    rows = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name in sorted(groups):
            members = sorted(groups[name], key=lambda i: i.item_id)
            sample = rng.sample(members, min(SAMPLE_PER_GROUP, len(members)))
            counts = list(pool.map(lambda it: count_tokens(client, model, it), sample))
            mean_in = Decimal(sum(counts)) / len(counts)
            cap = max_tokens_for(model, members[0])
            fmt = members[0].fmt
            reply = Decimal(str(min(lengths[fmt], members[0].max_new_tokens))) if fmt in lengths \
                else Decimal(members[0].max_new_tokens)
            n = len(members)
            rows[name] = {
                "calls": n, "sampled": len(sample), "mean_input_tokens": float(mean_in),
                "max_tokens": cap, "expected_reply_tokens": float(reply),
                "input_tokens": float(mean_in * n),
                "worst_usd": float(n * (mean_in * price.input_per_mtok + cap * price.output_per_mtok)
                                   / MILLION),
                "expected_usd": float(n * (mean_in * price.input_per_mtok + reply * price.output_per_mtok)
                                      / MILLION),
            }
    return {"model": model, "thinking": SETTINGS[model].label, "groups": rows,
            "calls": sum(r["calls"] for r in rows.values()),
            "input_tokens": sum(r["input_tokens"] for r in rows.values()),
            "worst_usd": sum(r["worst_usd"] for r in rows.values()),
            "expected_usd": sum(r["expected_usd"] for r in rows.values())}


def format_estimates(estimates: list[dict], cap: Decimal, spent: Decimal) -> str:
    lines = ["Pre-flight: no billed call was made. Input tokens counted by the token-counting endpoint",
             f"on a seeded sample (seed {SAMPLE_SEED}) of at most {SAMPLE_PER_GROUP} items per kind and "
             f"format, scaled to the list.",
             f"Follow-up items are built on a placeholder first answer of one letter "
             f"({PLACEHOLDER_FIRST_ANSWER!r}) for every question.",
             "Expected cost: short-format replies at the design's measured mean length, every other "
             "reply at its cap;", "thinking tokens are counted in the worst case only.", ""]
    for e in estimates:
        lines += [f"## {e['model']} ({e['thinking']})", "",
                  "| kind/format | calls | sampled | mean input tokens | max_tokens | expected reply tokens "
                  "| worst $ | expected $ |", "|---|---|---|---|---|---|---|---|"]
        for name, r in e["groups"].items():
            lines.append(f"| {name} | {r['calls']} | {r['sampled']} | {r['mean_input_tokens']:.1f} | "
                         f"{r['max_tokens']} | {r['expected_reply_tokens']:.1f} | {r['worst_usd']:.2f} | "
                         f"{r['expected_usd']:.2f} |")
        lines += [f"| all | {e['calls']} | | input tokens {e['input_tokens']:.0f} | | | "
                  f"{e['worst_usd']:.2f} | {e['expected_usd']:.2f} |", ""]
    worst = sum(e["worst_usd"] for e in estimates)
    expected = sum(e["expected_usd"] for e in estimates)
    lines += [f"Total: {sum(e['calls'] for e in estimates)} calls, worst case ${worst:.2f}, "
              f"expected ${expected:.2f}; spent so far ${float(spent):.2f}; cap ${float(cap):.2f}.",
              "The runner never lets spent plus reserved pass the cap, whatever the estimate."]
    return "\n".join(lines)
