"""Putting a kept question's options in a seeded order, with the key moved with its option.

A question with an option that refers to other options ("all of the above", "both A and C")
keeps its order, because moving the options would change what that option says.
"""
import random
import re

from check2.data.expansion.paths import OPTION_ORDER_SEED
from check2.data.paths import LETTERS

_LETTER = r"\(?\b[A-D]\b\)?"
_REFERS = [
    # "all of the above", "none of these", "both of the options", "neither of the choices"
    re.compile(r"\b(all|none|both|neither|either|any)\b(\s+\w+){0,2}\s+(above|below|these|those|options|choices|answers)\b",
               re.IGNORECASE),
    # "option B", "choices A and C", "answer (D)"
    re.compile(r"\b(?i:options?|choices?|answers?)\s+" + _LETTER),
    # "A and C", "A, B and D", "(A) or (B)", "A & B"
    re.compile(_LETTER + r"(\s*,\s*" + _LETTER + r")*\s*,?\s*(and|or|&)\s+" + _LETTER),
]


def refers_to_options(choices: list[str]) -> bool:
    """Whether any option names other options or the list of options."""
    return any(p.search(c) for c in choices for p in _REFERS)


def order_options(questions: list[dict]) -> list[dict]:
    """Every question, sorted by id, with `choices`, `key` and `options_reordered` set.

    One `random.Random(2040)` serves every question in id order. A question whose options
    refer to other options takes no draw and keeps its order.
    """
    rng = random.Random(OPTION_ORDER_SEED)
    out = []
    for q in sorted(questions, key=lambda q: q["qid"]):
        if refers_to_options(q["choices"]):
            out.append({**q, "options_reordered": False})
            continue
        order = rng.sample(range(4), 4)
        choices = [q["choices"][i] for i in order]
        key = LETTERS[order.index(LETTERS.index(q["key"]))]
        out.append({**q, "choices": choices, "key": key, "options_reordered": True})
    return out
