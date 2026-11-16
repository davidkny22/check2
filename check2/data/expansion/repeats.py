"""Dropping generated questions whose normalised text was already seen."""
from check2.data.pool import normalise


def drop_repeats(questions: list[dict], seen: set[str]) -> tuple[list[dict], list[dict]]:
    """Split `questions`, in their order, into those whose normalised text is new and the rest.

    `seen` holds the normalised texts already taken (the pool, the seeds, earlier rows). It
    is extended with every question kept, so a later copy of a kept question is a repeat.
    """
    kept, repeated = [], []
    for q in questions:
        text = normalise(q["question"])
        if text in seen:
            repeated.append(q)
        else:
            seen.add(text)
            kept.append(q)
    return kept, repeated
