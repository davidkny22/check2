"""Scored records built by hand for the reading, checkpoint and gate tests.

Every episode has the answer key's option A. Three kinds of episode carry a target:

- `right`: a `user_right` episode, first answer B, target A. Adopting means ending on A.
- `wrong`: a `hold` episode, first answer A, target B. Adopting means ending on B.
- `recheck`: a `recheck` episode, first answer C, target D. Adopting means ending on D.

A run is described per question as a list of booleans for each kind, one per episode,
true where the run adopted. Two runs built from descriptions of the same shape hold the
same episode ids, so they pair.
"""

from __future__ import annotations

KINDS = {
    # kind: (situation, a1, target, a2 when adopted, a2 when not)
    "right": ("user_right", "B", "A", "A", "B"),
    "wrong": ("hold", "A", "B", "B", "A"),
    "recheck": ("recheck", "C", "D", "D", "C"),
}


def record(episode_id, qid, fmt, form, situation, a1, target, a2, part=None):
    return {
        "episode_id": episode_id,
        "qid": qid,
        "form": form,
        "format": fmt,
        "situation": situation,
        "a1": a1,
        "gold": "A",
        "target": target,
        "a2": a2,
        "scored_by": "stated",
        "ended": True,
        "p": [1 / 3, 1 / 3, 1 / 3],
        "part": part,
    }


def build(questions, fmt="snap", form="reasoned", part=None):
    """Records from `{qid: {kind: [adopted, ...]}}`, in one format and form."""
    records = []
    for qid, kinds in questions.items():
        for kind, adoptions in kinds.items():
            situation, a1, target, adopted_a2, other_a2 = KINDS[kind]
            for i, adopted in enumerate(adoptions):
                records.append(
                    record(
                        f"{qid}-{fmt}-{form}-{kind}{i}",
                        qid,
                        fmt,
                        form,
                        situation,
                        a1,
                        target,
                        adopted_a2 if adopted else other_a2,
                        part,
                    )
                )
    return records


def adopting(count, of):
    """`count` adoptions among `of` episodes."""
    return [True] * count + [False] * (of - count)


QUESTIONS = [f"q{i:02d}" for i in range(10)]


def reasoned_run(wrong, right, fmt="snap", part="first", questions=QUESTIONS, right_episodes=20):
    """Per question one wrong-target episode and `right_episodes` right-target episodes, reasoned pushback.

    `wrong[i]` is whether the run adopted the wrong target on question i; `right[i]` is how
    many of the right targets it adopted.
    """
    return build(
        {
            qid: {"wrong": [bool(wrong[i])], "right": adopting(right[i], right_episodes)}
            for i, qid in enumerate(questions)
        },
        fmt=fmt,
        part=part,
    )


ALL = [1] * 10
NONE = [0] * 10


def settings(alpha=0.0, kind="none", signal="reader", seed=0, rank=32, beta=1.0, forms=("bare", "named", "reasoned"),
             **baseline):
    """A run's settings as `run.json` holds them, for the keys a run's role is read from."""
    return {
        "check2": {
            "writer": {"alpha": alpha, "disabled_layers": []},
            "adapter": {"kind": kind, "rank": rank, "beta": beta},
            "signal": signal,
            "baseline": {"system_prompt_suffix": None, "steering_strength": None, **baseline},
            "data": {"forms": list(forms), "reason_writer": "opus"},
            "signal_forming": "none",
            "seed": seed,
        }
    }
