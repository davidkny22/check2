"""Comparing two generation paths token by token, and telling a rounding tie from a disagreement.

Two paths that add the same numbers in a different order differ in float32 by a small
amount, so where two tokens are nearly tied either can come out on top with no fault
present. The arm without additions measures that amount: its largest difference in the top
logit, per format, over the steps where its tokens agree. In the arm with additions a
disagreement counts only where the reference path's margin between its top two tokens is
more than `TIE_FACTOR` times that amount; one inside it is a tie, recorded with its margin
and not counted. Every row is compared to its end, so every disagreement is examined. In
the arm without additions every disagreement counts.
"""

from __future__ import annotations

TIE_FACTOR = 10


def compare_rows(episodes: list[dict], expected: list[list[int]], scores, reference) -> dict:
    """Every step where `scores` puts another token on top than the reference path's tokens.

    `expected[row]` are the reference path's tokens and `reference[row]` its scores;
    `scores[row]` are the other path's scores along those tokens. Returns the disagreements,
    each with both paths' top-two margins, and the largest top-logit difference over the
    steps where the two paths agree.
    """
    found, steps, drift = [], 0, 0.0
    for row, (episode, want, got) in enumerate(zip(episodes, expected, scores)):
        tokens, ref = got.tokens, reference[row]
        steps += len(want)
        for step, token in enumerate(want):
            if step < len(tokens) and tokens[step] == token:
                drift = max(drift, abs(float(got.top_values[step, 0] - ref.top_values[step, 0])))
                continue
            found.append({"episode_id": episode["episode_id"], "step": step, "expected": token,
                          "got": tokens[step] if step < len(tokens) else None,
                          "top_two_margin": got.margin(step) if step < len(tokens) else None,
                          "reference_top_two_margin": ref.margin(step)})
    return {"rows": len(expected), "steps_compared": steps, "disagreements": found,
            "max_top_logit_difference_where_tokens_agree": drift}


def judge(result: dict, bound: float | None) -> dict:
    """Mark each disagreement as counted or a tie; with no bound, every disagreement counts."""
    for d in result["disagreements"]:
        d["counted"] = bound is None or d["reference_top_two_margin"] > bound
    counted = [d for d in result["disagreements"] if d["counted"]]
    result["tie_bound"] = bound
    result["counted_disagreements"] = len(counted)
    result["ties"] = len(result["disagreements"]) - len(counted)
    result["rows_disagreeing"] = len({d["episode_id"] for d in counted})
    return result


def tie_bounds(control_arm: dict[str, dict]) -> dict[str, float]:
    """Per format, `TIE_FACTOR` times the arm without additions' largest top-logit difference."""
    return {fmt: TIE_FACTOR * r["max_top_logit_difference_where_tokens_agree"] for fmt, r in control_arm.items()}


def verdict(values: dict, compare_arm, formats, library_meaning: str) -> dict:
    """Run the arm without additions, then, if it agrees, the arm with additions under its bounds.

    `compare_arm(arm, fmt)` returns `compare_rows` output for one arm, `"control"` or
    `"additions"`, and one format.
    """
    values["control_arm"] = {fmt: judge(compare_arm("control", fmt), None) for fmt in formats}
    control_agrees = all(r["counted_disagreements"] == 0 for r in values["control_arm"].values())
    values["control_arm_agrees"] = control_agrees
    if not control_agrees:
        values["meaning"] = library_meaning
        values["green"] = False
        return values
    bounds = tie_bounds(values["control_arm"])
    values["tie_bound_per_format"] = bounds
    values["additions_arm"] = {fmt: judge(compare_arm("additions", fmt), bounds[fmt]) for fmt in formats}
    values["ties"] = sum(r["ties"] for r in values["additions_arm"].values())
    values["counted_disagreements"] = sum(r["counted_disagreements"] for r in values["additions_arm"].values())
    values["green"] = values["counted_disagreements"] == 0
    return values
