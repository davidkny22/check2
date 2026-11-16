"""The tie rule of the padding and cache checks: a disagreement inside the rounding bound is a tie."""

from __future__ import annotations

import pytest
import torch

from check2.gates import cached_equals_uncached, padding_invariance, token_agreement
from check2.mechanism.generate import Pipeline, StepScores

DRIFT = 1e-3     # the top-logit difference planted in the arm without additions
STEP = 1         # the step at which the arm with additions is made to disagree
INSIDE, OUTSIDE = 0.9, 1.1   # the planted margin, as a share of the bound


def _scores(rows: list[list[tuple[int, float, int, float]]]) -> list[StepScores]:
    """Scores from (top id, top logit, second id, second logit) per step."""
    out = []
    for steps in rows:
        ids = torch.tensor([[a, c] for a, _, c, _ in steps])
        values = torch.tensor([[b, d] for _, b, _, d in steps], dtype=torch.float64)
        out.append(StepScores(ids, values))
    return out


EPISODES = [{"episode_id": "q:one"}, {"episode_id": "q:two"}]


def test_every_disagreement_of_a_row_is_examined_and_the_drift_is_where_tokens_agree():
    reference = _scores([[(5, 2.0, 6, 1.0), (7, 3.0, 8, 2.99), (9, 1.0, 4, 0.5)], [(1, 1.0, 2, 0.0)]])
    other = _scores([[(5, 2.25, 6, 1.0), (8, 3.0, 7, 2.999), (4, 1.0, 9, 0.9)], [(1, 1.5, 2, 0.0)]])
    result = token_agreement.compare_rows(EPISODES, [[5, 7, 9], [1]], other, reference)
    assert [(d["episode_id"], d["step"]) for d in result["disagreements"]] == [("q:one", 1), ("q:one", 2)]
    assert result["disagreements"][0]["reference_top_two_margin"] == pytest.approx(0.01)
    assert result["disagreements"][1]["reference_top_two_margin"] == pytest.approx(0.5)
    assert result["max_top_logit_difference_where_tokens_agree"] == pytest.approx(0.5)
    assert result["steps_compared"] == 4


def test_a_disagreement_inside_the_bound_is_a_tie_and_one_outside_counts():
    reference = _scores([[(7, 3.0, 8, 3.0 - 0.009)], [(7, 3.0, 8, 3.0 - 0.011)]])
    other = _scores([[(8, 3.0, 7, 2.99)], [(8, 3.0, 7, 2.99)]])
    result = token_agreement.judge(token_agreement.compare_rows(EPISODES, [[7], [7]], other, reference), 0.01)
    assert [d["counted"] for d in result["disagreements"]] == [False, True]
    assert (result["ties"], result["counted_disagreements"], result["rows_disagreeing"]) == (1, 1, 1)


def test_without_a_bound_every_disagreement_counts():
    reference = _scores([[(7, 3.0, 8, 3.0)]])
    other = _scores([[(8, 3.0, 7, 3.0)]])
    result = token_agreement.judge(token_agreement.compare_rows(EPISODES[:1], [[7]], other, reference), None)
    assert result["counted_disagreements"] == 1 and result["ties"] == 0


def test_the_bound_is_ten_times_the_drift_of_each_format():
    bounds = token_agreement.tie_bounds({"snap": {"max_top_logit_difference_where_tokens_agree": 2e-4},
                                         "short": {"max_top_logit_difference_where_tokens_agree": 0.0}})
    assert bounds == {"snap": pytest.approx(2e-3), "short": 0.0}


def _plant_near_tie(monkeypatch, target: str, share: float) -> None:
    """Plant a known drift in the arm without additions and a flip at a known margin in the other.

    In the arm without additions the other path's top logit at step 0 of the target row is
    raised by DRIFT, which leaves its tokens as they are. In the arm with additions the
    reference path's margin at STEP is set to `share` of ten times DRIFT, and the other path
    puts the second token on top there.
    """
    generate, teacher_forced = Pipeline.generate, Pipeline.teacher_forced_logits
    margin = share * token_agreement.TIE_FACTOR * DRIFT

    def planted_generate(self, prefilled, *args, **kwargs):
        replies = generate(self, prefilled, *args, **kwargs)
        for episode, reply in zip(prefilled.episodes, replies):
            if self.writers is not None and episode["episode_id"] == target:
                values = reply.scores.top_values
                with torch.inference_mode():
                    values[STEP, 1] = values[STEP, 0] - margin
        return replies

    def planted_teacher_forced(self, prefilled, *args, **kwargs):
        scores = teacher_forced(self, prefilled, *args, **kwargs)
        for episode, row in zip(prefilled.episodes, scores):
            if episode["episode_id"] != target:
                continue
            with torch.inference_mode():  # scores made under inference mode change only inside it
                if self.writers is None:
                    row.top_values[0, 0] += DRIFT
                else:
                    row.top_ids[STEP] = row.top_ids[STEP].flip(0)
        return scores

    monkeypatch.setattr(Pipeline, "generate", planted_generate)
    monkeypatch.setattr(Pipeline, "teacher_forced_logits", planted_teacher_forced)


@pytest.mark.parametrize("gate", [padding_invariance, cached_equals_uncached], ids=lambda g: g.NAME)
def test_on_the_tiny_model_a_flip_just_inside_the_bound_is_a_tie(bench, episodes, monkeypatch, gate):
    snap = {"snap": episodes["snap"]}
    _plant_near_tie(monkeypatch, snap["snap"][0]["episode_id"], INSIDE)
    values = gate.check(bench, snap)
    assert values["control_arm_agrees"] and values["green"], values
    assert values["ties"] == 1 and values["counted_disagreements"] == 0
    assert values["tie_bound_per_format"]["snap"] == pytest.approx(token_agreement.TIE_FACTOR * DRIFT, rel=1e-3)


@pytest.mark.parametrize("gate", [padding_invariance, cached_equals_uncached], ids=lambda g: g.NAME)
def test_on_the_tiny_model_a_flip_just_outside_the_bound_is_red(bench, episodes, monkeypatch, gate):
    snap = {"snap": episodes["snap"]}
    _plant_near_tie(monkeypatch, snap["snap"][0]["episode_id"], OUTSIDE)
    values = gate.check(bench, snap)
    assert values["control_arm_agrees"] and not values["green"]
    assert values["counted_disagreements"] == 1 and values["ties"] == 0


def test_removing_the_factor_turns_the_tie_red(bench, episodes, monkeypatch):
    snap = {"snap": episodes["snap"]}
    _plant_near_tie(monkeypatch, snap["snap"][0]["episode_id"], INSIDE)
    monkeypatch.setattr(token_agreement, "TIE_FACTOR", 1)
    values = cached_equals_uncached.check(bench, snap)
    assert not values["green"] and values["counted_disagreements"] == 1


def test_a_disagreement_without_additions_is_red_however_small(bench, episodes, monkeypatch):
    target = episodes["snap"][0]["episode_id"]
    teacher_forced = Pipeline.teacher_forced_logits

    def flipped(self, prefilled, *args, **kwargs):
        scores = teacher_forced(self, prefilled, *args, **kwargs)
        for episode, row in zip(prefilled.episodes, scores):
            if episode["episode_id"] == target and self.writers is None:
                with torch.inference_mode():
                    row.top_ids[STEP] = row.top_ids[STEP].flip(0)
        return scores

    monkeypatch.setattr(Pipeline, "teacher_forced_logits", flipped)
    values = cached_equals_uncached.check(bench, {"snap": episodes["snap"]})
    assert not values["control_arm_agrees"] and not values["green"]
    assert "additions_arm" not in values
