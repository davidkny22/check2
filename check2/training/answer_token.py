"""The answer token of a short-format target reply, and its place among the positions the loss counts.

A short reply states its final answer in a line the scorer's short-format pattern reads;
the scorer takes the letter of the last match. The answer token is the one token of the
teacher-forced reply whose characters include that letter. The tokeniser may join the
letter with a bracket or a space, and the token is then the joined one.

The loss counts the reply's tokens and the end-of-turn token, in that order, and the loss
at position `t` is the loss of predicting the `t`-th of them. The answer position is
therefore the index of the answer token among the reply's tokens.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch

from check2.evaluation.scoring import SHORT_PATTERN
from check2.training.examples import TrainingExample

SHORT_FORMAT = "short"
# The answer position of an episode that has no answer token: any episode not in the short format.
NO_ANSWER = -1


class AnswerTokenError(ValueError):
    """A short-format target reply whose final answer cannot be tied to one token."""


def answer_position(reply: str, tokenizer, episode_id: str) -> int:
    """The index, among the reply's tokens as they are teacher-forced, of the token that holds the option
    letter of the last match of the short-format pattern.

    The reply is tokenised as a training example tokenises it, on its own and with no
    special tokens. A tokeniser that wraps the model's own (`tokenizer.base`) changes ids
    and not token boundaries, so the boundaries are read from the wrapped one.
    """
    matches = list(re.finditer(SHORT_PATTERN, reply))
    if not matches:
        raise AnswerTokenError(f"{episode_id}: the short-format target reply states no final answer")
    letter = matches[-1].start(1)
    encoded = getattr(tokenizer, "base", tokenizer)(reply, add_special_tokens=False, return_offsets_mapping=True)
    teacher_forced = len(tokenizer.encode(reply, add_special_tokens=False))
    if len(encoded["offset_mapping"]) != teacher_forced:
        raise AnswerTokenError(f"{episode_id}: {len(encoded['offset_mapping'])} token boundaries for "
                               f"{teacher_forced} teacher-forced reply tokens")
    holding = [t for t, (start, end) in enumerate(encoded["offset_mapping"]) if start <= letter < end]
    if len(holding) != 1:
        raise AnswerTokenError(f"{episode_id}: {len(holding)} tokens hold the letter of the final answer, at "
                               f"character {letter} of the reply")
    return holding[0]


@dataclass(frozen=True)
class AnswerWeighting:
    """The answer token's weight in a training, and each short-format episode's answer position."""

    weight: float
    positions: Mapping[str, int]

    def positions_of(self, examples: list[TrainingExample]) -> torch.Tensor:
        """`[B]`: each example's answer position, or `NO_ANSWER` for an example not in the short format."""
        out = []
        for e in examples:
            if e.format != SHORT_FORMAT:
                out.append(NO_ANSWER)
                continue
            if e.episode_id not in self.positions:
                raise AnswerTokenError(f"{e.episode_id}: a short-format episode with no answer position")
            position = self.positions[e.episode_id]
            # The last of an example's targets is the end-of-turn token, which is never the answer.
            if not 0 <= position < len(e.target_ids) - 1:
                raise AnswerTokenError(f"{e.episode_id}: answer position {position} is outside the reply's "
                                       f"{len(e.target_ids) - 1} tokens")
            out.append(position)
        return torch.tensor(out, dtype=torch.long)


def weighting_for(weight: float, examples: Iterable[TrainingExample], replies: Mapping[str, str],
                  tokenizer) -> AnswerWeighting | None:
    """The weighting of a training's examples, from their target replies' text; None at a weight of zero."""
    if weight == 0:
        return None
    positions = {e.episode_id: answer_position(replies[e.episode_id], tokenizer, e.episode_id)
                 for e in examples if e.format == SHORT_FORMAT}
    return AnswerWeighting(weight, positions)
