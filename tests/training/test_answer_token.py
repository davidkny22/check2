"""The answer position of a short-format target reply, on hand-built replies under the model's tokeniser.

Each case gives the reply and the character index of the letter the scorer would read,
found by hand in the reply's text. The expected position is the token whose character
span holds that index, taken from the tokeniser's own offsets.
"""

from __future__ import annotations

import pytest

from check2.data.episodes import END_OF_TURN
from check2.data.tokenizer import get_tokenizer
from check2.evaluation.scoring import score_stated
from check2.training import answer_token
from check2.training.answer_token import AnswerTokenError, AnswerWeighting, answer_position, weighting_for
from check2.training.examples import TrainingExample
from tests.evaluation import faults

ONE_BEFORE = "answer-position-one-token-early"
ONE_BEFORE_FAULT = ("check2.training.answer_token", "answer_position", "    return holding[0]\n",
                    "    return holding[0] - 1\n")

PLAIN = "The reason holds, so I keep my answer.\n\nFinal answer: (B)"
BOLD_LINE = "Option C fits the question.\n\n**Final answer: (C)**"
BOLD_LETTER = "Option C fits the question.\n\nFinal answer: **(C)**"
TWO_LINES = "Final answer: (A)\n\nOn reflection the user is right about D.\n\nFinal answer: (D)"
NO_BRACKETS = "I checked again.\n\nFinal answer: D"
JOINED = "I checked again.\n\nFinal answer:B"
# reply, the letter, the character index of the letter the last final-answer line states
CASES = {
    "plain": (PLAIN, "B", len(PLAIN) - 2),
    "bold markers around the line": (BOLD_LINE, "C", len(BOLD_LINE) - 4),
    "bold markers around the letter": (BOLD_LETTER, "C", len(BOLD_LETTER) - 4),
    "two final-answer lines": (TWO_LINES, "D", len(TWO_LINES) - 2),
    "no brackets": (NO_BRACKETS, "D", len(NO_BRACKETS) - 1),
    "letter joined to the colon": (JOINED, "B", len(JOINED) - 1),
}


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer()


def _expected(reply: str, letter: str, index: int, tokenizer) -> int:
    assert reply[index] == letter and score_stated(reply, "short") == letter
    encoded = tokenizer(reply, add_special_tokens=False, return_offsets_mapping=True)
    holding = [t for t, (start, end) in enumerate(encoded["offset_mapping"]) if start <= index < end]
    assert len(holding) == 1
    return holding[0]


def _positions_are_the_letters_tokens(tokenizer) -> None:
    for name, (reply, letter, index) in CASES.items():
        expected = _expected(reply, letter, index, tokenizer)
        got = answer_position(reply, tokenizer, name)
        assert got == expected, name
        # The position indexes the labels of the teacher-forced reply: its tokens, then the end-of-turn token.
        labels = tokenizer.encode(reply, add_special_tokens=False) + [tokenizer.convert_tokens_to_ids(END_OF_TURN)]
        assert letter in tokenizer.decode([labels[got]]), name
        assert tokenizer.decode(labels[:got]) + tokenizer.decode([labels[got]]) == tokenizer.decode(labels[:got + 1])
        assert len(tokenizer.decode(labels[:got])) <= index < len(tokenizer.decode(labels[:got + 1])), name


def test_the_answer_position_is_the_token_that_holds_the_letter(tokenizer):
    _positions_are_the_letters_tokens(tokenizer)


def test_a_position_one_token_early_turns_the_check_red(tokenizer, monkeypatch):
    monkeypatch.setitem(faults.FAULTS, ONE_BEFORE, ONE_BEFORE_FAULT)
    faults.fails_under(ONE_BEFORE, lambda: _positions_are_the_letters_tokens(tokenizer))


def test_the_last_final_answer_line_counts(tokenizer):
    reply, letter, index = CASES["two final-answer lines"]
    first = reply.index("(A)") + 1
    assert answer_position(reply, tokenizer, "two") == _expected(reply, letter, index, tokenizer)
    assert answer_position(reply, tokenizer, "two") > _expected(reply[:first + 2], "A", first, tokenizer)


def test_a_letter_joined_to_its_neighbour_is_one_token(tokenizer):
    for name in ("no brackets", "letter joined to the colon"):
        reply, letter, index = CASES[name]
        ids = tokenizer.encode(reply, add_special_tokens=False)
        token = tokenizer.decode([ids[answer_position(reply, tokenizer, name)]])
        assert letter in token and len(token) > 1, (name, token)


@pytest.mark.parametrize("reply", ["I keep my answer, which is B.", "(B)", "Final answer: none of them", ""])
def test_a_reply_that_states_no_final_answer_is_refused_by_name(tokenizer, reply):
    with pytest.raises(AnswerTokenError, match="train-0042-short"):
        answer_position(reply, tokenizer, "train-0042-short")


def test_the_folded_tokeniser_of_the_tiny_model_gives_the_same_positions(bench, tokenizer):
    for name, (reply, _, _) in CASES.items():
        assert answer_position(reply, bench.tokenizer, name) == answer_position(reply, tokenizer, name), name


def _example(episode_id: str, fmt: str, reply_tokens: int) -> TrainingExample:
    return TrainingExample(episode_id=episode_id, format=fmt, prompt_ids=(5, 6, 7),
                           target_ids=tuple(range(10, 10 + reply_tokens)) + (2,), read_index=1)


def test_only_short_format_examples_have_an_answer_position(tokenizer):
    examples = [_example("a-snap", "snap", 2), _example("b-short", "short", 20), _example("c-snap", "snap", 1)]
    weighting = weighting_for(0.5, examples, {"a-snap": "(B)", "b-short": PLAIN, "c-snap": "B"}, tokenizer)
    assert weighting.weight == 0.5 and set(weighting.positions) == {"b-short"}
    assert weighting.positions_of(examples).tolist() == [answer_token.NO_ANSWER, weighting.positions["b-short"],
                                                         answer_token.NO_ANSWER]


def test_a_weight_of_zero_gives_no_weighting(tokenizer):
    assert weighting_for(0.0, [_example("b-short", "short", 20)], {"b-short": PLAIN}, tokenizer) is None


def test_a_short_example_without_a_position_or_with_one_outside_its_reply_is_refused():
    short = _example("b-short", "short", 4)
    with pytest.raises(AnswerTokenError, match="b-short"):
        AnswerWeighting(0.5, {}).positions_of([short])
    for position in (-1, 4, 9):  # index 4 is the end-of-turn token
        with pytest.raises(AnswerTokenError, match="b-short"):
            AnswerWeighting(0.5, {"b-short": position}).positions_of([short])
    assert AnswerWeighting(0.5, {"b-short": 3}).positions_of([short]).tolist() == [3]
