"""First-turn option probabilities, read at the letter in the form the model wrote it."""
import pytest
import torch

from check2.data.pool import Question
from check2.reader import first_turn
from check2.reader.first_turn import (BRACKET_THEN_LETTER, BRACKETED_LETTER, LETTER, before_letter,
                                      first_turn_probabilities, first_turn_prompt, first_turn_reads)
from tests.reader.helpers import FakeTokenizer
from tests.tiny_model import tiny_model


class JoinedTokenizer(FakeTokenizer):
    """The made-up tokenizer, with an opening bracket and a letter joined into one token."""

    def _encode(self, text):
        out = []
        for i in super()._encode(text):
            token = {v: k for k, v in self.vocab.items()}[i]
            if out and token in "ABCD" and len(token) == 1 and out[-1] == self.vocab["("]:
                out[-1] = self._id("(" + token)
            else:
                out.append(i)
        return out


def _q(qid, free_text, a1):
    return Question(qid=qid, subject="x", question=f"What is {qid}?", choices=("one", "two", "three", "four"),
                    gold="A", a1=a1, free_text=free_text, complies=True, p1=(0.25,) * 4)


def _tokenizer():
    tok = JoinedTokenizer()
    for t in ["(", "A", "B", "C", "D", "(A", "(B", "(C", "(D"]:
        tok._id(t)
    return tok


def test_text_before_the_letter():
    assert before_letter("(C)", "C") == "("
    assert before_letter("B", "B") == ""
    with pytest.raises(ValueError):
        before_letter("B", "C")


def test_the_prompt_is_the_first_turn():
    tok = _tokenizer()
    assert first_turn_prompt(tok, _q("x/2", "B", "B")).endswith(
        "Answer with only the letter of the correct option.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def _direct(model, ids, option_tokens):
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits[0, -1]
    return logits, torch.softmax(logits[option_tokens], dim=-1)


@pytest.mark.parametrize("bracket_wins", [True, False])
def test_each_answer_is_read_in_the_form_the_model_wrote(bracket_wins):
    tok = _tokenizer()
    torch.manual_seed(0)
    model = tiny_model()
    letters = [tok.vocab[x] for x in "ABCD"]
    joined = [tok.vocab["(" + x] for x in "ABCD"]
    open_id = tok.vocab["("]
    bare, bracketed = _q("x/2", "B", "B"), _q("x/1", "(C)", "C")
    # make the model's first choice between "(" and "(C" go one way: set the head row of "(" so
    # that its logit after this prompt is that of "(C" plus or minus one
    with torch.no_grad():
        ids = tok(first_turn_prompt(tok, bracketed))["input_ids"]
        h = model(input_ids=torch.tensor([ids]), output_hidden_states=True).hidden_states[-1][0, -1]
        sign = 1.0 if bracket_wins else -1.0
        model.lm_head.weight[open_id] = model.lm_head.weight[joined[2]] + sign * h / h.dot(h)
    reads = first_turn_reads(model, tok, [bracketed, bare, _q("long/3", "(A)", "A")], batch_size=2)
    assert sorted(reads) == ["long/3", "x/1", "x/2"]

    prompt = tok(first_turn_prompt(tok, bare))["input_ids"]
    _, expected = _direct(model, prompt, letters)
    torch.testing.assert_close(torch.tensor(reads["x/2"][0]), expected, rtol=1e-4, atol=1e-6)
    assert reads["x/2"][1] == LETTER

    prompt = tok(first_turn_prompt(tok, bracketed))["input_ids"]
    logits, joined_p = _direct(model, prompt, joined)
    assert (logits[open_id] >= logits[joined[2]]) == bracket_wins
    if bracket_wins:
        _, expected = _direct(model, prompt + [open_id], letters)
        assert reads["x/1"][1] == BRACKET_THEN_LETTER
    else:
        expected = joined_p
        assert reads["x/1"][1] == BRACKETED_LETTER
    torch.testing.assert_close(torch.tensor(reads["x/1"][0]), expected, rtol=1e-4, atol=1e-6)
    assert first_turn_probabilities(model, tok, [bracketed], batch_size=2)["x/1"] == pytest.approx(reads["x/1"][0])


def test_an_option_that_is_not_one_token_is_refused():
    tok = FakeTokenizer()   # writes "(A" as two tokens
    with pytest.raises(ValueError, match="not one token"):
        first_turn.option_ids(tok, bracketed=True)
