"""The window and the tail on the model's own tokenizer, and the template check's teeth."""
import json

import pytest

from check2.data import paths
from check2.data.episodes import TAIL_LENGTH, TemplateError
from check2.data.tokenizer import get_tokenizer
from check2.reader.template import check_episodes
from check2.reader.window import last_user_content, spans, window_length


@pytest.fixture(scope="module")
def heldout():
    with (paths.EPISODES / "heldout.jsonl").open(encoding="utf-8") as f:
        return [json.loads(next(f)) for _ in range(12)]


@pytest.fixture(scope="module")
def meta():
    return json.loads((paths.EPISODES / "meta.json").read_text(encoding="utf-8"))


def test_window_holds_exactly_the_pushback(heldout):
    tok = get_tokenizer()
    for e in heldout:
        ids = tok(e["prompt"])["input_ids"]
        window_start, tail_start = spans(ids, tok)
        assert tail_start == len(ids) - TAIL_LENGTH
        assert tok.decode(ids[window_start:tail_start]) == last_user_content(e["prompt"])
        assert last_user_content(e["prompt"]).startswith("I don't think that's right.")


def test_window_of_a_reason_only_prompt_holds_the_reason():
    tok = get_tokenizer()
    reason = "The first isomorphism theorem settles it, so the statement holds."
    prompt = tok.apply_chat_template([{"role": "system", "content": "You are a helpful assistant."},
                                      {"role": "user", "content": reason}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(prompt)["input_ids"]
    window_start, tail_start = spans(ids, tok)
    assert tok.decode(ids[window_start:tail_start]) == reason


def test_a_prompt_without_the_tail_is_refused(heldout):
    tok = get_tokenizer()
    ids = tok(heldout[0]["prompt"])["input_ids"]
    with pytest.raises(TemplateError):
        spans(ids[:-1], tok)


def test_window_length_by_read_position():
    assert window_length(20, 0) == 20 - TAIL_LENGTH + 1
    assert window_length(20, TAIL_LENGTH - 2) == 19
    with pytest.raises(ValueError):
        window_length(20, TAIL_LENGTH - 1)
    with pytest.raises(ValueError):
        window_length(TAIL_LENGTH, 0)


def test_template_check_passes_on_the_episodes_and_fails_on_a_wrong_record(heldout, meta):
    tok = get_tokenizer()
    assert check_episodes(tok, heldout, meta["tail_ids"], meta["tail_position"]) == (0, [])
    wrong = [dict(e) for e in heldout[:3]]
    wrong[0]["n_prompt_tokens"] += 1
    wrong[1]["read_index"] -= 1
    other = {fmt: list(t) for fmt, t in meta["tail_ids"].items()}
    other[wrong[2]["format"]][3] += 1
    failing, findings = check_episodes(tok, wrong, other, meta["tail_position"])
    assert failing == 3 and len(findings) == 3
