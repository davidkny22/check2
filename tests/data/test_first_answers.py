"""First answers for new questions: the prompt, the reading of the answer, the draw from the pool."""

from check2.data.first_answers import compare_with_pool, draw_from_pool, first_turn_prompt, read_first_answer
from check2.data.pool import complying, load_pool
from check2.data.splits import load_splits
from check2.data.tokenizer import get_tokenizer


def test_one_option_letter_bare_or_in_brackets_complies():
    assert read_first_answer("C") == {"free_text": "C", "complies": True, "a1": "C"}
    assert read_first_answer(" (B)\n") == {"free_text": "(B)", "complies": True, "a1": "B"}
    assert read_first_answer("D.") == {"free_text": "D.", "complies": True, "a1": "D"}
    for text in ("The answer is C", "E", "(A) because", "A\n</think>\n\nA", ""):
        assert read_first_answer(text)["complies"] is False
        assert read_first_answer(text)["a1"] is None


def test_the_stored_text_is_cut_to_sixty_characters():
    assert read_first_answer("x" * 80)["free_text"] == "x" * 60


def test_the_prompt_is_the_first_turn_the_pool_was_made_with():
    q = {"qid": "n1", "question": "What is 2 + 2?", "choices": ["1", "2", "3", "4"]}
    # The measured script's first user turn: question, options in brackets, a blank line, the request.
    user = "What is 2 + 2?\n(A) 1\n(B) 2\n(C) 3\n(D) 4\n\nAnswer with only the letter of the correct option."
    tokenizer = get_tokenizer()
    expected = tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert first_turn_prompt(tokenizer, q) == expected


def test_the_pool_draw_takes_train_questions_and_repeats():
    drawn = draw_from_pool(32)
    assert len(drawn) == 32 and drawn == draw_from_pool(32)
    train = set(load_splits().train)
    assert all(q["qid"] in train for q in drawn)
    assert [q["qid"] for q in drawn] == sorted(q["qid"] for q in drawn)


def test_the_comparison_counts_equal_first_answers():
    pool = complying(load_pool())
    qids = sorted(load_splits().train)[:3]
    answers = [{"qid": qid, **read_first_answer(pool[qid].free_text)} for qid in qids]
    answers[1] = {"qid": qids[1], "free_text": "not a letter", "complies": False, "a1": None}
    result = compare_with_pool(answers)
    assert (result["questions"], result["equal_free_text"], result["equal_a1"]) == (3, 2, 2)
    assert [d["qid"] for d in result["differ"]] == [qids[1]]
