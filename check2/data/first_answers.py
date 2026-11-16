"""The model's own first answers for questions that are not in the pool.

    python -m check2.data.first_answers <questions file> <output file>
    python -m check2.data.first_answers --draw-from-pool N <questions file> <output file>

Each question `{qid, question, choices}` (one JSON object per line) is put to the original
model as the first turn of the conversation format, and the model answers greedily. The
output line is `{qid, free_text, complies, a1}`: the text it wrote, whether that text is
exactly one option letter, bare or in brackets, and that letter, or null when it is not
one letter. The generation and the reading of the answer repeat the measured script that
made the pool's first answers (`measurements/protocol/protocol_check.py`): batches of 64
prompts sorted by length, 24 new tokens, the text stripped and cut to 60 characters.

With `--draw-from-pool N`, N `train` questions of the pool are drawn first and written to
the questions file, and after generation the answers are compared with the pool's stored
first answers; the count of equal texts is written as a measurement record.
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

from check2.data.episodes import END_OF_TURN, SYSTEM, first_user_turn
from check2.data.pool import Question, complying, load_pool
from check2.data.sampling import RUN_SEED, left_padded, load_original_model, model_revision, read_jsonl, \
    split_reply, write_jsonl
from check2.data.splits import load_splits
from check2.data.targets import FIRST_ANSWER_PATTERN

# From the measured script that made the pool's first answers.
BATCH_PROMPTS = 64
MAX_NEW_TOKENS = 24
FREE_TEXT_CHARACTERS = 60


def read_first_answer(text: str) -> dict:
    """The stored first answer, whether it is exactly one option letter, and that letter."""
    free_text = text.strip()[:FREE_TEXT_CHARACTERS]
    match = re.fullmatch(FIRST_ANSWER_PATTERN, free_text)
    return {"free_text": free_text, "complies": match is not None,
            "a1": match.group(2) if match else None}


def first_turn_prompt(tokenizer, question: dict) -> str:
    """The system turn and the first user turn, rendered for the model's reply."""
    q = Question(qid=question["qid"], subject="", question=question["question"],
                 choices=tuple(question["choices"]), gold="", a1="", free_text="", complies=False,
                 p1=(0.0, 0.0, 0.0, 0.0))
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": first_user_turn(q)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def generate_first_answers(model, tokenizer, questions: list[dict]) -> list[dict]:
    """Greedy first answers, in the order of `questions`."""
    import torch

    end_id = tokenizer.convert_tokens_to_ids(END_OF_TURN)
    prompts = [first_turn_prompt(tokenizer, q) for q in questions]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    texts: list[str | None] = [None] * len(prompts)
    device = next(model.parameters()).device
    for start in range(0, len(order), BATCH_PROMPTS):
        idx = order[start:start + BATCH_PROMPTS]
        ids, mask = left_padded([tokenizer(prompts[i])["input_ids"] for i in idx], tokenizer.pad_token_id)
        with torch.inference_mode():
            out = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device), do_sample=False,
                                 max_new_tokens=MAX_NEW_TOKENS, eos_token_id=end_id,
                                 pad_token_id=tokenizer.pad_token_id)
        for j, i in enumerate(idx):
            reply, _ended = split_reply(out[j, ids.shape[1]:].tolist(), end_id)
            texts[i] = tokenizer.decode(reply, skip_special_tokens=True)
    return [{"qid": q["qid"], **read_first_answer(t)} for q, t in zip(questions, texts)]


def draw_from_pool(n: int, seed: int = RUN_SEED) -> list[dict]:
    """N `train` questions of the pool: `random.Random(seed).sample` of the sorted ids."""
    pool = complying(load_pool())
    train = sorted(load_splits().train)
    drawn = random.Random(seed).sample(train, n)
    return [{"qid": qid, "question": pool[qid].question, "choices": list(pool[qid].choices)}
            for qid in sorted(drawn)]


def compare_with_pool(answers: list[dict]) -> dict:
    """How many generated first answers equal the pool's stored text, and which differ."""
    pool = complying(load_pool())
    differ = [{"qid": a["qid"], "generated": a["free_text"], "pool": pool[a["qid"]].free_text}
              for a in answers if a["free_text"] != pool[a["qid"]].free_text]
    return {"questions": len(answers), "equal_free_text": len(answers) - len(differ),
            "equal_a1": sum(a["a1"] == pool[a["qid"]].a1 for a in answers), "differ": differ}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("questions", type=Path, help="one {qid, question, choices} per line")
    parser.add_argument("output", type=Path, help="one {qid, free_text, complies, a1} per line")
    parser.add_argument("--draw-from-pool", type=int, default=None, metavar="N",
                        help="first draw N train questions of the pool into the questions file")
    args = parser.parse_args(argv)
    if args.draw_from_pool is not None:
        write_jsonl(args.questions, draw_from_pool(args.draw_from_pool))
    questions = read_jsonl(args.questions)
    revision = model_revision()
    model, tokenizer = load_original_model(revision)
    answers = generate_first_answers(model, tokenizer, questions)
    write_jsonl(args.output, answers)
    complying_count = sum(a["complies"] for a in answers)
    print(f"{len(answers)} first answers, {complying_count} complying, to {args.output}")
    if args.draw_from_pool is not None:
        from check2.record import write_record

        comparison = compare_with_pool(answers)
        record = write_record("measurement", "first_answers_reproduction", comparison,
                              config={"model_revision": revision, "drawn": args.draw_from_pool,
                                      "seed": RUN_SEED, "batch": BATCH_PROMPTS,
                                      "max_new_tokens": MAX_NEW_TOKENS})
        print(f"equal to the pool's free_text: {comparison['equal_free_text']} of {comparison['questions']}; "
              f"equal a1: {comparison['equal_a1']}; record {record}")
        for d in comparison["differ"]:
            print(f"  {d['qid']}: generated {d['generated']!r}, pool {d['pool']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
