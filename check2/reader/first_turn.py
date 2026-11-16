"""The model's probabilities over the four options at its first answer's letter.

One forward pass of the original model over the first-turn prompt, continued with the tokens
the model wrote before its letter. A bare answer such as `D` is one letter token. A bracketed
answer such as `(D)` was written either as `(` then `D`, or as the single token `(D`; which of
the two the model wrote is its greedy choice at the first position, between those two tokens.
The four probabilities are the softmax over the logits of the four options' tokens in the form
the model wrote (`A` to `D`, or `(A` to `(D`) at the letter's position.
"""
import torch

from check2.data.episodes import SYSTEM, first_user_turn
from check2.data.paths import LETTERS
from check2.data.pool import Question

BATCH_SIZE = 64
OPEN = "("
LETTER = "letter"                    # the answer's letter is one token
BRACKETED_LETTER = "bracketed_letter"  # the letter and its opening bracket are one token
BRACKET_THEN_LETTER = "bracket_then_letter"


def first_turn_prompt(tokenizer, question: Question) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": first_user_turn(question)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def before_letter(free_text: str, letter: str) -> str:
    """The text the model wrote before its letter."""
    position = free_text.find(letter)
    if position < 0:
        raise ValueError(f"the first answer {free_text!r} does not hold its letter {letter}")
    return free_text[:position]


def single_id(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(f"{text!r} is not one token: {ids}")
    return ids[0]


def option_ids(tokenizer, bracketed: bool) -> list[int]:
    """The four options' tokens: the letters, or the letters each joined to an opening bracket."""
    return [single_id(tokenizer, (OPEN if bracketed else "") + letter) for letter in LETTERS]


@torch.inference_mode()
def next_logits(model, sequences: list[list[int]], pad_id: int, batch_size: int = BATCH_SIZE) -> list[torch.Tensor]:
    """The next-token logits after each sequence, from left-padded forward passes, shortest first."""
    device = next(model.parameters()).device
    order = sorted(range(len(sequences)), key=lambda i: (len(sequences[i]), i))
    out: list[torch.Tensor | None] = [None] * len(sequences)
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        width = max(len(sequences[i]) for i in chunk)
        input_ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for b, i in enumerate(chunk):
            n = len(sequences[i])
            input_ids[b, width - n:] = torch.tensor(sequences[i], dtype=torch.long)
            attention_mask[b, width - n:] = 1
        logits = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device),
                       logits_to_keep=1).logits[:, -1].float().cpu()
        for b, i in enumerate(chunk):
            out[i] = logits[b]
    return out


def first_turn_reads(model, tokenizer, questions: list[Question],
                     batch_size: int = BATCH_SIZE) -> dict[str, tuple[list[float], str]]:
    """For each question, the four option probabilities and the form in which its letter was read."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    letters, joined, open_id = option_ids(tokenizer, False), option_ids(tokenizer, True), single_id(tokenizer, OPEN)
    prompts = [list(tokenizer(first_turn_prompt(tokenizer, q))["input_ids"]) for q in questions]
    first = next_logits(model, prompts, pad_id, batch_size)
    reads: dict[str, tuple[torch.Tensor, str]] = {}
    continued = []
    for i, q in enumerate(questions):
        prefix = before_letter(q.free_text, q.a1)
        if prefix == "":
            reads[q.qid] = (first[i][letters], LETTER)
        elif prefix != OPEN:
            raise ValueError(f"{q.qid}: the first answer {q.free_text!r} has {prefix!r} before its letter")
        elif first[i][open_id] >= first[i][joined[LETTERS.index(q.a1)]]:
            continued.append(i)
        else:
            reads[q.qid] = (first[i][joined], BRACKETED_LETTER)
    second = next_logits(model, [prompts[i] + [open_id] for i in continued], pad_id, batch_size)
    for i, logits in zip(continued, second):
        reads[questions[i].qid] = (logits[letters], BRACKET_THEN_LETTER)
    return {qid: (torch.softmax(reads[qid][0], dim=-1).tolist(), reads[qid][1]) for qid in sorted(reads)}


def first_turn_probabilities(model, tokenizer, questions: list[Question],
                             batch_size: int = BATCH_SIZE) -> dict[str, list[float]]:
    """For each question, the probabilities of options A to D at its first answer's letter."""
    return {qid: p for qid, (p, _) in first_turn_reads(model, tokenizer, questions, batch_size).items()}
