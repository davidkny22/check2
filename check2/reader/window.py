"""Where the reader's window starts and where the fixed template tail begins, in a prompt's token ids.

The window runs from the first token of the last user turn's content through the read
token. The tail is the fixed template tokens after that content: the turn close, the
assistant header and the empty thinking block, `TAIL_LENGTH` tokens ending the prompt.
"""
from check2.data.episodes import END_OF_TURN, TAIL_LENGTH, TemplateError, check_tail_position

USER_HEADER = "<|im_start|>user\n"


def last_user_content(prompt: str) -> str:
    """The text of the last user turn of a rendered prompt: the content the window covers."""
    return prompt.rsplit(USER_HEADER, 1)[1].split(END_OF_TURN, 1)[0]


def user_header_ids(tokenizer) -> list[int]:
    """The token ids that open a user turn, before its content."""
    return tokenizer(USER_HEADER, add_special_tokens=False)["input_ids"]


def spans(token_ids: list[int], tokenizer) -> tuple[int, int]:
    """The index of the first token of the last user turn's content, and of the first tail token.

    Raises `TemplateError` when the prompt does not end in a tail of `TAIL_LENGTH` tokens that
    opens with the end-of-turn token, or holds no user turn with content before it.
    """
    end_id = tokenizer.convert_tokens_to_ids(END_OF_TURN)
    tail_start = len(token_ids) - TAIL_LENGTH
    if tail_start < 0 or token_ids[tail_start] != end_id or end_id in token_ids[tail_start + 1:]:
        raise TemplateError(f"the prompt does not end in a template tail of {TAIL_LENGTH} tokens")
    header = user_header_ids(tokenizer)
    n = len(header)
    starts = [i for i in range(tail_start - n + 1) if token_ids[i:i + n] == header]
    if not starts:
        raise TemplateError("the prompt has no user turn before its template tail")
    window_start = starts[-1] + n
    if window_start >= tail_start:
        raise TemplateError("the last user turn has no content")
    return window_start, tail_start


def window_length(saved_length: int, tail_position: int) -> int:
    """The window's length at `tail[tail_position]`, from the number of tokens saved for an episode.

    An episode's saved tokens run from the first token of the last user turn's content through
    the last tail token, so the window at `tail[k]` drops the last `TAIL_LENGTH - 1 - k` of them.
    """
    check_tail_position(tail_position)
    if saved_length <= TAIL_LENGTH:
        raise ValueError(f"{saved_length} saved tokens hold no content before the {TAIL_LENGTH} tail tokens")
    return saved_length - (TAIL_LENGTH - 1 - tail_position)
