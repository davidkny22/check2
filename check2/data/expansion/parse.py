"""Splitting a generated row's prompt into one question and four options.

The generation service returns each row as a prompt and a completion. The prompt holds the
question and its four options in one of three layouts, tried in this order:

1. The last four non-empty lines are labelled `A.` to `D.` (or `A)`, `(A)`), in order.
2. The last blank-line-separated block has four or more lines: its last four lines are the
   options, one per line, and everything before them is the question.
3. There are five or more blocks and each of the last four is a single line: those four
   blocks are the options, and the blocks before them are the question.

The first layout that matches decides. A row is usable only when it gives a non-empty
question and four non-empty options that are distinct once trimmed (in the second
layout an indented line continues the line above it, as in code, and is no option), the question does not itself
split by the third layout (which marks a prompt holding several questions), and the
completion is one of the letters A to D.
"""
import re
from dataclasses import dataclass

from check2.data.paths import LETTERS

_BLANK_LINE = re.compile(r"\n\s*\n")
_LABEL = re.compile(r"^\(?([A-D])[.)]\s+(\S.*)$")


@dataclass(frozen=True)
class Parsed:
    question: str
    choices: tuple[str, str, str, str]


def _labelled(text: str) -> Parsed | None:
    lines = text.split("\n")
    filled = [i for i, line in enumerate(lines) if line.strip()]
    if len(filled) < 5:
        return None
    last = filled[-4:]
    found = [_LABEL.match(lines[i].strip()) for i in last]
    if not all(found) or [m.group(1) for m in found] != list(LETTERS):
        return None
    return Parsed("\n".join(lines[:last[0]]).strip(), tuple(m.group(2).strip() for m in found))


def _one_per_line(text: str) -> Parsed | None:
    blocks = _BLANK_LINE.split(text)
    lines = blocks[-1].split("\n")
    if len(lines) < 4:
        return None
    if any(line[:1].isspace() for line in lines[-4:]):
        # An indented line continues a line above it, as in code; it is not an option.
        return Parsed("", ())
    question = "\n\n".join(blocks[:-1] + ["\n".join(lines[:-4])]).strip()
    return Parsed(question, tuple(lines[-4:]))


def _one_per_block(text: str) -> Parsed | None:
    blocks = _BLANK_LINE.split(text)
    if len(blocks) < 5 or any("\n" in b for b in blocks[-4:]):
        return None
    return Parsed("\n\n".join(blocks[:-4]).strip(), tuple(blocks[-4:]))


def _options_ok(options: tuple[str, ...]) -> bool:
    if len(options) != 4 or any(not o.strip() for o in options):
        return False
    return len({o.strip() for o in options}) == 4


def split_prompt(prompt: str) -> Parsed | None:
    """The question and its four options, or None when the prompt does not split cleanly."""
    text = prompt.strip("\n").rstrip()
    for layout in (_labelled, _one_per_line, _one_per_block):
        parsed = layout(text)
        if parsed is None:
            continue
        if not parsed.question or not _options_ok(parsed.choices):
            return None
        if _one_per_block(parsed.question) is not None:
            return None
        return Parsed(parsed.question, tuple(o.strip() for o in parsed.choices))
    return None


def parse_key(completion: str) -> str | None:
    """The key letter of a completion that is exactly one of A to D, after trimming."""
    key = completion.strip()
    return key if key in LETTERS and len(key) == 1 else None
