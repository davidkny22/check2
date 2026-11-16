"""Episodes under two conditions the main runs do not cover: reworded pushback, and the expanded question set.

    python -m check2.data.condition_episodes

*Reworded pushback.* Every reasoned `test` episode again, with the base sentence of its
pushback replaced by one of the three rewordings and its reason and format request
unchanged. The episode keeps its id, so that it pairs with the same episode as the main runs
ran it. One file per base sentence, `data/episodes/test_<name>.jsonl`.

*Expanded pairs.* Reasoned episodes for the pairs of the expanded question set: every pair
in the snap format, and `random.Random(2051).sample` of 1,000 of the sorted pairs in the
short format. A question whose first answer does not comply has no first answer to push
back on, and its pairs are left out and counted, as the pool's are. One file,
`data/expansion/episodes.jsonl`.

Every new episode's template tail is asserted equal to the main episodes' tail of its
format, and its read index is the main episodes' tail position.
"""

from __future__ import annotations

import json
import random
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from check2.data import paths
from check2.data.episodes import FORMATS, NAMED_BASES, TemplateError, episode_id, render_prompt, tail_of
from check2.data.pairs import situation
from check2.data.pool import Question, complying, load_pool
from check2.data.tokenizer import get_tokenizer

REWORDINGS = tuple(name for name in NAMED_BASES if name != "standard")
EXPANSION = paths.DATA / "expansion"
EXPANDED_EPISODES = EXPANSION / "episodes.jsonl"
EXPANDED_QUESTIONS = EXPANSION / "questions.jsonl"
EXPANDED_FIRST_ANSWERS = EXPANSION / "first_answers.jsonl"
EXPANDED_PAIRS = EXPANSION / "pairs.jsonl"
EXPANDED_SPLIT = "expanded"
SHORT_SAMPLE = 1000
SHORT_SAMPLE_SEED = 2051
META = paths.EPISODES / "meta.json"


def reworded_file(base_sentence: str) -> str:
    """The name of the episode file of one reworded base sentence, under `data/episodes/`."""
    if base_sentence not in REWORDINGS:
        raise ValueError(f"{base_sentence!r} is not one of the rewordings {list(REWORDINGS)}")
    return f"test_{base_sentence}.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main_tails(meta_path: Path = META) -> tuple[dict[str, list[int]], int]:
    """The main episodes' template tail per format, and their tail position, from their `meta.json`."""
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return {fmt: list(ids) for fmt, ids in meta["tail_ids"].items()}, int(meta["tail_position"])


def with_prompts(rows: Sequence[dict], prompts: Sequence[str], tails: Mapping[str, list[int]],
                 tail_position: int) -> list[dict]:
    """The rows with their prompts, token counts and read indices; a tail unlike the main episodes' is an error."""
    encoded = get_tokenizer()(list(prompts))["input_ids"]
    out = []
    for row, prompt, ids in zip(rows, prompts, encoded):
        try:
            start, tail = tail_of(ids)
        except TemplateError as err:
            raise TemplateError(f"{row['episode_id']}: {err}") from err
        if tail != tails[row["format"]]:
            raise TemplateError(f"{row['episode_id']}: tail {tail} differs from the main episodes' "
                                f"{row['format']} tail {tails[row['format']]}")
        out.append({**row, "prompt": prompt, "read_index": start + tail_position, "n_prompt_tokens": len(ids)})
    return out


def reworded_episodes(base_sentence: str, test_episodes: Sequence[dict], questions: Mapping[str, Question],
                      tails: Mapping[str, list[int]], tail_position: int) -> list[dict]:
    """The reasoned `test` episodes with the named base sentence, in the order the file holds them."""
    reworded_file(base_sentence)
    rows = [dict(e) for e in test_episodes if e["form"] == "reasoned"]
    prompts = [render_prompt(questions[e["qid"]], e["form"], e["format"], e["target"], e["reason"],
                             base_sentence=base_sentence) for e in rows]
    return with_prompts(rows, prompts, tails, tail_position)


def expanded_questions(question_rows: Sequence[dict], first_answers: Sequence[dict]) -> dict[str, Question]:
    """The expanded set's questions whose first answer complies, as the pool's are held.

    The expanded set records no first-turn probabilities, so `p1` is empty.
    """
    answers = {r["qid"]: r for r in first_answers}
    out = {}
    for q in question_rows:
        a = answers[q["qid"]]
        if a["complies"]:
            out[q["qid"]] = Question(qid=q["qid"], subject=q["subject"], question=q["question"],
                                     choices=tuple(q["choices"]), gold=q["key"], a1=a["a1"],
                                     free_text=a["free_text"], complies=True, p1=())
    return out


def short_sample(pairs: Sequence[dict]) -> list[dict]:
    """The seeded sample of pairs run in the short format, drawn from the pairs sorted by question id then target."""
    ordered = sorted(pairs, key=lambda p: (p["qid"], p["target"]))
    return random.Random(SHORT_SAMPLE_SEED).sample(ordered, SHORT_SAMPLE)


def expanded_episodes(questions: Mapping[str, Question], pairs: Sequence[dict], tails: Mapping[str, list[int]],
                      tail_position: int) -> tuple[list[dict], dict]:
    """Every usable pair in the snap format and the seeded 1,000 in the short format, and the counts."""
    usable = [p for p in pairs if p["qid"] in questions]
    keys = [(p["qid"], p["target"]) for p in usable]
    if len(set(keys)) != len(keys):
        raise ValueError("two expanded pairs share a question and a target")
    by_format = {"snap": usable, "short": short_sample(usable)}
    rows = []
    for fmt in FORMATS:
        for p in by_format[fmt]:
            q = questions[p["qid"]]
            rows.append({"episode_id": episode_id(p["qid"], "reasoned", fmt, p["target"], p["writer"]),
                         "qid": p["qid"], "split": EXPANDED_SPLIT, "form": "reasoned", "format": fmt,
                         "target": p["target"], "reason": p["reason"], "writer": p["writer"], "a1": q.a1,
                         "gold": q.gold, "situation": situation("reasoned", q.a1, q.gold, p["target"])})
    rows.sort(key=lambda r: (r["qid"], r["target"], FORMATS.index(r["format"])))
    prompts = [render_prompt(questions[r["qid"]], "reasoned", r["format"], r["target"], r["reason"]) for r in rows]
    counts = {"pairs": len(pairs), "pairs_without_complying_first_answer": len(pairs) - len(usable),
              "questions_without_complying_first_answer": len({p["qid"] for p in pairs} - set(questions)),
              "episodes": {fmt: len(by_format[fmt]) for fmt in FORMATS}}
    return with_prompts(rows, prompts, tails, tail_position), counts


def main(argv: Sequence[str] | None = None) -> int:
    tails, tail_position = main_tails()
    pool = complying(load_pool())
    test = read_jsonl(paths.EPISODES / "test.jsonl")
    for name in REWORDINGS:
        episodes = reworded_episodes(name, test, pool, tails, tail_position)
        write_jsonl(paths.EPISODES / reworded_file(name), episodes)
        print(f"{reworded_file(name)}: {len(episodes)} episodes")
    questions = expanded_questions(read_jsonl(EXPANDED_QUESTIONS), read_jsonl(EXPANDED_FIRST_ANSWERS))
    episodes, counts = expanded_episodes(questions, read_jsonl(EXPANDED_PAIRS), tails, tail_position)
    write_jsonl(EXPANDED_EPISODES, episodes)
    print(f"{EXPANDED_EPISODES.relative_to(paths.ROOT).as_posix()}: {json.dumps(counts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
