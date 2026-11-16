"""No question id and no normalised question text in more than one split, in any data file.

Checked over the saved splits, every episode file, the target replies, the reader's cache
index, and the reader's folds, where no fold may hold an id or a normalised text of
another fold. Run as `python -m check2.gates.split_isolation [--plant]`; `--plant` copies
one `test` question's text, with a space added, into a `train` episode in memory, and the
check must then fail.
"""
import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from check2.data import paths
from check2.data.pool import Question, complying, load_pool, normalise
from check2.data.splits import Splits, load_splits
from check2.record import write_record

SPLIT_NAMES = ("test", "heldout", "train")
FIRST_USER_TURN = re.compile(r"<\|im_start\|>user\n(.*?)\n\(A\) ", re.DOTALL)
MAX_LISTED = 20


@dataclass
class Result:
    findings: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def green(self) -> bool:
        return not self.findings

    def add(self, finding: str) -> None:
        self.findings.append(finding)


def _owners(splits: Splits, pool: dict[str, Question]) -> tuple[dict[str, str], dict[str, set[str]]]:
    """The split of each id, and the splits that hold each normalised text."""
    split_of: dict[str, str] = {}
    text_splits: dict[str, set[str]] = {}
    for name in SPLIT_NAMES:
        for qid in getattr(splits, name):
            split_of[qid] = name
            text_splits.setdefault(normalise(pool[qid].question), set()).add(name)
    return split_of, text_splits


def check_splits(splits: Splits, pool: dict[str, Question], result: Result) -> None:
    """Each id in one split, each text in one split, and the folds covering `train` exactly."""
    seen: dict[str, str] = {}
    for name in SPLIT_NAMES + ("in_no_split",):
        for qid in getattr(splits, name):
            if qid in seen:
                result.add(f"id {qid} is in {seen[qid]} and in {name}")
            seen[qid] = name
            if qid not in pool:
                result.add(f"id {qid} in {name} is not a complying pool question")
    split_of, text_splits = _owners(splits, pool)
    for text, owners in sorted(text_splits.items()):
        if len(owners) > 1:
            result.add(f"a normalised text is in {sorted(owners)}: {text[:60]}")
    test_texts = {normalise(pool[q].question) for q in splits.test}
    for qid in splits.in_no_split:
        if normalise(pool[qid].question) not in test_texts:
            result.add(f"{qid} is in no split but shares no text with a test question")
    if sorted(i for f in splits.folds for i in f) != sorted(splits.train):
        result.add("the folds do not hold exactly the train ids")
    result.counts.update({name: len(getattr(splits, name)) for name in SPLIT_NAMES + ("in_no_split",)})


def check_folds(folds: list[list[str]], pool: list[Question] | dict[str, Question],
                result: Result | None = None) -> Result:
    """No fold holds an id or a normalised text of another fold."""
    result = result or Result()
    by_id = pool if isinstance(pool, dict) else complying(pool)
    id_fold: dict[str, int] = {}
    text_fold: dict[str, int] = {}
    for k, fold in enumerate(folds):
        for qid in fold:
            if qid in id_fold and id_fold[qid] != k:
                result.add(f"id {qid} is in folds {id_fold[qid]} and {k}")
            id_fold.setdefault(qid, k)
            text = normalise(by_id[qid].question)
            if text in text_fold and text_fold[text] != k:
                result.add(f"{qid}'s text is in folds {text_fold[text]} and {k}")
            text_fold.setdefault(text, k)
    result.counts["folds"] = len(folds)
    return result


def question_text_of(prompt: str) -> str | None:
    """The question text of a rendered conversation: the first user turn before its options."""
    match = FIRST_USER_TURN.search(prompt)
    return match.group(1) if match else None


def check_episodes(name: str, episodes: list[dict], split_of: dict[str, str],
                   text_splits: dict[str, set[str]], result: Result) -> None:
    """Each episode's id and question text belong to the split the episode says it is in."""
    bad = 0
    for e in episodes:
        problems = []
        owner = split_of.get(e["qid"])
        if owner is None:
            problems.append(f"its question {e['qid']} is in no split")
        elif owner != e["split"]:
            problems.append(f"its question {e['qid']} is in {owner}")
        text = question_text_of(e["prompt"])
        if text is None:
            problems.append("its prompt has no question")
        else:
            others = text_splits.get(normalise(text), set()) - {e["split"]}
            if not text_splits.get(normalise(text)):
                problems.append("its question text is not a text of any split")
            elif others:
                problems.append(f"its question text belongs to {sorted(others)}")
        if problems:
            bad += 1
            if bad <= MAX_LISTED:
                result.add(f"{name}: episode {e['episode_id']} ({e['split']}): " + "; ".join(problems))
    if bad > MAX_LISTED:
        result.add(f"{name}: {bad - MAX_LISTED} more episodes fail")
    result.counts[name] = len(episodes)


def check_keyed(name: str, keys: list[str], episode_split: dict[str, str], split_of: dict[str, str],
                result: Result, only: set[str] | None = None) -> None:
    """Records keyed by episode id: each names a known episode whose question is in its split."""
    bad = 0
    for key in keys:
        qid = key.split(":", 1)[0]
        problem = None
        if key not in episode_split:
            problem = "names no episode"
        elif split_of.get(qid) != episode_split[key]:
            problem = f"its question is in {split_of.get(qid)}, its episode in {episode_split[key]}"
        elif only is not None and episode_split[key] not in only:
            problem = f"it belongs to {episode_split[key]}"
        if problem:
            bad += 1
            if bad <= MAX_LISTED:
                result.add(f"{name}: {key}: {problem}")
    if bad > MAX_LISTED:
        result.add(f"{name}: {bad - MAX_LISTED} more records fail")
    result.counts[name] = len(keys)


def _rel(path: Path) -> str:
    return path.relative_to(paths.ROOT).as_posix()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_episode_files(folder: Path = paths.EPISODES) -> dict[str, list[dict]]:
    if not folder.exists():
        return {}
    return {p.name: _read_jsonl(p) for p in sorted(folder.glob("*.jsonl"))}


def plant(episode_files: dict[str, list[dict]], splits: Splits, pool: dict[str, Question]) -> str:
    """Copy the first `test` question's text, with a space added, into the first `train` episode."""
    train = episode_files.get("train.jsonl") or []
    if not train:
        raise RuntimeError("no train episode to plant a test question into")
    episode = train[0]
    own = pool[episode["qid"]].question
    test_q = pool[sorted(splits.test)[0]]
    episode["prompt"] = episode["prompt"].replace(own, test_q.question + " ", 1)
    return f"{test_q.qid}'s text into {episode['episode_id']}"


def check(splits: Splits | None = None, pool: list[Question] | None = None,
          episode_files: dict[str, list[dict]] | None = None, targets: list[dict] | None = None,
          cache_index: dict | None = None) -> Result:
    """Run every part of the check over what is given, or over the files that exist."""
    result = Result()
    by_id = complying(pool if pool is not None else load_pool())
    if splits is None:
        splits = load_splits()
        result.files.append(_rel(paths.SPLITS))
    if episode_files is None:
        episode_files = load_episode_files()
        result.files += [_rel(paths.EPISODES / n) for n in episode_files]
    if targets is None and paths.TARGETS.exists():
        targets = _read_jsonl(paths.TARGETS)
        result.files.append(_rel(paths.TARGETS))
    if cache_index is None and paths.CACHE_INDEX.exists():
        cache_index = json.loads(paths.CACHE_INDEX.read_text(encoding="utf-8"))
        result.files.append(_rel(paths.CACHE_INDEX))
    check_splits(splits, by_id, result)
    check_folds(splits.folds, by_id, result)
    split_of, text_splits = _owners(splits, by_id)
    episode_split = {}
    for name, episodes in episode_files.items():
        check_episodes(name, episodes, split_of, text_splits, result)
        episode_split.update({e["episode_id"]: e["split"] for e in episodes})
    if targets is not None:
        check_keyed("targets", [t["episode_id"] for t in targets], episode_split, split_of, result,
                    only={"train"})
    if cache_index is not None:
        check_keyed("cache_index", sorted(cache_index), episode_split, split_of, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plant", action="store_true",
                        help="copy a test question's text into a train episode; the check must fail")
    args = parser.parse_args(argv)
    pool = load_pool()
    splits = load_splits()
    episode_files = load_episode_files()
    planted = plant(episode_files, splits, complying(pool)) if args.plant else None
    result = check(splits=splits, pool=pool, episode_files=episode_files)
    result.files = ([_rel(paths.SPLITS)]
                    + [_rel(paths.EPISODES / n) for n in episode_files]
                    + result.files)
    verdict = "green" if result.green else "red"
    record = write_record("gate", "split_isolation",
                          {"findings": result.findings, "counts": result.counts, "files": result.files},
                          verdict=verdict, extra={"planted": planted})
    print(f"files: {', '.join(result.files)}")
    for name, n in result.counts.items():
        print(f"  {name}: {n}")
    if planted:
        print(f"planted: {planted}")
    for finding in result.findings:
        print(f"FINDING {finding}")
    print(f"verdict: {verdict}  record: {_rel(record)}")
    return 0 if result.green else 1


if __name__ == "__main__":
    sys.exit(main())
