"""A subject's raw generated rows turned into usable questions, with every dropped row listed."""
import csv
from pathlib import Path

from check2.data.expansion import files, paths
from check2.data.expansion.parse import parse_key, split_prompt
from check2.data.expansion.repeats import drop_repeats
from check2.data.pool import load_pool, normalise


def raw_files(subject: str, raw_dir: Path) -> list[Path]:
    """The raw files of a subject, in the order their rows are numbered."""
    if subject in paths.EARLIER_RAW:
        return [paths.ADAPTION / name for name in paths.EARLIER_RAW[subject]]
    return [raw_dir / f"raw_{subject}.jsonl"]


def reference_texts() -> set[str]:
    """The normalised texts of every pool question and every seed question."""
    texts = {normalise(q.question) for q in load_pool()}
    for path in sorted(paths.ADAPTION.glob("seed_*.csv")):
        with path.open(encoding="utf-8", newline="") as f:
            texts.update(normalise(row["question"]) for row in csv.DictReader(f))
    return texts


def parse_subject(subject: str, sources: list[Path], reference: set[str]) -> tuple[list[dict], list[dict], dict]:
    """The subject's questions for curation, its dropped rows with the reason, and the counts.

    Rows are numbered across `sources` in order; question ids are `expansion/<subject>/<n>`.
    """
    usable, dropped, n = [], [], 0
    for source in sources:
        for row_index, row in enumerate(files.read_jsonl(source)):
            qid = f"expansion/{subject}/{n}"
            n += 1
            where = {"qid": qid, "source": source.name, "row": row_index}
            parsed = split_prompt(row["original_prompt"])
            key = parse_key(row["original_completion"])
            if parsed is None:
                dropped.append({**where, "why": "does_not_split"})
            elif key is None:
                dropped.append({**where, "why": "key_not_a_letter"})
            else:
                usable.append({"qid": qid, "subject": subject, "question": parsed.question,
                               "choices": list(parsed.choices), "key": key, **where})
    known = [q for q in usable if normalise(q["question"]) in reference]
    fresh = [q for q in usable if normalise(q["question"]) not in reference]
    kept, repeated = drop_repeats(fresh, set())
    for q, why in [(q, "pool_or_seed") for q in known] + [(q, "repeat") for q in repeated]:
        dropped.append({"qid": q["qid"], "source": q["source"], "row": q["row"], "why": why})
    dropped.sort(key=lambda d: int(d["qid"].rsplit("/", 1)[1]))
    counts = {
        "rows": n,
        "does_not_split": sum(d["why"] == "does_not_split" for d in dropped),
        "key_not_a_letter": sum(d["why"] == "key_not_a_letter" for d in dropped),
        "usable": len(usable),
        "dropped_pool_or_seed": len(known),
        "dropped_repeat": len(repeated),
        "for_curation": len(kept),
    }
    return kept, dropped, counts


def write_subject(subject: str, raw_dir: Path) -> dict:
    """Parse one subject, write its question files and dropped list, and return its counts."""
    kept, dropped, counts = parse_subject(subject, raw_files(subject, raw_dir), reference_texts())
    files.write_jsonl(paths.GENERATED / f"{subject}.jsonl", kept)
    files.write_jsonl(paths.DROPPED / f"{subject}.jsonl", dropped)
    files.write_jsonl(paths.WITHOUT_KEYS / f"{subject}.jsonl",
                      [{"qid": q["qid"], "question": q["question"], "choices": q["choices"]} for q in kept])
    return counts
