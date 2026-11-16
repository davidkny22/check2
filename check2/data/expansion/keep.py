"""The kept questions of every subject, in a seeded option order, written as one file.

Subjects are taken in sorted order. A question whose normalised text matches a usable row
of an earlier subject is dropped as a repeat; the rest are kept where the curator confirms
them, and the kept questions are put in their seeded option order.
"""
from check2.data.expansion import files, paths
from check2.data.expansion.curation import apply_curation
from check2.data.expansion.options import order_options
from check2.data.expansion.repeats import drop_repeats

QUESTION_FIELDS = ("qid", "subject", "question", "choices", "key", "options_reordered")


def keep_questions(generated: dict[str, list[dict]], curations: dict[str, dict]) -> tuple[list[dict], dict]:
    """The kept questions in id order with their options ordered, and the counts per subject."""
    seen: set[str] = set()
    kept, counts = [], {}
    for subject in sorted(generated):
        _, repeated = drop_repeats(generated[subject], seen)
        confirmed, curated = apply_curation(generated[subject], curations[subject])
        repeated_ids = {q["qid"] for q in repeated}
        subject_kept = [q for q in confirmed if q["qid"] not in repeated_ids]
        counts[subject] = {**curated, "dropped_repeat_of_earlier_subject": len(repeated),
                           "kept": len(subject_kept), "options_reordered": 0, "options_kept_in_order": 0}
        kept.extend(subject_kept)
    ordered = order_options(kept)
    for q in ordered:
        counts[q["subject"]]["options_reordered" if q["options_reordered"] else "options_kept_in_order"] += 1
    return [{f: q[f] for f in QUESTION_FIELDS} for q in ordered], counts


def write_questions(subjects: list[str]) -> dict:
    """Keep and order the questions of `subjects` and write the two question files."""
    generated = {s: files.read_jsonl(paths.GENERATED / f"{s}.jsonl") for s in subjects}
    curations = {s: files.read_json(paths.CURATION / f"{s}.json") for s in subjects}
    questions, counts = keep_questions(generated, curations)
    files.write_jsonl(paths.QUESTIONS, questions)
    files.write_jsonl(paths.QUESTIONS_WITHOUT_KEYS,
                      [{"qid": q["qid"], "question": q["question"], "choices": q["choices"]} for q in questions])
    return counts
