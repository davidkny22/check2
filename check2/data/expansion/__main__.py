"""Build the expanded question set one part at a time.

    python -m check2.data.expansion parse --raw-dir DIR SUBJECT [SUBJECT ...]
    python -m check2.data.expansion reuse-curation
    python -m check2.data.expansion keep SUBJECT [SUBJECT ...]
    python -m check2.data.expansion pairs
    python -m check2.data.expansion check-reasons
    python -m check2.data.expansion split-unwritten --size N --stem STEM --writer MODEL JOBFILE [JOBFILE ...]
    python -m check2.data.expansion set-aside JOBFILE [JOBFILE ...]
    python -m check2.data.expansion review-jobs
    python -m check2.data.expansion finish
    python -m check2.data.expansion readme

`parse` splits raw rows into questions for the curators. `reuse-curation` writes the
college mathematics curation from the earlier Sonnet answers. `keep` applies the curation,
drops repeats across subjects and orders the options. `pairs` draws the second pairs and
writes the writer jobs. `check-reasons` checks every writer file against its jobs; a reason
counts on its own terms even where its file is not whole. `split-unwritten` puts the jobs
of the named job files that have no counted reason into smaller job files, and `set-aside`
lists those still without one.
`review-jobs` joins the reasons, draws the review sample over the pairs with a reason, and
writes review jobs for the sampled pairs that no earlier review judged with the same reason.
`finish` applies the reviews, scores the style and writes `pairs.jsonl`. `readme` writes the README from the
counts. Each command stores its counts in `data/expansion/counts.json`.
"""
import argparse
import json
import sys
from pathlib import Path

from check2.data.expansion import files, paths
from check2.data.expansion.curation import earlier_curation
from check2.data.expansion.generated import write_subject
from check2.data.expansion.jobs import review_jobs_of, sample_jobs, write_review_jobs, write_writer_jobs
from check2.data.expansion.keep import write_questions
from check2.data.expansion.pairs import build_pairs
from check2.data.expansion.record import readme
from check2.data.expansion.reasons import counted_reasons, join_reasons, reason_faults, unwritten_jobs
from check2.data.expansion.smaller import subject_of, write_split
from check2.data.expansion.reviews import carried_judgements, judged_jobs, sample_outcome, without_removed
from check2.data.style import style_score


def _merge(name: str, new: dict) -> dict:
    current = files.read_json(paths.COUNTS).get(name, {}) if paths.COUNTS.exists() else {}
    merged = dict(sorted({**current, **new}.items()))
    files.record_counts(name, merged)
    return merged


def _parse(args) -> dict:
    return _merge("parse", {s: write_subject(s, Path(args.raw_dir)) for s in args.subjects})


def _reuse_curation(args) -> dict:
    questions = files.read_jsonl(paths.GENERATED / "college_mathematics.jsonl")
    curation = earlier_curation(questions)
    files.write_json(paths.CURATION / "college_mathematics.json", curation)
    answered = sum(a["answer"] is not None for a in curation["answers"])
    return {"questions": len(questions), "answered": answered, "not_answered": len(questions) - answered}


def _keep(args) -> dict:
    counts = write_questions(sorted(args.subjects))
    files.record_counts("keep", counts)
    return counts


def _first_answers() -> tuple[dict, str]:
    if not paths.FIRST_ANSWERS.exists():
        return {}, "absent: the second pair excludes only the key"
    return {r["qid"]: r for r in files.read_jsonl(paths.FIRST_ANSWERS)}, "used"


def _pairs(args) -> dict:
    questions = files.read_jsonl(paths.QUESTIONS)
    first, state = _first_answers()
    pairs = build_pairs(questions, first)
    files.write_jsonl(paths.EXPANSION / "pairs_drawn.jsonl", pairs)
    jobs = write_writer_jobs(questions, pairs)
    complying = [r for r in first.values() if r["complies"]]
    by_qid = {q["qid"]: q for q in questions}
    counts = {"first_answers": state, "questions": len(questions),
              "first_answers_given": sum(q["qid"] in first for q in questions),
              "complying": sum(r["qid"] in by_qid for r in complying),
              "wrong_first": sum(r["qid"] in by_qid and r["a1"] != by_qid[r["qid"]]["key"] for r in complying),
              "pairs": len(pairs), "writer_files": jobs}
    files.record_counts("pairs", counts)
    return counts


def _job_files() -> list[Path]:
    """Every writer job file: the files of 150 and the smaller files of their unwritten jobs."""
    return sorted(paths.JOBS.glob("*.json"))


def _reasons_path(job_path: Path) -> Path:
    return paths.REASONS / f"reasons_{job_path.name}"


def _check_reasons(args) -> dict:
    out = {}
    for job_path in _job_files():
        reason_path = _reasons_path(job_path)
        if not reason_path.exists():
            out[job_path.name] = "no reasons file"
            continue
        faults = reason_faults(files.read_json(job_path), files.read_json(reason_path))
        out[job_path.name] = "pass" if not faults else faults
    return out


def _counted_files() -> dict[str, dict]:
    """Each job file that has a reasons file, with only the reasons that count."""
    out = {}
    for job_path in _job_files():
        reason_path = _reasons_path(job_path)
        if reason_path.exists():
            out[job_path.name] = counted_reasons(files.read_json(job_path), files.read_json(reason_path))
    return out


def _passing_reasons() -> list[dict]:
    return list(_counted_files().values())


def _split_unwritten(args) -> dict:
    """The jobs of the named job files that no file yet has a counted reason for, in job
    files of at most `--size` named `<stem>_NNN.json`."""
    written = {r["job_id"] for w in _passing_reasons() for r in w["reasons"]}
    jobs = [j for name in args.sources for j in unwritten_jobs(files.read_json(paths.JOBS / name), written)]
    made = write_split(jobs, args.size, args.stem)
    counts = {"sources": args.sources, "size": args.size, "writer": args.writer, "unwritten": len(jobs),
              "files": made}
    _merge("smaller_files", {args.stem: counts})
    return counts


def _file_group(name: str) -> str:
    """The name a job file shares with the files made the same way: `writer_012.json` gives
    `writer`, `writer30_004.json` gives `writer30`."""
    return name.removesuffix(".json").rsplit("_", 1)[0]


def _reason_groups() -> dict[str, str]:
    """Each job id with a counted reason, with the group of the file that holds it."""
    return {r["job_id"]: _file_group(name) for name, w in _counted_files().items() for r in w["reasons"]}


def _set_aside(args) -> dict:
    """List the jobs of the named job files that still have no counted reason, once."""
    if paths.SET_ASIDE.exists():
        raise ValueError(f"{paths.SET_ASIDE.name} is written already")
    written = {r["job_id"] for w in _passing_reasons() for r in w["reasons"]}
    jobs = [j for name in args.sources for j in unwritten_jobs(files.read_json(paths.JOBS / name), written)]
    entries = [{"job_id": j["job_id"], "qid": j["job_id"].rsplit(":", 1)[0], "subject": subject_of(j["job_id"]),
                "target": j["argue_for"]} for j in jobs]
    files.write_json(paths.SET_ASIDE, {"sources": args.sources, "set_aside": entries})
    per_subject: dict[str, int] = {}
    for e in entries:
        per_subject[e["subject"]] = per_subject.get(e["subject"], 0) + 1
    counts = {"jobs": len(entries), "per_subject": dict(sorted(per_subject.items()))}
    files.record_counts("set_aside", counts)
    return counts


def _without_reasons() -> dict[str, str]:
    """Each writer job file without a passing reasons file, with why it has none."""
    stopped = files.read_json(paths.STOPPED_WRITERS) if paths.STOPPED_WRITERS.exists() else {}
    out = {}
    for name, state in _check_reasons(None).items():
        if state == "pass":
            continue
        why = ("it has no reasons file" if state == "no reasons file" else
               "its reasons file fails its check (" + "; ".join(state) + ")")
        if name in stopped:
            why = f"its writer was stopped by the model provider's safety classifier; {why}"
        out[name] = why
    return out


def _judged() -> list[tuple[dict, dict]]:
    """Every judgement made so far, from each review job file and its reviews file. Each
    review is marked with the name of the file that holds it."""
    judged = []
    for path in sorted(paths.REVIEW_JOBS.glob("review_*.json")):
        review_path = paths.REVIEWS / path.name
        if not review_path.exists():
            raise ValueError(f"{path.name} has no reviews file yet")
        judged += [(job, {**review, "review_file": path.name}) for job, review in
                   judged_jobs(files.read_json(path), files.read_json(review_path)["reviews"])]
    return judged


def _review_jobs(args) -> dict:
    questions = files.read_jsonl(paths.QUESTIONS)
    pairs = join_reasons(files.read_jsonl(paths.EXPANSION / "pairs_drawn.jsonl"), _passing_reasons())
    with_reason = [p for p in pairs if p["reason"]]
    files.write_jsonl(paths.EXPANSION / "pairs_with_reasons.jsonl", with_reason)
    sample = sample_jobs(questions, with_reason)
    carried, to_review = carried_judgements(sample, _judged())
    written = write_review_jobs(to_review)
    if not written and paths.COUNTS.exists():
        # Every sampled pair is judged already: the files written for this sample stay as recorded.
        written = files.read_json(paths.COUNTS).get("reasons", {}).get("review_files_written", {})
    carried_earlier = sum(r["review_file"] not in written for r in carried.values())
    counts = {"pairs": len(pairs), "reasons_written": len(with_reason),
              "writer_files_without_reasons": _without_reasons(),
              "reasons_by_file_group": _count_by(_reason_groups().values()),
              "sample": len(sample), "sample_judgement_carried": carried_earlier,
              "sample_to_review": len(sample) - carried_earlier, "review_files_written": written}
    files.record_counts("reasons", counts)
    return counts


def _count_by(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


def _per_subject(questions: list[dict], drawn: list[dict], pairs: list[dict], sample: list[dict],
                 sample_removed: list[dict], removed: list[dict], kept: list[dict]) -> dict:
    subject = {q["qid"]: q["subject"] for q in questions}
    groups = _reason_groups()
    set_aside = files.read_json(paths.SET_ASIDE)["set_aside"] if paths.SET_ASIDE.exists() else []
    by_group = {g: [jid.rsplit(":", 1)[0] for jid, has in groups.items() if has == g] for g in sorted(set(groups.values()))}
    columns = {"pairs": [p["qid"] for p in drawn], "with_reason": [p["qid"] for p in pairs],
               **{f"reasons_from_{g}": qids for g, qids in by_group.items()},
               "set_aside": [e["qid"] for e in set_aside],
               "reviewed": [j["job_id"].rsplit(":", 1)[0] for j in sample],
               "contradicted": [r["qid"] for r in sample_removed], "removed": [r["qid"] for r in removed],
               "kept": [p["qid"] for p in kept]}
    out = {s: dict.fromkeys(columns, 0) for s in sorted(set(subject.values()))}
    for column, qids in columns.items():
        for qid in qids:
            out[subject[qid]][column] += 1
    return out


def _finish(args) -> dict:
    questions = files.read_jsonl(paths.QUESTIONS)
    pairs = files.read_jsonl(paths.EXPANSION / "pairs_with_reasons.jsonl")
    sample = sample_jobs(questions, pairs)
    sample_removed, removed, review_counts = sample_outcome(review_jobs_of(questions, pairs), sample, _judged())
    files.write_json(paths.EXPANSION / "removed_pairs.json", {"removed": removed})
    kept = without_removed(pairs, removed)
    files.write_jsonl(paths.PAIRS, kept)
    style = {"reasons": len(pairs), "key_target": sum(p["key_target"] for p in pairs),
             "auroc": round(style_score([p["reason"] for p in pairs], [p["key_target"] for p in pairs],
                                        [p["qid"] for p in pairs]), 4)}
    drawn = files.read_jsonl(paths.EXPANSION / "pairs_drawn.jsonl")
    sample_files = files.read_json(paths.COUNTS).get("reasons", {}).get("review_files_written", {})
    first = [r for r in removed if r["review_file"] not in sample_files]
    counts = {"review": review_counts, "style": style, "pairs_kept": len(kept), "removed": len(removed),
              "removed_by_first_sample": len(first),
              "removed_by_first_sample_in_this_sample": sum(r["in_review_sample"] for r in first),
              "per_subject": _per_subject(questions, drawn, pairs, sample, sample_removed, removed, kept)}
    files.record_counts("checks", counts)
    return counts


def _readme(args) -> dict:
    paths.README.write_text(readme(files.read_json(paths.COUNTS)) + "\n", encoding="utf-8", newline="\n")
    return {"written": paths.README.name}


COMMANDS = {"parse": _parse, "reuse-curation": _reuse_curation, "keep": _keep, "pairs": _pairs,
            "check-reasons": _check_reasons, "split-unwritten": _split_unwritten, "set-aside": _set_aside,
            "review-jobs": _review_jobs, "finish": _finish,
            "readme": _readme}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("parse", help="split raw rows into questions for curation")
    p.add_argument("--raw-dir", required=True, help="folder holding raw_<subject>.jsonl")
    p.add_argument("subjects", nargs="+")
    k = sub.add_parser("keep", help="apply the curation and order the options")
    k.add_argument("subjects", nargs="+")
    u = sub.add_parser("split-unwritten", help="put the unwritten jobs of job files into smaller job files")
    u.add_argument("--size", type=int, required=True, help="the most jobs a file holds")
    u.add_argument("--stem", required=True, help="the new files are named <stem>_NNN.json")
    u.add_argument("--writer", required=True, choices=("opus", "sonnet"), help="the model that writes them")
    u.add_argument("sources", nargs="+", help="job file names whose unwritten jobs are split")
    a = sub.add_parser("set-aside", help="list the jobs of job files that still have no reason")
    a.add_argument("sources", nargs="+", help="job file names whose unwritten jobs are set aside")
    for name in ("reuse-curation", "pairs", "check-reasons", "review-jobs", "finish", "readme"):
        sub.add_parser(name)
    args = parser.parse_args(argv)
    result = COMMANDS[args.command](args)
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
