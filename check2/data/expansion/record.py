"""The expanded set's README, written from its counts file."""
from check2.data.expansion.paths import JOB_FILE_SIZE

_PARSE_COLUMNS = ("rows", "does_not_split", "dropped_pool_or_seed", "dropped_repeat")
_KEEP_COLUMNS = ("curated", "answer_equals_key", "kept_by_curator", "dropped_repeat_of_earlier_subject",
                 "kept", "options_kept_in_order")


def _table(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return lines


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" + ("" if n == 1 else "s")


def smaller_groups(smaller: dict) -> list[str]:
    """The stems of the smaller job files in the order they were written: the Opus files from
    the largest down, then the Sonnet files."""
    return sorted(smaller, key=lambda g: (smaller[g].get("writer") == "sonnet", -smaller[g]["size"], g))


def _reasons_lines(reasons: dict, smaller: dict, set_aside: dict) -> list[str]:
    groups = reasons.get("reasons_by_file_group", {})
    out = [f"- Pairs with a reason that counts: {reasons['reasons_written']} of {reasons['pairs']}. A reason "
           f"counts for its pair on its own terms, whether or not its writer's file is whole: its job id is "
           f"one of its file's jobs and is written once there, and its text keeps the writer brief's rules "
           f"that need no judgement (20 to 40 words, none of the phrases of confidence or doubt the brief "
           f"names, one line with no heading or bullet point)."]
    order = smaller_groups(smaller)
    opus = [g for g in order if smaller[g].get("writer") != "sonnet"]
    if opus:
        sources = smaller[opus[0]]["sources"]
        names = ", ".join(f"`{n}`" for n in sources)
        verb = "was" if len(sources) == 1 else "were"
        out.append(f"- {_count(len(sources), 'writer job file')} of {JOB_FILE_SIZE} {verb} stopped part-way by "
                   f"the model provider's safety classifier ({names}; the notices are in `stopped_writers.json`). The "
                   f"reasons their writers finished count; {groups.get('writer', 0)} reasons in all come from "
                   f"the files of {JOB_FILE_SIZE}. A stopped file was written again in smaller files by fresh "
                   f"writers, each saving its output after every ten reasons so that a stop keeps what it "
                   f"finished:")
        for g in opus:
            c = smaller[g]
            out.append(f"  - {c['unwritten']} unwritten jobs went to Opus writers at medium effort in "
                       f"{_count(len(c['files']), 'file')} of at most {c['size']} (`{g}_NNN.json`), which added "
                       f"{groups.get(g, 0)} reasons.")
    if set_aside:
        out.append(f"- {set_aside['jobs']} jobs still unwritten after the files of at most "
                   f"{smaller[opus[-1]]['size']} were set aside, and are listed with their question, subject and "
                   f"target in `set_aside.json`. The table below gives them per subject.")
    for g in order:
        c = smaller[g]
        if c.get("writer") == "sonnet":
            out.append(f"- The owner agreed a second writer model for the jobs set aside. They went once to "
                       f"Sonnet writers at medium effort in {_count(len(c['files']), 'file')} of at most {c['size']} "
                       f"(`{g}_NNN.json`), under the same brief, which wrote {groups.get(g, 0)} reasons. Those "
                       f"pairs carry `\"writer\": \"sonnet\"` in `pairs.jsonl`; every other pair carries "
                       f"`\"writer\": \"opus\"`.")
    out.append(f"- Pairs without a reason: {reasons['pairs'] - reasons['reasons_written']}.")
    missing = reasons.get("writer_files_without_reasons", {})
    if missing:
        out.append("- Writer job files that are not whole, with why; the reasons they hold that count are kept:")
        out += [f"  - `{name}`: {why}." for name, why in sorted(missing.items())]
    return out


def _subject_columns(counts: dict) -> list[tuple[str, str]]:
    """The per-subject table's columns, as (key in the per-subject counts, heading)."""
    smaller = counts.get("smaller_files", {})
    cols = [("pairs", "pairs"), ("reasons_from_writer", f"reason from the files of {JOB_FILE_SIZE}")]
    for g in smaller_groups(smaller):
        c = smaller[g]
        if c.get("writer") == "sonnet" and counts.get("set_aside"):
            cols.append(("set_aside", "set aside"))
        who = "Sonnet" if c.get("writer") == "sonnet" else "Opus"
        cols.append((f"reasons_from_{g}", f"added by {who} files of {c['size']}"))
    if counts.get("set_aside") and ("set_aside", "set aside") not in cols:
        cols.append(("set_aside", "set aside"))
    return cols


def readme(counts: dict) -> str:
    """The README text: what the set is, how each question and pair was checked, and the counts."""
    parse, keep = counts.get("parse", {}), counts.get("keep", {})
    pairs, reasons, checks = counts.get("pairs", {}), counts.get("reasons", {}), counts.get("checks", {})
    subjects = sorted(parse)
    out = [
        "# Expanded question set",
        "",
        "Four-option multiple-choice questions written by a generation service from the `train`",
        "questions of one subject per job, kept apart from the reviewed set. No training and no run",
        "reads this folder. Each kept question has two written reasons, one arguing for its key and",
        "one for another option.",
        "",
        "## How a question was kept",
        "",
        "- A generated row is usable when its prompt splits into one question and exactly four",
        "  distinct options and its key is one of A to D. A row whose question text, lower-cased",
        "  with all whitespace removed, matches a question of the reviewed pool, a seed question, or",
        "  an earlier usable row is dropped. Subjects are compared in sorted order.",
        "- A Sonnet curator answered every remaining question without its key. A question is kept",
        "  only where the curator's answer equals the key and the curator records no problem with",
        "  the question. The keys were therefore checked by one model.",
        "- The four options of a kept question are in an order drawn with `random.Random(2040)`",
        "  over the questions sorted by id, with the key moved with its option. A question with an",
        "  option that names other options keeps its order, and its `options_reordered` is false.",
        "- Every kept question has two pairs: its key, and one other option drawn with",
        "  `random.Random(2041)` from the options that are neither the key nor the model's complying",
        "  wrong first answer. A writer wrote one reason per pair without seeing the key: an Opus",
        "  writer, except for the pairs the record below names as written by Sonnet.",
        "- The reasons were checked by a sample: a soundness review of 200 pairs drawn with",
        "  `random.Random(2042)` from the pairs with a reason, sorted by question id and then",
        "  target. A sampled pair that an earlier review judged with the same reason text keeps",
        "  that judgement; the others were reviewed for this sample. A pair is contradicted when a",
        "  reason for the key is rated unsound or a reason for another option is rated sound;",
        "  contradicted pairs are removed and listed in `removed_pairs.json`.",
        "",
        "## Counts per subject",
        "",
    ]
    rows = []
    for s in subjects:
        p, k = parse[s], keep.get(s, {})
        rows.append([s] + [p[c] for c in _PARSE_COLUMNS] + [p["usable"]] + [k.get(c, "") for c in _KEEP_COLUMNS])
    total = ["total"] + [sum(parse[s][c] for s in subjects) for c in _PARSE_COLUMNS] + \
            [sum(parse[s]["usable"] for s in subjects)] + \
            [sum(keep.get(s, {}).get(c, 0) for s in subjects) for c in _KEEP_COLUMNS]
    out += _table(["subject", "rows returned", "do not split", "pool or seed", "repeat", "usable", "curated",
                   "curator agrees with key", "kept by curator", "repeat of an earlier subject", "kept",
                   "options kept in order"], rows + [total])
    out += ["", "`usable` counts rows that split cleanly with a letter key, before the pool, seed and",
            "repeat drops; `curated` counts the questions the curator answered.", ""]
    if pairs:
        given = "were generated" if pairs["first_answers"] == "used" else "were not available"
        out += ["## First answers, pairs and reasons", "",
                f"- Kept questions: {pairs['questions']}. The model's first answers {given}: "
                f"{pairs['complying']} comply, and {pairs['wrong_first']} of those are wrong.",
                f"- Pairs: {pairs['pairs']}, two per kept question."]
        if reasons:
            out += _reasons_lines(reasons, counts.get("smaller_files", {}), counts.get("set_aside", {}))
        out.append("")
    if checks:
        review, style = checks["review"], checks["style"]
        key, other = review["key_target"], review["other_target"]
        carried = reasons.get("sample_judgement_carried")
        out += ["## Checks on the reasons", "",
                f"- Style check: the cross-validated AUROC of a text-only classifier at telling reasons "
                f"for the key from the others, grouped by question, is {style['auroc']} over "
                f"{style['reasons']} reasons. It is recorded, not used to remove anything.",
                f"- Soundness review of {review['reviewed']} sampled pairs: {review['contradicted']} "
                f"contradicted, a share of {review['contradicted_share']:.3f}. Of the reasons for the key, "
                f"{key.get('sound', 0)} were rated sound, {key.get('unsound', 0)} unsound and "
                f"{key.get('unclear', 0)} unclear; of the reasons for another option, "
                f"{other.get('unsound', 0)} were rated unsound, {other.get('sound', 0)} sound and "
                f"{other.get('unclear', 0)} unclear."
                + (f" {carried} of the judgements were kept from earlier review samples, whose "
                   f"reviewer saw the same reason, and {reasons['sample_to_review']} were made for "
                   f"this sample." if carried is not None else "")]
        first = checks.get("removed_by_first_sample", 0)
        if first:
            both = checks.get("removed_by_first_sample_in_this_sample", 0)
            out.append(f"- Earlier review samples of 200, drawn with the same seed before the last reasons "
                       f"were written, found {first} pairs contradicted, and they stay removed. Of "
                       f"these, the number this sample also drew, with the same reason and the same "
                       f"judgement, is {both}.")
        removed = f"- Pairs removed: {checks['removed']}. " if "removed" in checks else "- "
        out += [removed + f"Pairs in `pairs.jsonl` after the contradicted pairs are removed: "
                f"{checks['pairs_kept']}.", ""]
        per_subject = checks.get("per_subject")
        if per_subject:
            cols = _subject_columns(counts)
            if "reasons_from_writer" not in next(iter(per_subject.values())):
                cols = [("pairs", "pairs"), ("with_reason", "with a reason")]
            rows = []
            for s, c in sorted(per_subject.items()):
                rows.append([s] + [c.get(k, 0) for k, _ in cols] + [c["pairs"] - c["with_reason"]]
                            + [c[k] for k in ("reviewed", "contradicted", "removed", "kept")])
            total = ["total"] + [sum(r[i] for r in rows) for i in range(1, len(rows[0]))]
            out += ["## Pairs per subject", ""]
            out += _table(["subject"] + [h for _, h in cols] + ["without a reason", "reviewed in the sample",
                                                                "contradicted in the sample", "removed",
                                                                "in `pairs.jsonl`"], rows + [total])
            out += ["", "`removed` counts every contradicted pair, from this sample and from earlier review "
                        "samples.", ""]
    return "\n".join(out)
