"""Checking a writer's reasons against its jobs, one reason at a time, and joining the
reasons that count back to their pairs.

A reason counts for its pair when its job id is one of the jobs of the file it answers,
that id is written once in the file, and its text keeps the writer brief's rules that can
be checked without judgement: 20 to 40 words, none of the words of confidence or doubt the
brief lists by name, one line, and no heading or bullet point. A reason counts whether or
not the rest of its file is whole. A job with no reason that counts is unwritten.
"""
import re

from check2.data.expansion.jobs import job_id
from check2.data.expansion.paths import REASON_WORDS

# The phrases the writer brief's fifth rule forbids by name.
REFUSED_PHRASES = ("I think", "maybe", "obviously", "clearly", "definitely", "in my view", "as I see it",
                   "everyone knows")
_PHRASE = re.compile(r"\b(" + "|".join(re.escape(p) for p in REFUSED_PHRASES) + r")\b", re.IGNORECASE)
# The part of the writer brief's second rule that needs no judgement: one line, not opened
# as a heading or a bullet point. Starred or underscored text is left alone, since a
# formula or a name in code can hold it.
_MARKDOWN = re.compile(r"\n|^\s*(#|[-*] )")


def text_fault(text: str) -> str | None:
    """Why a reason's text breaks a checkable rule of the writer brief, or None when it keeps them."""
    words = len(text.split())
    low, high = REASON_WORDS
    if not low <= words <= high:
        return f"{words} words"
    phrase = _PHRASE.search(text)
    if phrase:
        return f"uses '{phrase.group(0)}'"
    if _MARKDOWN.search(text):
        return "has a heading, a bullet point or more than one line"
    return None


def reason_verdicts(jobs: list[dict], written: dict) -> list[tuple[dict, str | None]]:
    """Each entry of a writer's file with why it does not count, or None when it counts."""
    want = {j["job_id"] for j in jobs}
    entries = written.get("reasons", [])
    ids = [r.get("job_id") for r in entries]
    out = []
    for r in entries:
        jid = r.get("job_id")
        if jid not in want:
            out.append((r, "job id not in this file"))
        elif ids.count(jid) > 1:
            out.append((r, "job id written twice"))
        else:
            out.append((r, text_fault(str(r.get("reason", "")))))
    return out


def counted_reasons(jobs: list[dict], written: dict) -> dict:
    """The writer's file with only the reasons that count."""
    return {**written, "reasons": [r for r, fault in reason_verdicts(jobs, written) if fault is None]}


def unwritten_jobs(jobs: list[dict], counted: set[str]) -> list[dict]:
    """The jobs, in their order, whose id has no counted reason."""
    return [j for j in jobs if j["job_id"] not in counted]


def reason_faults(jobs: list[dict], written: dict) -> list[str]:
    """Every way a writer's file falls short of its jobs: an id missing, unknown or written
    twice, or a reason that breaks a checkable rule. An empty list means the file is whole."""
    faults = []
    want = [j["job_id"] for j in jobs]
    got = [r["job_id"] for r in written.get("reasons", [])]
    if len(set(got)) != len(got):
        faults.append(f"{len(got) - len(set(got))} job ids written twice")
    missing, unknown = set(want) - set(got), set(got) - set(want)
    if missing:
        faults.append(f"{len(missing)} job ids missing")
    if unknown:
        faults.append(f"{len(unknown)} unknown job ids")
    for r in written.get("reasons", []):
        fault = text_fault(str(r.get("reason", "")))
        if fault:
            faults.append(f"{r['job_id']}: {fault}")
    return faults


def join_reasons(pairs: list[dict], written: list[dict]) -> list[dict]:
    """Each pair with its reason and writer, from writer files holding only reasons that
    count. A job with two reasons is refused."""
    by_job = {}
    for w in written:
        for r in w["reasons"]:
            if r["job_id"] in by_job:
                raise ValueError(f"job {r['job_id']} has two reasons")
            by_job[r["job_id"]] = (r["reason"], w["writer"])
    out = []
    for p in pairs:
        reason, writer = by_job.get(job_id(p), (None, None))
        out.append({**p, "reason": reason, "writer": writer})
    return out
