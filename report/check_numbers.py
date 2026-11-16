"""Check every number in the report's text against the files that hold the results.

usage: python -I report/check_numbers.py [path to the report's Markdown source]

A number in the report passes when the same value, at the precision the report shows it, can be read from one
of the source files: as written there, or as a figure there rounded to the report's digits, or as a count over
a total written there as a rate. Signs are ignored, because the report states drops where the tables state
signed differences. Numbers that are not results (years, section and reference numbers, layer indices and the
like) are listed in ALLOW with the reason. The script prints every number it could not find, with its line.
"""
import io
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = [
    "RESULTS.md", "report/appendix/all-runs.md", "report/tables/pass_bar.md", "report/tables/readings.md",
    "report/tables/snap_against_short.md", "report/tables/differences_between_runs.md",
    "report/tables/pass_bar_quantities.md", "report/tables/neutral_follow_up.md", "docs/SPEC.md",
    "data/expansion/README.md", "report/tables/rates.md",
    "docs/research/2026-11-15-7-related-work-quotations-research.md",
    "docs/research/2026-11-15-6-reference-list-research.md",
]
NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")


def numbers(text):
    return [m.group(0) for m in NUM.finditer(text)]


def value(tok):
    return float(tok.replace(",", ""))


def decimals(tok):
    return len(tok.split(".")[1]) if "." in tok else 0


def load_sources():
    vals, pairs = [], []
    for rel in SOURCES:
        p = os.path.join(REPO, rel)
        if not os.path.exists(p):
            print("missing source:", rel)
            continue
        text = io.open(p, encoding="utf-8").read().replace("\u2212", "-")
        vals.extend(value(t) for t in numbers(text))
        for a, b in re.findall(r"(\d[\d,]*) of (\d[\d,]*)", text) + re.findall(r"(\d[\d,]*)/(\d[\d,]*) =", text):
            pairs.append((value(a), value(b)))
    return vals, pairs


def found(tok, vals, pairs):
    v, d = value(tok), decimals(tok)
    tol = 0.5 * 10 ** (-d) + 1e-9
    for s in vals:
        if abs(s - v) <= tol and (d > 0 or s == v):
            return True
        if abs(s * 100 - v) <= tol and d <= 2 and s <= 1.0:      # a rate written as a share
            return True
    for a, b in pairs:
        if b and abs(100.0 * a / b - v) <= tol:
            return True
    return False


# Numbers that are not results, each with why it is allowed.
ALLOW_PATTERNS = [
    (r"\[\d+(, \d+)*\]", "a reference number"),
    (r"arXiv:\d+\.\d+", "an arXiv identifier"),
    (r"arxiv\.org/abs/\d+\.\d+", "an arXiv address"),
    (r"\(20\d\d\)", "a year of publication"),
    (r"(Section|Appendix|Table|Figure) \d+", "a pointer inside the report"),
    (r"^\d+\. ", "a list number"),
    (r"^#+ \d+\. ", "a section number"),
    (r"(EMNLP|ICLR) 20\d\d", "a venue's year"),
    (r"\d+ November 2026", "a date"),
    (r"at \d\d:\d\d", "a clock time from the record of decisions"),
    (r"\d\d:\d\d", "a clock time from the record of decisions"),
]
ALLOW_WORDS = {"95": "the level of every interval", "2,000": "the number of resamples", "0.5": "chance on the score, or half strength",
               "100": "the reply's word limit", "20": "the reason's word range", "40": "the reason's word range, or a card's memory",
               "80": "a card's memory", "96": "a card's memory", "4": "the model's size in the text, or a count spelled in a name",
               "3.5": "part of a model's name", "5.5": "part of a model's name", "2": "part of a name or a strength", "8": "a strength",
               "9": "part of a model's name", "7": "part of a model's name", "2.0": "part of a dataset's name", "6000": "part of a card's name",
               "3": "a layer index", "23": "a layer index", "31": "a layer index", "32": "the adapter's rank", "16": "a layer index, or the cap in tokens",
               "0": "zero", "1": "one", "10": "the pass bar's limit", "150": "the abstract's or the held-out set's count"}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "report", "report.md")
    vals, pairs = load_sources()
    missing, checked = [], 0
    for n, line in enumerate(io.open(path, encoding="utf-8").read().replace("\u2212", "-").replace("\u00a0", " ").split("\n"), 1):
        if line.startswith(("title:", "author:", "affiliation:")):
            continue
        work = line
        for pat, _ in ALLOW_PATTERNS:
            work = re.sub(pat, " ", work)
        for tok in numbers(work):
            if tok in ALLOW_WORDS:
                continue
            checked += 1
            if not found(tok, vals, pairs):
                missing.append((n, tok, line.strip()[:110]))
    print("numbers checked against the sources:", checked)
    print("not found:", len(missing))
    for n, tok, ctx in missing:
        print("  line %d: %s   | %s" % (n, tok, ctx))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
