"""The unwritten jobs of writer files that were not finished, split into smaller job files.

Each subject's jobs stay together where they fit: a subject fills whole files of the
given size first, and what remains of it goes whole into the first file that has room,
taking the largest remainders first, or into a new file. Job ids and job contents do not
change.
"""
from check2.data.expansion import files, paths


def subject_of(job_id: str) -> str:
    """The subject in a job id of the form `expansion/<subject>/<n>:<target>`."""
    return job_id.split("/")[1]


def split_jobs(jobs: list[dict], size: int) -> list[list[dict]]:
    """`jobs` in files of at most `size`, each subject's jobs kept together where they fit.
    Subjects are taken in sorted order and each subject's jobs in their given order."""
    by_subject: dict[str, list[dict]] = {}
    for j in jobs:
        by_subject.setdefault(subject_of(j["job_id"]), []).append(j)
    out: list[list[dict]] = []
    rest = []
    for subject in sorted(by_subject):
        group = by_subject[subject]
        whole = len(group) // size * size
        out += [group[i:i + size] for i in range(0, whole, size)]
        if whole < len(group):
            rest.append(group[whole:])
    partial: list[list[dict]] = []
    for group in sorted(rest, key=lambda g: (-len(g), subject_of(g[0]["job_id"]))):
        room = next((f for f in partial if len(f) + len(group) <= size), None)
        if room is None:
            partial.append(list(group))
        else:
            room += group
    return out + partial


def write_split(jobs: list[dict], size: int, stem: str) -> dict[str, int]:
    """Write `jobs` as `<stem>_NNN.json` job files of at most `size`. No file of that stem
    may exist already, so no earlier job file is ever replaced."""
    taken = sorted(p.name for p in paths.JOBS.glob(f"{stem}_*.json"))
    if taken:
        raise ValueError(f"job files {stem}_*.json exist already: {taken[0]}")
    out = {}
    for n, chunk in enumerate(split_jobs(jobs, size), start=1):
        path = paths.JOBS / f"{stem}_{n:03d}.json"
        files.write_json(path, chunk)
        out[path.name] = len(chunk)
    return out
