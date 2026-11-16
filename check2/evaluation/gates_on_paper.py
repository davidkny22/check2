"""Gates on paper: three scores that could gate a reply, applied to the Original run's saved outputs.

    python -m check2.evaluation.gates_on_paper --run <Original run folder>

Scores, per episode: the first answer's probability at the first turn (an episode passes at
or below the threshold); the target's probability at the first turn and the reader's
`p[user_right]` (each passes at or above). For each score, form and format, a threshold is
fitted on `train` episodes, with the cross-fitted `p` for the reader, at two settings: the
tightest threshold at which 90%, or 95%, of `user_right` episodes pass. An episode that does
not pass has its final answer replaced by its first answer. The target-probability gate does
not apply to `bare`, which keeps the Original answer.

Each score and setting is also applied under a second rule, adoption only: an episode that
does not pass has its final answer replaced by its first answer only where the final answer
is the target, so that a gate blocks an adoption and leaves every other reply alone. A
`bare` episode has no target and is treated as under the first rule. The two rules share
their thresholds and give the same adoption rates; they differ in how many conversations
end on the answer key's option.

Nothing is generated: the result is a derived record that names the run it came from, with
the thresholds, the counts they were fitted on, and for each of the twelve gated settings
the episodes whose answer was replaced and the rates.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from check2.data import paths
from check2.data.paths import LETTERS
from check2.evaluation.derived_runs import ADOPTION_ONLY
from check2.evaluation.rates import rates
from check2.evaluation.tables import load_run, run_role
from check2.record import write_record

SCORES = ("first_answer_probability", "target_probability", "reader_user_right")
PASSES_AT_OR_BELOW = ("first_answer_probability",)
SETTINGS = (0.90, 0.95)
FIRST_TURN_FILE = paths.DATA / "reader" / "first_turn.jsonl"
P_TRAIN_FILE = paths.DATA / "reader" / "p_train.jsonl"
TRAIN_EPISODES = paths.EPISODES / "train.jsonl"
USER_RIGHT = 0  # p is ordered (user_right, hold, recheck)


def _read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def applies(score: str, form: str) -> bool:
    return not (score == "target_probability" and form == "bare")


def value(score: str, episode: Mapping[str, Any], first_turn: Mapping[str, Sequence[float]],
          p: Sequence[float]) -> float:
    if score == "first_answer_probability":
        return float(first_turn[episode["qid"]][LETTERS.index(episode["a1"])])
    if score == "target_probability":
        return float(first_turn[episode["qid"]][LETTERS.index(episode["target"])])
    if score == "reader_user_right":
        return float(p[USER_RIGHT])
    raise ValueError(f"unknown score {score!r}")


def passes(score: str, x: float, threshold: float) -> bool:
    return x <= threshold if score in PASSES_AT_OR_BELOW else x >= threshold


def threshold(score: str, values: Sequence[float], share: float) -> float:
    """The tightest threshold at which at least `share` of the values pass."""
    if not values:
        raise ValueError(f"no user_right training episode to fit the {score} threshold on")
    ordered = sorted(values, reverse=score not in PASSES_AT_OR_BELOW)
    return ordered[math.ceil(share * len(ordered)) - 1]


def fit(train: Sequence[Mapping[str, Any]], first_turn: Mapping[str, Sequence[float]],
        p_train: Mapping[str, Sequence[float]]) -> dict:
    """Per score, setting, form and format: the threshold and the counts it was fitted on."""
    cells: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for e in train:
        if e["situation"] == "user_right":
            cells.setdefault((e["form"], e["format"]), []).append(e)
    out: dict = {}
    for score in SCORES:
        for share in SETTINGS:
            for (form, fmt), episodes in sorted(cells.items()):
                if not applies(score, form):
                    continue
                values = [value(score, e, first_turn, p_train[e["episode_id"]] if score == "reader_user_right" else ())
                          for e in episodes]
                t = threshold(score, values, share)
                out.setdefault(score, {}).setdefault(str(share), {})[f"{form}/{fmt}"] = {
                    "threshold": t, "user_right_episodes": len(values),
                    "passing": sum(passes(score, v, t) for v in values)}
    return out


def blocked(record: Mapping[str, Any], adoption_only: bool) -> bool:
    """Whether an episode that does not pass has its answer replaced: always under the first rule; under
    adoption only, where the reply adopted the target, and always for `bare`, which has no target."""
    return not adoption_only or record["form"] == "bare" or record["a2"] == record["target"]


def apply(records: Sequence[Mapping[str, Any]], thresholds: Mapping, first_turn: Mapping[str, Sequence[float]],
          score: str, share: float, adoption_only: bool = False) -> tuple[list[dict], list[str]]:
    """The run's records with `a2` replaced by `a1` where the episode does not pass, under the first rule or
    under adoption only; and those episodes' ids."""
    cells = thresholds[score][str(share)]
    out, replaced = [], []
    for r in records:
        if not applies(score, r["form"]):
            out.append(dict(r))
            continue
        t = cells[f"{r['form']}/{r['format']}"]["threshold"]
        if passes(score, value(score, r, first_turn, r["p"]), t) or not blocked(r, adoption_only):
            out.append(dict(r))
        else:
            out.append({**r, "a2": r["a1"]})
            replaced.append(r["episode_id"])
    return out, replaced


def derive(meta: Mapping[str, Any], records: Sequence[Mapping[str, Any]], train: Sequence[Mapping[str, Any]],
           first_turn: Mapping[str, Sequence[float]], p_train: Mapping[str, Sequence[float]]) -> dict:
    if run_role(meta["config"]) != ("original", ""):
        raise ValueError(f"run {meta['run_id']} is not an Original run over the main episodes")
    thresholds = fit(train, first_turn, p_train)
    gated = {}
    for adoption_only in (False, True):
        for score in SCORES:
            for share in SETTINGS:
                changed, replaced = apply(records, thresholds, first_turn, score, share, adoption_only)
                name = f"{score}@{share} {ADOPTION_ONLY}" if adoption_only else f"{score}@{share}"
                gated[name] = {"replaced_episodes": replaced, "rates": rates(changed),
                               "rates_wrong_first": rates(changed, wrong_first_only=True)}
    return {"source_run": meta["run_id"], "thresholds": thresholds, "gated": gated}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m check2.evaluation.gates_on_paper",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, type=Path, help="the Original run's folder")
    args = parser.parse_args(argv)
    meta, records = load_run(args.run)
    first_turn = {r["qid"]: r["probabilities"] for r in _read_jsonl(FIRST_TURN_FILE)}
    p_train = {r["episode_id"]: r["p"] for r in _read_jsonl(P_TRAIN_FILE)}
    values = derive(meta, records, _read_jsonl(TRAIN_EPISODES), first_turn, p_train)
    path = write_record("run", f"{meta['run_id']}-gates-on-paper", values, config={"run": meta["run_id"]},
                        extra={"source": {k: meta.get(k) for k in ("run_id", "commit", "config_sha256",
                                                                    "order_sha256", "split", "gpu")}})
    for name, g in values["gated"].items():
        print(f"{name}: {len(g['replaced_episodes'])} answers replaced by the first answer")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
