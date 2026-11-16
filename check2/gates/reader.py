"""Does the reader read more than the model's confidence and the reason's wording already give?

On the cross-fitted `train` outputs: for each baseline, fitted in the same five folds on the
same episodes, the paired difference in selection score, reader minus baseline, with its 95%
interval from 2,000 resamples of question ids. Green when every interval's lower end is above
zero, and no fold's fit held a question id or a normalised question text of the fold it
scored. Run as `python -m check2.gates.reader [--plant]`; `--plant` copies one question of
fold 0 into the questions its fit trained on, and the gate must then go red.
"""
import argparse
import sys
from collections.abc import Mapping, Sequence

from check2.data.pool import complying, load_pool
from check2.data.splits import load_splits
from check2.evaluation.intervals import paired_difference
from check2.gates.split_isolation import Result, check_folds
from check2.reader import files
from check2.reader.baselines import BASELINES, cross_fitted_scores
from check2.reader.scores import Scored, contrast_records, selection_score, user_right_scores
from check2.record import write_record


def same_episodes(a: Mapping[str, Sequence[Scored]], b: Mapping[str, Sequence[Scored]]) -> bool:
    """Whether two score tables hold the same episodes under the same question ids."""
    return ({q: [r.episode_id for r in v] for q, v in a.items()}
            == {q: [r.episode_id for r in v] for q, v in b.items()})


def fold_findings(folds: list[list[str]], fold_fits: list[dict], pool) -> Result:
    """The folds are disjoint in ids and texts, and no fit trained on a question of the fold it scored."""
    result = check_folds(folds, pool)
    if len(fold_fits) != len(folds):
        result.add(f"{len(fold_fits)} fold fits for {len(folds)} folds")
    for f in fold_fits:
        k = f["fold"]
        if not set(f["scored_questions"]) <= set(folds[k]):
            result.add(f"fold {k}'s fit scored questions outside fold {k}")
        seen = check_folds([f["scored_questions"], f["trained_questions"]], pool)
        for finding in seen.findings:
            result.add(f"fold {k}'s fit: {finding}")
    return result


def plant(fold_fits: list[dict]) -> str:
    """Copy fold 0's first scored question into the questions its fit trained on."""
    qid = fold_fits[0]["scored_questions"][0]
    fold_fits[0]["trained_questions"] = sorted(fold_fits[0]["trained_questions"] + [qid])
    return f"{qid} into the training questions of fold 0's fit"


def compare(reader_by_qid: Mapping[str, Sequence[Scored]],
            baseline_by_qid: Mapping[str, Mapping[str, Sequence[Scored]]]) -> dict:
    """Each baseline's selection score and the paired difference, reader minus baseline."""
    out = {}
    for name, table in baseline_by_qid.items():
        if not same_episodes(reader_by_qid, table):
            raise ValueError(f"the reader and baseline {name} are not scored on the same episodes")
        d = paired_difference(selection_score, reader_by_qid, table)
        out[name] = {"baseline_score": selection_score([r for v in table.values() for r in v]),
                     "difference": d.estimate, "interval": [d.low, d.high], "skipped": d.skipped,
                     "above_zero": d.low is not None and d.low > 0}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plant", action="store_true",
                        help="copy a scored question into its fold fit's training questions; the gate must fail")
    args = parser.parse_args(argv)
    fold_fits = files.read_json(files.FOLD_FITS)
    planted = plant(fold_fits) if args.plant else None
    values, green = evaluate(
        train=files.load_split("train"), p_train=files.read_p(files.P_TRAIN), folds=load_splits().folds,
        fold_fits=fold_fits, pool=complying(load_pool()),
        first_turn={r["qid"]: r["probabilities"] for r in files.read_jsonl(files.FIRST_TURN)})
    verdict = "green" if green else "red"
    record = write_record("gate", "reader", values, verdict=verdict, extra={"planted": planted})
    print(f"reader selection score on cross-fitted train: {values['reader_score']} "
          f"({values['contrast_questions']} questions, {values['contrast_episodes']} episodes)")
    for name, c in values["baselines"].items():
        print(f"  {name}: baseline {c['baseline_score']}, reader minus baseline {c['difference']} "
              f"interval {c['interval']}")
    if planted:
        print(f"planted: {planted}")
    for finding in values["fold_findings"]:
        print(f"FINDING {finding}")
    print(f"verdict: {verdict}  record: {record}")
    return 0 if green else 1


def evaluate(train: list[dict], p_train: Mapping[str, Sequence[float]], folds: list[list[str]],
             fold_fits: list[dict], pool, first_turn) -> tuple[dict, bool]:
    """The gate's values and whether it is green."""
    findings = fold_findings(folds, fold_fits, pool)
    missing = sorted({e["episode_id"] for e in train} - set(p_train))
    if missing:
        findings.add(f"{len(missing)} train episodes have no cross-fitted output, e.g. {missing[:3]}")
        return {"reader_score": None, "contrast_questions": None, "contrast_episodes": None, "baselines": {},
                "fold_findings": findings.findings, "fails_by_less_than_its_interval": []}, False
    reader_by_qid = contrast_records(train, user_right_scores(p_train))
    baselines = {name: contrast_records(train, cross_fitted_scores(name, train, folds, first_turn))
                 for name in BASELINES}
    comparisons = compare(reader_by_qid, baselines)
    green = findings.green and all(c["above_zero"] for c in comparisons.values())
    narrow_miss = [n for n, c in comparisons.items() if not c["above_zero"] and (c["difference"] or 0) > 0]
    return {"reader_score": selection_score([r for v in reader_by_qid.values() for r in v]),
            "contrast_questions": len(reader_by_qid),
            "contrast_episodes": sum(len(v) for v in reader_by_qid.values()),
            "baselines": comparisons, "fold_findings": findings.findings,
            "fails_by_less_than_its_interval": narrow_miss}, green


if __name__ == "__main__":
    sys.exit(main())
