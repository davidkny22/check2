"""The outcome of every training episode and the counts of the report, on hand-built samples."""

from check2.data.pool import Question
from check2.data.sampling import write_jsonl
from check2.data.select_targets import build_report, samples_by_episode, select_all


def question(qid: str, a1: str, gold: str) -> Question:
    return Question(qid=qid, subject="s", question=f"Which {qid}?", choices=("w", "x", "y", "z"), gold=gold,
                    a1=a1, free_text=f"({a1})", complies=True, p1=(0.25, 0.25, 0.25, 0.25))


POOL = {"q1": question("q1", "C", "B"), "q2": question("q2", "B", "B")}


def ep(qid: str, fmt: str, target: str, form: str = "named") -> dict:
    q = POOL[qid]
    situation = "user_right" if target == q.gold else ("hold" if q.a1 == q.gold else "recheck")
    return {"episode_id": f"{qid}:{form}:{fmt}:{target}", "qid": qid, "form": form, "format": fmt,
            "situation": situation, "a1": q.a1, "gold": q.gold, "target": target}


def snap(*letters):
    return [{"text": x, "ended": True} for x in letters]


def short(*letters):
    return [{"text": f"Final answer: ({x})", "ended": True} for x in letters]


EPISODES = [ep("q1", "snap", "B"), ep("q1", "snap", "A"), ep("q1", "short", "B"), ep("q1", "short", "A"),
            ep("q2", "snap", "A"), ep("q2", "short", "A"), ep("q2", "short", "D", form="reasoned")]
SAMPLES = {
    "q1:named:snap:B": snap("A", "B", "B", "A"),         # user_right: sampled, index 1
    "q1:named:snap:A": snap("A", "A", "D", "A"),         # recheck: none qualifies, constructed (C)
    "q1:named:short:B": short("A", "A", "A", "A"),       # user_right: dropped
    "q1:named:short:A": short("C", "B", "A", "D"),       # recheck: gold at index 1
    "q2:named:snap:A": snap("A", "A", "A", "A"),         # hold: constructed (B)
    "q2:named:short:A": short("A", "A", "A", "B"),       # hold: sampled, index 3
}                                                        # q2 reasoned short: never sampled


def test_every_episode_has_exactly_one_outcome_and_the_right_target():
    targets, outcomes = select_all(EPISODES, SAMPLES, POOL)
    assert outcomes == {"q1:named:snap:B": "sampled", "q1:named:snap:A": "constructed",
                        "q1:named:short:B": "dropped", "q1:named:short:A": "sampled",
                        "q2:named:snap:A": "constructed", "q2:named:short:A": "sampled",
                        "q2:reasoned:short:D": "unsampled"}
    got = {t["episode_id"]: (t["reply"], t["source"], t["sample_index"]) for t in targets}
    assert got == {"q1:named:snap:B": ("B", "sampled", 1), "q1:named:snap:A": ("(C)", "constructed", None),
                   "q1:named:short:A": ("Final answer: (B)", "sampled", 1),
                   "q2:named:snap:A": ("(B)", "constructed", None),
                   "q2:named:short:A": ("Final answer: (B)", "sampled", 3)}


def test_the_report_counts_each_outcome_per_situation_and_format():
    targets, outcomes = select_all(EPISODES, SAMPLES, POOL)
    report = build_report(EPISODES, outcomes, SAMPLES, ["samples.jsonl.gz"])
    assert (report["episodes"], report["sampled"], report["constructed"], report["dropped"],
            report["unsampled"]) == (7, 3, 2, 1, 1)
    cells = {(c["situation"], c["format"]): c for c in report["cells"]}
    assert len(cells) == 6
    assert cells[("user_right", "short")]["dropped"] == 1
    assert cells[("user_right", "short")]["dropped_share"] == 1.0
    assert cells[("hold", "short")]["unsampled"] == 1 and cells[("hold", "short")]["sampled"] == 1
    assert cells[("recheck", "snap")]["constructed"] == 1
    forms = {(r["form"], r["situation"], r["format"]): r for r in report["by_form"]}
    assert forms[("named", "recheck", "snap")]["no_qualifying_share"] == 1.0
    assert forms[("named", "user_right", "snap")]["no_qualifying_share"] == 0.0
    assert report["replies"]["snap"] == {"replies": 12, "ended": 12, "ended_share": 1.0, "stated": 12,
                                         "stated_share": 1.0}


def test_resampled_replies_replace_the_first_samples_of_their_episodes(tmp_path):
    write_jsonl(tmp_path / "samples.jsonl.gz", [{"episode_id": "a", "samples": snap("A")},
                                                {"episode_id": "b", "samples": snap("B")}])
    write_jsonl(tmp_path / "samples_resampled.jsonl.gz", [{"episode_id": "b", "samples": snap("C", "D")}])
    samples, files = samples_by_episode(tmp_path)
    assert samples == {"a": snap("A"), "b": snap("C", "D")}
    assert files == ["samples.jsonl.gz", "samples_resampled.jsonl.gz"]
