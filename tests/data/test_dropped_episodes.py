"""The dropped-episode gate on hand-built reports: the 15% line, missing episodes and a false report."""

import copy

from check2.data.select_targets import build_report
from check2.gates.dropped_episodes import judge

SITUATIONS = ("user_right", "hold", "recheck")


def cell_episodes(n: int) -> list[dict]:
    """n episodes in every (situation, format) cell."""
    return [{"episode_id": f"q{i}:{sit}:{fmt}", "situation": sit, "format": fmt, "form": "named"}
            for fmt in ("snap", "short") for sit in SITUATIONS for i in range(n)]


def case(n: int, dropped_in_short_hold: int):
    """Every episode sampled, except the first `dropped_in_short_hold` short `hold` episodes."""
    episodes = cell_episodes(n)
    outcomes, targets = {}, []
    for e in episodes:
        cell = (e["situation"], e["format"])
        index = int(e["episode_id"].split(":")[0][1:])
        if cell == ("hold", "short") and index < dropped_in_short_hold:
            outcomes[e["episode_id"]] = "dropped"
            continue
        outcomes[e["episode_id"]] = "sampled"
        targets.append({"episode_id": e["episode_id"], "source": "sampled"})
    samples = {e["episode_id"]: [{"text": "B", "ended": True}] for e in episodes}
    return build_report(episodes, outcomes, samples, ["samples.jsonl.gz"]), episodes, targets


def test_green_when_every_short_cell_drops_less_than_fifteen_percent():
    findings, red, table = judge(*case(20, 2))
    assert findings == [] and red == []
    assert {(r["situation"], r["format"]): r["dropped_share"] for r in table}[("hold", "short")] == 0.1


def test_red_when_a_short_cell_drops_exactly_fifteen_percent():
    findings, red, _ = judge(*case(20, 3))
    assert red == [{"situation": "hold", "format": "short"}]
    assert len(findings) == 1 and "15%" in findings[0]


def test_red_when_an_episode_was_never_sampled():
    report, episodes, targets = case(20, 0)
    report["unsampled"] = 1
    findings, red, _ = judge(report, episodes, targets)
    assert red == [] and any("never sampled" in f for f in findings)


def test_red_when_the_report_disagrees_with_the_targets_it_describes():
    report, episodes, targets = case(20, 3)
    false = copy.deepcopy(report)
    cell = next(c for c in false["cells"] if (c["situation"], c["format"]) == ("hold", "short"))
    cell["dropped"], cell["sampled"] = 0, 20
    findings, red, _ = judge(false, episodes, targets)
    assert red == [] and any("the files give" in f for f in findings)


def test_red_when_a_snap_episode_is_dropped():
    episodes = cell_episodes(4)
    outcomes = {e["episode_id"]: "sampled" for e in episodes}
    outcomes["q0:user_right:snap"] = "dropped"
    targets = [{"episode_id": k, "source": "sampled"} for k, v in outcomes.items() if v == "sampled"]
    samples = {e["episode_id"]: [{"text": "B", "ended": True}] for e in episodes}
    findings, _, _ = judge(build_report(episodes, outcomes, samples, []), episodes, targets)
    assert any("snap user_right" in f and "dropped" in f for f in findings)
