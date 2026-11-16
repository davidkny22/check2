"""Agreement between a behaviour run and another run's final answers, on hand-built folders."""

import gzip
import json

from check2.behaviour.agreement import agreement


def _reply(item_id, fmt, text):
    return {"item_id": item_id, "qid": "q", "source": "pool", "kind": "reasoned", "fmt": fmt, "target": "A",
            "writer": "opus", "text": text, "ended": True, "refused": False, "detail": {}}


def _episode(episode_id, fmt, a2, scored_by="stated", form="reasoned"):
    return {"episode_id": episode_id, "form": form, "format": fmt, "a2": a2, "scored_by": scored_by}


def test_counts_on_a_hand_built_pair_of_runs(tmp_path):
    mine, other = tmp_path / "mine", tmp_path / "other"
    mine.mkdir()
    other.mkdir()
    replies = [_reply("e1", "snap", "A"), _reply("e2", "snap", "B"), _reply("e3", "short", "no letter"),
               _reply("e4", "short", "Final answer: (C)"), _reply("only-mine", "snap", "A")]
    (mine / "replies.jsonl").write_text("".join(json.dumps(r) + "\n" for r in replies), encoding="utf-8")
    episodes = [_episode("e1", "snap", "A"), _episode("e2", "snap", "C"), _episode("e3", "short", "D"),
                _episode("e4", "short", "C", scored_by="appended"), _episode("e5", "snap", "A"),
                _episode("e1-named", "snap", "A", form="named")]
    with gzip.open(other / "episodes.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write("".join(json.dumps(e) + "\n" for e in episodes))
    result = agreement(mine, other)
    assert result["snap"] == {"both_hold": 2, "agree": 1, "share": 0.5, "stated_in_this_run": 2,
                              "stated_in_both": 2, "agree_where_both_stated": 1}
    assert result["short"] == {"both_hold": 2, "agree": 1, "share": 0.5, "stated_in_this_run": 1,
                               "stated_in_both": 0, "agree_where_both_stated": 0}
    assert result["all"]["both_hold"] == 4 and result["all"]["agree"] == 2
