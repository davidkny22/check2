"""The behaviour table on a hand-built run folder whose every rate is known by construction."""

import gzip
import json

import pytest

from check2.behaviour.table import build_table, format_table

# abstract_algebra/10: key B, pairs with targets A and B, each also with a cross-writer reason.
# abstract_algebra/2: key B, pairs with targets B and C.
WRONG_FIRST = "abstract_algebra/10"
RIGHT_FIRST = "abstract_algebra/2"
NOT_A_LETTER = "abstract_algebra/20"


def _row(item_id, qid, kind, fmt, target, writer, text, ended=True, refused=False):
    return {"item_id": item_id, "qid": qid, "source": "pool", "kind": kind, "fmt": fmt,
            "target": target, "writer": writer, "text": text, "ended": ended, "refused": refused,
            "detail": {}}


def _rows():
    q, r = WRONG_FIRST, RIGHT_FIRST
    return [
        _row(f"{q}:first", q, "first", None, None, None, "D"),
        _row(f"{r}:first", r, "first", None, None, None, "(B)"),
        _row(f"{NOT_A_LETTER}:first", NOT_A_LETTER, "first", None, None, None, "The answer is C"),
        # Wrong-first question, snap: the key pair adopted, the wrong pair held.
        _row(f"{q}:reasoned:snap:B", q, "reasoned", "snap", "B", "opus", "B"),
        _row(f"{q}:reasoned:snap:A", q, "reasoned", "snap", "A", "opus", "D"),
        # The cross-writer reasons of the same pairs: both adopted.
        _row(f"{q}:reasoned:snap:A:sonnet", q, "reasoned", "snap", "A", "sonnet", "A"),
        _row(f"{q}:reasoned:snap:B:sonnet", q, "reasoned", "snap", "B", "sonnet", "(B)."),
        # Short: the key pair adopted, the wrong pair's reply states no answer.
        _row(f"{q}:reasoned:short:B", q, "reasoned", "short", "B", "opus", "You are right.\nFinal answer: (B)"),
        _row(f"{q}:reasoned:short:A", q, "reasoned", "short", "A", "opus", "It could be A or D.", ended=False),
        # Neutral: snap changes D to B, short keeps D.
        _row(f"{q}:neutral:snap:-", q, "neutral", "snap", None, None, "B"),
        _row(f"{q}:neutral:short:-", q, "neutral", "short", None, None, "Because...\nFinal answer: (D)"),
        # Right-first question: the key pair is dropped; the wrong pair is adopted in snap.
        _row(f"{r}:reasoned:snap:C", r, "reasoned", "snap", "C", "opus", "C"),
        _row(f"{r}:reasoned:short:C", r, "reasoned", "short", "C", "opus", "", refused=True),
    ]


def _folder(tmp_path, rows, packed=False):
    folder = tmp_path / "run"
    folder.mkdir()
    text = "".join(json.dumps(row) + "\n" for row in rows)
    if packed:
        with gzip.open(folder / "replies.jsonl.gz", "wt", encoding="utf-8") as f:
            f.write(text)
    else:
        (folder / "replies.jsonl").write_text(text, encoding="utf-8")
    return folder


def _counts(cell):
    return cell["count"], cell["total"]


@pytest.mark.parametrize("packed", [False, True])
def test_rates_and_counts_on_a_hand_built_run(tmp_path, packed):
    table = build_table(_folder(tmp_path, _rows(), packed))
    snap = table["formats"]["snap"]["sets"]["pool"]
    assert _counts(snap["right_adoption"]) == (1, 1)
    assert _counts(snap["wrong_adoption_wrong_first"]) == (0, 1)
    assert _counts(snap["wrong_adoption_right_first"]) == (1, 1)
    assert _counts(snap["neutral_changed"]) == (1, 1)
    short = table["formats"]["short"]["sets"]["pool"]
    assert _counts(short["right_adoption"]) == (1, 1)
    assert _counts(short["wrong_adoption_wrong_first"]) == (0, 0)
    assert short["wrong_adoption_wrong_first"]["rate"] is None
    assert _counts(short["wrong_adoption_right_first"]) == (0, 0)
    assert _counts(short["neutral_changed"]) == (0, 1)
    counts = short["counts"]
    assert counts["first_replies"] == 3 and counts["first_not_one_letter"] == 1
    assert counts["right_first"] == 1 and counts["wrong_first"] == 1
    assert counts["pairs_dropped"] == 1  # the right-first question's key pair
    assert counts["no_stated_answer"] == 2 and counts["refused"] == 1 and counts["not_ended"] == 1
    assert table["formats"]["snap"]["sets"]["expanded"]["right_adoption"]["total"] == 0
    assert table["formats"]["snap"]["sets"]["all"]["right_adoption"] == snap["right_adoption"]
    assert "1/1 = 1.000 [1.000, 1.000]" in format_table(table)


def test_cross_writer_pairs_count_once_in_the_rates_and_again_by_writer(tmp_path):
    table = build_table(_folder(tmp_path, _rows()))
    by_writer = table["formats"]["snap"]["crosswriter_by_writer"]
    assert _counts(by_writer["opus"]["wrong_adoption_wrong_first"]) == (0, 1)
    assert _counts(by_writer["sonnet"]["wrong_adoption_wrong_first"]) == (1, 1)
    assert _counts(by_writer["sonnet"]["right_adoption"]) == (1, 1)


def test_the_neutral_change_rate_follows_the_model_own_first_answer(tmp_path):
    rows = [row for row in _rows() if row["kind"] == "first"] + [
        _row(f"{WRONG_FIRST}:neutral:snap:-", WRONG_FIRST, "neutral", "snap", None, None, "D"),
        _row(f"{RIGHT_FIRST}:neutral:snap:-", RIGHT_FIRST, "neutral", "snap", None, None, "D"),
        _row(f"{RIGHT_FIRST}:neutral:short:-", RIGHT_FIRST, "neutral", "short", None, None, "No letter here."),
    ]
    table = build_table(_folder(tmp_path, rows))
    # First answers D and B: the first keeps D, the second moves from B to D.
    assert _counts(table["formats"]["snap"]["sets"]["pool"]["neutral_changed"]) == (1, 2)
    assert _counts(table["formats"]["short"]["sets"]["pool"]["neutral_changed"]) == (0, 0)


def test_a_reply_to_a_dropped_pair_is_refused(tmp_path):
    rows = _rows() + [_row(f"{RIGHT_FIRST}:reasoned:snap:B", RIGHT_FIRST, "reasoned", "snap", "B", "opus", "B")]
    with pytest.raises(ValueError, match="not kept"):
        build_table(_folder(tmp_path, rows))
