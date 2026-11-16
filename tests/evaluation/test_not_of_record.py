"""A run folder marked as not of record is not read by the tables, and is listed with its reason."""

import json

import pytest

from check2.evaluation.tables import main, write_tables
from tests.evaluation.faults import fails_under
from tests.evaluation.hand_records import settings
from tests.evaluation.test_tables import ORIGINAL, episodes, read_csv, write_run

REASON = "Made on another card to see whether it reproduces this one; no reported number uses it."
CARD = "NVIDIA A100-SXM4-80GB"


def mark(results, run_id, reason=REASON):
    (results / run_id / "not_of_record.json").write_text(json.dumps({"reason": reason}), encoding="utf-8")


def write_marked_steering(results):
    """An Original run and a steering run on test, the second marked as not of record, both on one card."""
    write_run(results, "original", ORIGINAL, episodes([1] * 6, [1] * 6), gpu=CARD)
    write_run(results, "steering-old", settings(steering_strength=2), episodes([0] * 6, [0] * 6), gpu=CARD)
    mark(results, "steering-old")


def check_marked_folder_left_out_and_listed(results, output):
    _, _, marked = write_tables(results, output)
    assert marked == [{"folder": "steering-old", "reason": REASON}]
    assert {row["runs"] for row in read_csv(output / "rates.csv")} == {"original"}
    assert not [row for row in read_csv(output / "differences_from_original.csv") if "steering-old" in row["runs"]]
    assert read_csv(output / "not_of_record.csv") == [{"folder": "steering-old", "reason": REASON}]


def test_a_marked_folder_is_left_out_and_listed_with_its_reason(tmp_path, capsys):
    results, output = tmp_path / "results", tmp_path / "tables"
    write_marked_steering(results)
    check_marked_folder_left_out_and_listed(results, output)
    assert main([str(results), str(tmp_path / "again")]) == 0
    assert f"left out, as not of record: steering-old ({REASON})" in capsys.readouterr().out


def test_reading_a_marked_folder_turns_the_check_red(tmp_path):
    results = tmp_path / "results"
    write_marked_steering(results)
    fails_under("marked-folder-still-read", lambda: check_marked_folder_left_out_and_listed(results, tmp_path / "t"))


def test_a_marked_folder_from_another_card_does_not_stop_the_tables(tmp_path):
    results, output = tmp_path / "results", tmp_path / "tables"
    write_run(results, "original", ORIGINAL, episodes([1] * 6, [1] * 6), gpu=CARD)
    write_run(results, "original-other-card", ORIGINAL, episodes([1] * 6, [1] * 6), gpu="NVIDIA A100-SXM4-40GB")
    mark(results, "original-other-card")
    assert main([str(results), str(output)]) == 0
    # Unmarked, the same folder is refused as a run from a second card.
    (results / "original-other-card" / "not_of_record.json").unlink()
    assert main([str(results), str(tmp_path / "refused")]) == 1


def test_a_marker_without_a_reason_is_refused(tmp_path):
    results = tmp_path / "results"
    write_marked_steering(results)
    mark(results, "steering-old", reason=" ")
    with pytest.raises(ValueError, match="gives no reason"):
        write_tables(results, tmp_path / "tables")
