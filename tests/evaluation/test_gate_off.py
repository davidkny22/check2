"""Gate Off on held-out runs built by hand, one case each way."""

import gzip
import json

import pytest

from check2.gates import off as gate
from tests.evaluation.hand_records import build, settings

QUESTIONS = [f"q{i}" for i in range(8)]


def heldout(snap_wrong, short_wrong):
    """Named and reasoned episodes in both formats; per question one wrong target and one right target.

    `snap_wrong` and `short_wrong` say, per question, whether the run adopted the wrong target.
    """
    records = []
    for fmt, wrong in (("snap", snap_wrong), ("short", short_wrong)):
        for form in ("named", "reasoned"):
            records += build({q: {"wrong": [bool(wrong[i])], "right": [True]} for i, q in enumerate(QUESTIONS)},
                             fmt=fmt, form=form)
    return records


MIXED = [1, 0, 1, 0, 1, 0, 1, 0]


def test_off_green_when_both_intervals_include_zero():
    # Off adopts exactly where Original does: the difference is 0 in every resample, in both formats.
    result = gate.evaluate(heldout(MIXED, MIXED), heldout(MIXED, MIXED))
    for fmt in ("snap", "short"):
        d = result["formats"][fmt]["wrong_adoption_off_minus_original"]
        assert (d["estimate"], d["low"], d["high"]) == (0.0, 0.0, 0.0)
        assert d["episodes"] == 16
        assert result["formats"][fmt]["verdict"] == "green"
    assert result["verdict"] == "green"


def test_off_red_when_one_format_moves_wrong_adoption():
    # In the snap format Off adopts every wrong target and Original none: 100 points in every
    # resample. The short format is unchanged and green; the gate is red.
    result = gate.evaluate(heldout([1] * 8, MIXED), heldout([0] * 8, MIXED))
    assert result["formats"]["snap"]["wrong_adoption_off_minus_original"]["low"] == 100.0
    assert result["formats"]["snap"]["verdict"] == "red"
    assert result["formats"]["short"]["verdict"] == "green"
    assert result["verdict"] == "red"


def test_off_pools_named_and_reasoned_episodes():
    # Off adopts the wrong target in reasoned episodes only: 8 of 16 pooled, against none.
    off = [r if r["form"] == "named" else {**r, "a2": r["target"]} for r in heldout([0] * 8, [0] * 8)]
    d = gate.evaluate(off, heldout([0] * 8, [0] * 8))["formats"]["snap"]["wrong_adoption_off_minus_original"]
    assert d["estimate"] == 50.0


OFF = settings(alpha=0.0, kind="consumer")
ORIGINAL = settings()


def write_run(folder, run_id, records, split="heldout", config=None):
    folder.mkdir(parents=True)
    meta = {"run_id": run_id, "kind": "run", "config": config, "split": split, "status": "complete",
            "episodes_expected": len(records), "episodes_written": len(records), "commit": "c", "order_sha256": "o"}
    (folder / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    with gzip.open(folder / "episodes.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.writelines(json.dumps(r) + "\n" for r in records)


def test_the_command_writes_its_record_through_write_record(tmp_path, monkeypatch):
    calls = []

    def fake_write_record(kind, name, values, verdict=None, config=None, extra=None):
        calls.append({"kind": kind, "name": name, "values": values, "verdict": verdict, "config": config})
        return tmp_path / "record.json"

    monkeypatch.setattr(gate, "write_record", fake_write_record)
    write_run(tmp_path / "off", "off-heldout", heldout([1] * 8, MIXED), config=OFF)
    write_run(tmp_path / "original", "original-heldout", heldout([0] * 8, MIXED), config=ORIGINAL)
    assert gate.main(["--run", str(tmp_path / "off"), "--original", str(tmp_path / "original")]) == 0
    (call,) = calls
    assert (call["kind"], call["name"], call["verdict"]) == ("gate", "off", "red")
    assert call["config"] == {"run": "off-heldout", "original": "original-heldout"}
    assert set(call["values"]) == {"snap", "short"}


def test_the_command_refuses_a_run_that_is_not_on_heldout(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "write_record", lambda *a, **k: pytest.fail("no record for a refused run"))
    write_run(tmp_path / "off", "off-test", heldout(MIXED, MIXED), split="test", config=OFF)
    write_run(tmp_path / "original", "original-heldout", heldout(MIXED, MIXED), config=ORIGINAL)
    with pytest.raises(ValueError):
        gate.main(["--run", str(tmp_path / "off"), "--original", str(tmp_path / "original")])


@pytest.mark.parametrize("given_as_off, given_as_original, refused", [
    (ORIGINAL, ORIGINAL, "this gate reads the off run"),
    (settings(alpha=1.0, kind="consumer"), ORIGINAL, "this gate reads the off run"),
    (OFF, OFF, "compares with the Original run"),
    (OFF, settings(alpha=0.0, kind="none", steering_strength=4), "compares with the Original run"),
])
def test_the_command_refuses_runs_that_are_not_an_off_run_and_its_original(tmp_path, monkeypatch, given_as_off,
                                                                          given_as_original, refused):
    monkeypatch.setattr(gate, "write_record", lambda *a, **k: pytest.fail("no record for a refused pair"))
    write_run(tmp_path / "off", "first-heldout", heldout(MIXED, MIXED), config=given_as_off)
    write_run(tmp_path / "original", "second-heldout", heldout(MIXED, MIXED), config=given_as_original)
    with pytest.raises(ValueError, match=refused):
        gate.main(["--run", str(tmp_path / "off"), "--original", str(tmp_path / "original")])


def test_the_command_takes_the_off_run_of_the_repeat_at_beta_four(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "write_record", lambda *a, **k: tmp_path / "record.json")
    write_run(tmp_path / "off", "off-beta4-heldout", heldout(MIXED, MIXED), config=settings(kind="consumer", beta=4.0))
    write_run(tmp_path / "original", "original-heldout", heldout(MIXED, MIXED), config=ORIGINAL)
    assert gate.main(["--run", str(tmp_path / "off"), "--original", str(tmp_path / "original")]) == 0
