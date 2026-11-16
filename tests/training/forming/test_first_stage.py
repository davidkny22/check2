"""The first stage's signal values on made-up outputs, and its behaviour command on made-up runs."""

from __future__ import annotations

import gzip
import json

import pytest

from check2 import record
from check2.training.forming import first_stage
from tests.training.forming.test_sharpness import _contrast_episodes, _p


def _train_and_heldout():
    train, heldout = _contrast_episodes("t"), _contrast_episodes("h")
    p = {"original_train": _p([0.6, 0.4, 0.3, 0.2], "t"), "original_heldout": _p([0.6, 0.4, 0.3, 0.2], "h"),
         "formed_train": _p([0.9, 0.1, 0.8, 0.2], "t"), "formed_heldout": _p([0.9, 0.1, 0.8, 0.2], "h")}
    # The class weights need every situation among the training episodes.
    train.append({"episode_id": "t9:bare", "qid": "t9", "form": "bare", "format": "snap", "a1": "A", "gold": "B",
                  "situation": "hold"})
    for name in ("original_train", "formed_train"):
        p[name]["t9:bare"] = [0.2, 0.6, 0.2]
    return train, heldout, p


def test_the_signal_values_hold_every_quantity_of_the_first_stage():
    train, heldout, p = _train_and_heldout()
    values = first_stage.signal_values(p["original_train"], p["original_heldout"], p["formed_train"],
                                       p["formed_heldout"], train, heldout)
    assert values["selection_score_formed_minus_original"]["estimate"] == pytest.approx(0.25)
    assert values["cross_entropy_formed_minus_original"]["estimate"] < 0
    assert values["gap_before_forming"]["estimate"] == pytest.approx(0.0)
    assert values["gap_after_forming"]["estimate"] == pytest.approx(0.0)
    assert values["episodes"] == {"train": 9, "heldout": 8}
    assert "sharpened" in values and values["reading"] in (first_stage.sharpness.SHARPENED,
                                                         first_stage.sharpness.NOT_SHARPENED)


def test_formed_outputs_without_original_ones_are_refused():
    train, heldout, p = _train_and_heldout()
    del p["original_heldout"]["h0:snap"]
    with pytest.raises(ValueError, match="no original ones"):
        first_stage.signal_values(p["original_train"], p["original_heldout"], p["formed_train"],
                                  p["formed_heldout"], train, heldout)


def _run(folder, run_id, a2s):
    folder.mkdir(parents=True)
    rows = []
    for q, a2 in enumerate(a2s):
        for fmt in ("snap", "short"):
            rows.append({"episode_id": f"q{q}:reasoned:{fmt}", "qid": f"q{q}", "format": fmt, "form": "reasoned",
                         "situation": "hold", "target": "C", "gold": "B", "a1": "A", "a2": a2, "part": None})
    meta = {"run_id": run_id, "status": "complete", "episodes_written": len(rows), "episodes_expected": len(rows),
            "gpu": "card", "order_sha256": "o", "versions": {"torch": "x"}}
    (folder / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    with gzip.open(folder / "episodes.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_the_behaviour_command_records_the_formed_run_against_the_original(tmp_path, monkeypatch):
    monkeypatch.setattr(record, "RESULTS", str(tmp_path / "results"))
    _run(tmp_path / "formed", "formed-model-heldout", ["C", "A", "A", "A"])
    _run(tmp_path / "original", "original-heldout", ["C", "C", "C", "A"])
    assert first_stage.main(["behaviour", "--formed", str(tmp_path / "formed"),
                             "--original", str(tmp_path / "original")]) == 0
    (path,) = (tmp_path / "results" / "measurements" / "formed_model_behaviour").glob("*.json")
    values = json.loads(path.read_text(encoding="utf-8"))["values"]
    assert values["formats"]["snap"]["reasoned"]["wrong_adoption"]["estimate"] == pytest.approx(25.0 - 75.0)
