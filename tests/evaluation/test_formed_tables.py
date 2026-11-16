"""Signal forming's runs in the tables, and gate Off on the formed stage, from run folders built by hand."""

import pytest

from check2.evaluation.tables import main
from check2.gates.off import require_runs
from check2.training.forming.excess import HELPS
from tests.evaluation.hand_records import settings
from tests.evaluation.test_tables import episodes, read_csv, write_run


def formed(cfg, forming):
    out = {"check2": {**cfg["check2"], "signal_forming": forming}}
    return out


ORIGINAL = settings()
CHECK2 = settings(alpha=1.0, kind="consumer")
FORMED_MODEL = formed(settings(), "formed_model_only")
SECOND_STAGE = formed(settings(alpha=1.0, kind="consumer"), "formed_with_channel")
FORMED_OFF = formed(settings(alpha=0.0, kind="consumer"), "formed_with_channel")


def write_formed_runs(results):
    """Check² gains nothing over Original; the second stage drops wrong adoption by 100 points from the formed model."""
    write_run(results, "original", ORIGINAL, episodes([1] * 6, [1] * 6))
    write_run(results, "check2", CHECK2, episodes([1] * 6, [1] * 6))
    write_run(results, "formed-model", FORMED_MODEL, episodes([1] * 6, [1] * 6))
    write_run(results, "formed-check2", SECOND_STAGE, episodes([0] * 6, [1] * 6))


def _differences(rows, runs):
    return [r for r in rows if r["runs"] == runs]


def test_each_formed_run_is_compared_with_its_own_reference_and_the_excess_stands_beside_them(tmp_path):
    results, output = tmp_path / "results", tmp_path / "tables"
    write_formed_runs(results)
    assert main([str(results), str(output)]) == 0
    differences = read_csv(output / "differences_from_original.csv")

    second_stage = _differences(differences, "formed-check2;formed-model")
    assert second_stage and all(" check2 minus formed model, " in r["difference"] for r in second_stage)
    hold = next(r for r in second_stage if r["question_set"] == "all questions" and r["format"] == "snap"
                and r["form"] == "reasoned" and r["situation"] == "hold" and r["difference"].startswith("wrong_adoption"))
    assert float(hold["estimate"]) == -100.0

    formed_model = _differences(differences, "formed-model;original")
    assert formed_model and all(" formed model minus original, " in r["difference"] for r in formed_model)
    assert all(float(r["estimate"]) == 0.0 for r in formed_model if r["estimate"])

    assert _differences(differences, "check2;original")
    assert not _differences(differences, "formed-check2;original")
    assert not [r for r in differences if r["runs"].startswith("original;")]

    excess_rows = [r for r in read_csv(output / "readings.csv") if r["reading"] == "excess of a sharper signal"]
    assert {r["format"] for r in excess_rows} == {"snap", "short"}
    assert all(r["runs"] == "formed-check2;formed-model;check2;original" for r in excess_rows)
    assert all(r["outcome"] == HELPS for r in excess_rows)
    wrong = [r for r in excess_rows if r["quantity"] == "wrong_adoption_excess"]
    # Wrong adoption counts each question's hold and recheck episodes; the recheck one is adopted in every run,
    # so the second stage's drop from the formed model is 50 points, and Check²'s from Original none.
    assert len(wrong) == 2 and all(float(r["estimate"]) == -50.0 for r in wrong)
    assert len([r for r in excess_rows if r["quantity"] == "right_adoption_excess"]) == 2


def _meta(run_id, cfg):
    return {"run_id": run_id, "config": cfg}


def test_gate_off_compares_the_formed_stage_s_off_run_with_the_formed_model():
    require_runs(_meta("formed-off", FORMED_OFF), _meta("formed-model", FORMED_MODEL), "off")
    with pytest.raises(ValueError):
        require_runs(_meta("formed-off", FORMED_OFF), _meta("original", ORIGINAL), "off")
    require_runs(_meta("off", settings(alpha=0.0, kind="consumer")), _meta("original", ORIGINAL), "off")
    with pytest.raises(ValueError):
        require_runs(_meta("off", settings(alpha=0.0, kind="consumer")), _meta("formed-model", FORMED_MODEL), "off")
