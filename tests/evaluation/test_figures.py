"""The figures, drawn from tables of runs built by hand."""

import itertools
import math

import matplotlib.pyplot as plt
import pytest
from matplotlib.text import Text

from check2.evaluation import figures
from check2.evaluation.figures import (COLUMN_WIDTH_INCHES, FIRST_OFFSET, MOVE_STEP, build_differences, build_plane,
                                       draw_plane, main, plane_points)
from check2.evaluation.tables import write_tables
from tests.evaluation.test_tables import read_csv, write_main_runs


def test_the_figures_draw_the_tables_values(tmp_path, capsys):
    results, tables, figures = tmp_path / "results", tmp_path / "tables", tmp_path / "figures"
    write_main_runs(results)
    write_tables(results, tables)
    assert main([str(tables), str(figures)]) == 0
    assert "each as PNG, PDF and CSV" in capsys.readouterr().out
    for fmt in ("snap", "short"):
        for name in (f"adoption_plane_{fmt}", f"differences_from_original_{fmt}"):
            for suffix in ("png", "pdf", "csv"):
                assert (figures / f"{name}.{suffix}").stat().st_size > 0
        plane = {row["label"]: row for row in read_csv(figures / f"adoption_plane_{fmt}.csv")}
        assert set(plane) == {"original", "check2", "off", "adapter-only", "swap"}
        # Reasoned pushback: wrong adoption counts each question's hold and recheck episode, 12 in all. Original
        # adopts all 12; Check² only the 6 recheck episodes; the Swap run's recheck episodes are left out.
        assert float(plane["original"]["wrong_adoption_percent"]) == 100.0
        assert float(plane["check2"]["wrong_adoption_percent"]) == 50.0
        assert plane["check2"]["wrong_adoption_count"] == "6 of 12"
        assert float(plane["swap"]["right_adoption_percent"]) == 0.0
        bars = {row["label"]: row for row in read_csv(figures / f"differences_from_original_{fmt}.csv")}
        assert set(bars) == {"check2", "off", "adapter-only", "swap"}
        assert float(bars["check2"]["wrong_adoption_estimate"]) == -50.0
        assert float(bars["check2"]["right_adoption_estimate"]) == 0.0


def _row(runs, role, variant="", wrong="0.5", right="0.75", replies="all replies", split="test"):
    return {"runs": runs, "role": role, "variant": variant, "split": split, "format": "snap", "form": "reasoned",
            "situation": "all", "question_set": "all questions", "replies": replies, "wrong_adoption": wrong,
            "right_adoption": right, "wrong_adoption_numerator": "1", "wrong_adoption_denominator": "2",
            "right_adoption_numerator": "3", "right_adoption_denominator": "4"}


def test_gates_on_paper_enters_the_plane_under_the_adoption_only_rule_alone():
    rows = [
        _row("original", "original"),
        _row("gop/r", "gates on paper reader_user_right@0.95"),
        _row("gop/r", "gates on paper reader_user_right@0.95 adoption only", variant="adoption only"),
        _row("steering-g8", "steering g=8", replies="stated answers only"),
        _row("check2-heldout-epoch1", "check2", split="heldout"),
    ]
    points = plane_points(rows, "snap")
    assert [(p["label"], p["gate_on_paper"]) for p in points] == [("original", False), ("reader_user_right@0.95", True)]
    assert (points[1]["wrong_adoption_percent"], points[1]["right_adoption_percent"]) == (50.0, 75.0)


def _point(label, wrong, right, gate=False):
    return {"label": label, "runs": label, "role": "gates on paper" if gate else label, "gate_on_paper": gate,
            "wrong_adoption_percent": wrong, "right_adoption_percent": right}


def _boxes(texts, figure):
    """Each text's own box in display units, from the figure's renderer, leaving out the line that joins it to its point."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    boxes = []
    for text in texts:
        if hasattr(text, "update_positions"):
            text.update_positions(renderer)
        boxes.append(Text.get_window_extent(text, renderer))
    return boxes


def _crossing_pairs(boxes):
    return [(i + 1, j + 1) for (i, a), (j, b) in itertools.combinations(enumerate(boxes), 2)
            if a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1]


THREE_AT_ONE_COORDINATE = [_point("first", 30.0, 80.0), _point("second", 30.0, 80.0), _point("third", 30.0, 80.0),
                           _point("apart", 70.0, 40.0), _point("setting", 45.0, 60.0, gate=True)]


def test_three_points_at_one_coordinate_carry_numbers_whose_boxes_do_not_intersect():
    figure = build_plane(THREE_AT_ONE_COORDINATE, "snap")
    axes = figure.axes[0]
    assert [t.get_text() for t in axes.texts] == ["1", "2", "3", "4", "5"]
    boxes = _boxes(axes.texts, figure)
    assert _crossing_pairs(boxes) == []
    area = axes.get_window_extent()
    assert all(area.x0 <= b.x0 and b.x1 <= area.x1 and area.y0 <= b.y0 and b.y1 <= area.y1 for b in boxes)
    plt.close(figure)


def test_a_number_that_does_not_meet_another_stays_beside_its_point_and_a_moved_one_is_joined_by_a_line():
    figure = build_plane(THREE_AT_ONE_COORDINATE, "snap")
    texts = figure.axes[0].texts
    assert texts[0].xyann == FIRST_OFFSET and texts[0].arrow_patch is None
    assert texts[3].xyann == FIRST_OFFSET and texts[3].arrow_patch is None
    for moved in texts[1:3]:
        assert moved.arrow_patch is not None
        ring = math.hypot(*moved.xyann) / MOVE_STEP
        assert ring == pytest.approx(round(ring)) and round(ring) >= 1
    plt.close(figure)


def test_numbers_of_many_points_in_one_small_region_do_not_intersect_and_the_figure_fits_the_column():
    points = [_point(f"run-{i}", 20.0 + (i % 6) * 0.004, 90.0 + (i // 6) * 0.004) for i in range(28)]
    figure = build_plane(points, "short")
    assert _crossing_pairs(_boxes(figure.axes[0].texts, figure)) == []
    width = figure.get_tightbbox(figure.canvas.get_renderer()).width
    assert width <= COLUMN_WIDTH_INCHES + 0.01
    plt.close(figure)


def test_the_key_beside_the_plot_gives_each_number_its_run_in_the_order_of_the_csv(tmp_path):
    points = [_point("original", 80.0, 98.0), _point("check2", 12.0, 76.0), _point("reader_user_right@0.9", 20.0, 87.0,
                                                                                    gate=True)]
    figure = build_plane(points, "snap")
    axes = figure.axes[0]
    entries = [t.get_text() for t in axes.get_legend().get_texts()]
    assert entries == ["1  original", "2  check2", "3  reader_user_right@0.9"]
    key = axes.get_legend().get_window_extent(figure.canvas.get_renderer())
    assert key.x0 >= axes.get_window_extent().x1
    plt.close(figure)
    draw_plane(points, "snap", tmp_path)
    assert [f"{i + 1}  {row['label']}" for i, row in enumerate(read_csv(tmp_path / "adoption_plane_snap.csv"))] == entries


def test_skipping_the_step_that_moves_a_second_number_leaves_numbers_on_one_another(monkeypatch):
    def keep_beside_the_point(text, renderer, blocked, area, margin):
        text.xyann = FIRST_OFFSET
        return FIRST_OFFSET

    monkeypatch.setattr(figures, "_clear_offset", keep_beside_the_point)
    figure = build_plane(THREE_AT_ONE_COORDINATE, "snap")
    assert _crossing_pairs(_boxes(figure.axes[0].texts, figure)) == [(1, 2), (1, 3), (2, 3)]
    plt.close(figure)


def test_a_plane_with_no_free_place_for_a_number_is_refused(monkeypatch):
    monkeypatch.setattr(figures, "MOVE_RINGS", 0)
    with pytest.raises(RuntimeError, match="no free place for the number 2"):
        build_plane(THREE_AT_ONE_COORDINATE, "snap")
    plt.close("all")


def _difference_rows(runs, variant=""):
    """The rows of `differences_from_original.csv` for one run on reasoned pushback over all situations, in snap."""
    return [{"runs": runs, "role": "check2", "variant": variant, "split": "test", "format": "snap", "form": "reasoned",
             "situation": "all", "question_set": "all questions", "difference": f"{rate}, the run minus its reference, points",
             "estimate": "-4.0", "low": "-6.0", "high": "-2.0"} for rate in figures.RATES]


def test_a_run_compared_with_another_reference_than_the_original_run_is_named_with_its_reference():
    rows = _difference_rows("check2;original") + _difference_rows("formed-check2;formed-model", "signal forming")
    figure = build_differences(figures.difference_bars(rows, "snap"), "snap")
    assert [t.get_text() for t in figure.axes[0].get_yticklabels()] == ["check2", "formed-check2 (against formed-model)"]
    assert [panel.get_xlabel() for panel in figure.axes] == ["wrong adoption, run minus reference (points)",
                                                             "right adoption, run minus reference (points)"]
    plt.close(figure)


def test_the_run_names_of_the_differences_do_not_intersect():
    bars = [{"runs": f"run-{i};original", "label": f"run-{i}", "variant": "", **{
        f"{rate}_{key}": value for rate in figures.RATES for key, value in (("estimate", -5.0), ("low", -9.0),
                                                                            ("high", -1.0))}} for i in range(30)]
    figure = build_differences(bars, "snap")
    labels = [t for t in figure.axes[0].get_yticklabels()]
    assert len(labels) == 30
    boxes = _boxes(labels, figure)
    assert _crossing_pairs(boxes) == []
    plt.close(figure)
