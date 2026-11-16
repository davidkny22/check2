"""The result figures, drawn from the tables the tables command writes.

    python -m check2.evaluation.figures <tables folder> <output folder>

Two figures per reply format, each written as PNG and PDF with the CSV of the values it
draws:

- `adoption_plane_<format>`: every `test` run, and every setting of Gates on paper under the
  adoption-only rule, as a point of wrong adoption (x) against right adoption (y) on reasoned
  pushback, over all replies and all questions. The reader's gate is drawn as a line through
  the Original run and its two settings. Each point carries the number of its row in the CSV beside the
  figure, and a key outside the plot area gives the run for each number. A number that would sit on
  another is moved off its point by a fixed step and joined to it by a thin line. Each axis spans the
  points with a margin, so that points close together are drawn apart.
- `differences_from_original_<format>`: each `test` run's paired difference from the Original
  run in wrong adoption and in right adoption on reasoned pushback, with its interval.

Adoption rates are in percent and differences in points. The marks are black and grey with
distinct shapes, so the figures read the same in greyscale.
"""

from __future__ import annotations

import csv
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.text import Text  # noqa: E402

from check2.evaluation.derived_runs import ADOPTION_ONLY, GATES_ON_PAPER_ROLE  # noqa: E402
from check2.evaluation.scoring import FORMATS  # noqa: E402

READER_GATE = f"{GATES_ON_PAPER_ROLE} reader_user_right@"
RATES = ("wrong_adoption", "right_adoption")

COLUMN_WIDTH_INCHES = 16 / 2.54          # the report's column width
LABEL_FONT_SIZE = 7.0                    # points
KEY_FONT_SIZE = 6.5
MARKER_AREA = 28.0                       # points squared
FIRST_OFFSET = (6.0, 5.0)                # points: a number's centre from its point, up and to the right
MOVE_STEP = 8.0                          # points: the fixed step a number is moved by
MOVE_DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1))
MOVE_RINGS = 40                          # most steps a number is moved in each direction
LIMIT_MARGIN = 0.08                      # of the span of the values, on each side of an axis
BOX_MARGIN = 1.0                         # points kept clear around every number and marker


def _read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _gated(row: Mapping[str, str]) -> bool:
    return row["role"].startswith(GATES_ON_PAPER_ROLE)


def _label(row: Mapping[str, str]) -> str:
    """The run's id, or for a setting of Gates on paper its score and setting."""
    if _gated(row):
        return row["role"][len(GATES_ON_PAPER_ROLE) + 1:].replace(f" {ADOPTION_ONLY}", "")
    return row["runs"]


def plane_points(rates_rows: Sequence[Mapping[str, str]], fmt: str) -> list[dict]:
    """The points of the adoption plane in one format, in percent; a run with no episode for either rate has none."""
    points = []
    for row in rates_rows:
        if (row["split"], row["format"], row["form"], row["situation"], row["question_set"], row.get("replies"))\
                != ("test", fmt, "reasoned", "all", "all questions", "all replies"):
            continue
        if _gated(row) and not row["variant"].endswith(ADOPTION_ONLY):
            continue
        if not row["wrong_adoption"] or not row["right_adoption"]:
            continue
        points.append({"label": _label(row), "runs": row["runs"], "role": row["role"], "variant": row["variant"],
                       "gate_on_paper": _gated(row),
                       "wrong_adoption_percent": 100 * float(row["wrong_adoption"]),
                       "right_adoption_percent": 100 * float(row["right_adoption"]),
                       "wrong_adoption_count": f"{row['wrong_adoption_numerator']} of {row['wrong_adoption_denominator']}",
                       "right_adoption_count": f"{row['right_adoption_numerator']} of {row['right_adoption_denominator']}"})
    return points


def difference_bars(difference_rows: Sequence[Mapping[str, str]], fmt: str) -> list[dict]:
    """Each `test` run's paired differences from the Original run in one format, in points."""
    bars: dict[str, dict] = {}
    for row in difference_rows:
        if (row["split"], row["format"], row["form"], row["situation"], row["question_set"])\
                != ("test", fmt, "reasoned", "all", "all questions") or _gated(row):
            continue
        rate_name = row["difference"].split(",")[0]
        if rate_name not in RATES:
            continue
        run, reference = row["runs"].split(";")
        label = run if reference == "original" else f"{run} (against {reference})"
        bar = bars.setdefault(row["runs"], {"runs": row["runs"], "label": label, "variant": row["variant"]})
        for key in ("estimate", "low", "high"):
            bar[f"{rate_name}_{key}"] = float(row[key])
    return [bars[k] for k in sorted(bars)]


def _write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    columns: list[str] = []
    for row in rows:
        columns += [c for c in row if c not in columns]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _save(figure, output: Path, name: str, rows: Sequence[Mapping[str, object]]) -> None:
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(figure)
    _write_csv(rows, output / f"{name}.csv")


def _marker(point: Mapping[str, object]) -> tuple[str, str]:
    """Marker and grey level of a point: a black circle for a run, a grey square for a setting of Gates on paper."""
    return ("s", "0.55") if point["gate_on_paper"] else ("o", "black")


def _offsets():
    """Where a number is tried, in points from its point: beside it, then ring by ring at the fixed step in each direction."""
    yield FIRST_OFFSET
    for ring in range(1, MOVE_RINGS + 1):
        for dx, dy in MOVE_DIRECTIONS:
            length = math.hypot(dx, dy)
            yield ring * MOVE_STEP * dx / length, ring * MOVE_STEP * dy / length


def text_extent(text: Text, renderer):
    """The box of the text alone, in display units, without the line that joins it to its point."""
    text.update_positions(renderer)
    return Text.get_window_extent(text, renderer)


def _box(text: Text, renderer, margin: float) -> tuple[float, float, float, float]:
    extent = text_extent(text, renderer)
    return extent.x0 - margin, extent.y0 - margin, extent.x1 + margin, extent.y1 + margin


def _overlap(a: Sequence[float], b: Sequence[float]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _clear_offset(text: Text, renderer, blocked: Sequence[Sequence[float]], area, margin: float) -> tuple[float, float]:
    """The first offset at which the number's box is inside the plot and meets no number or marker in `blocked`."""
    for offset in _offsets():
        text.xyann = offset
        box = _box(text, renderer, margin)
        inside = area.x0 <= box[0] and box[2] <= area.x1 and area.y0 <= box[1] and box[3] <= area.y1
        if inside and not any(_overlap(box, other) for other in blocked):
            return offset
    raise RuntimeError(f"no free place for the number {text.get_text()}")


def place_numbers(axes, positions: Sequence[tuple[float, float]]) -> list[Text]:
    """Write the numbers 1, 2, ... beside the points at `positions`, in that order, and return them.

    A number is written beside its point. Where that place meets a number already written or the marker of
    another point, it is moved off its point by the fixed step, ring by ring, to the first place that is
    clear, and joined to its point by a thin line. A marker drawn on the same spot as the point's own is
    the point's own. A number that finds no place raises an error.
    """
    figure = axes.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    scale = figure.dpi / 72
    margin = BOX_MARGIN * scale
    reach = MARKER_AREA ** 0.5 / 2 * scale + margin
    centres = [axes.transData.transform(position) for position in positions]
    markers = [(x - reach, y - reach, x + reach, y + reach) for x, y in centres]
    area = axes.get_window_extent(renderer)
    written: list[tuple[float, float, float, float]] = []
    numbers: list[Text] = []
    for index, position in enumerate(positions):
        style = dict(textcoords="offset points", ha="center", va="center", fontsize=LABEL_FONT_SIZE, zorder=3)
        text = axes.annotate(str(index + 1), position, xytext=FIRST_OFFSET, **style)
        others = [box for box, centre in zip(markers, centres) if math.dist(centre, centres[index]) > 0.5]
        offset = _clear_offset(text, renderer, written + others, area, margin)
        if offset != FIRST_OFFSET:
            text.remove()
            text = axes.annotate(str(index + 1), position, xytext=offset, **style,
                                 arrowprops=dict(arrowstyle="-", color="0.35", linewidth=0.5, shrinkA=0, shrinkB=3))
        written.append(_box(text, renderer, margin))
        numbers.append(text)
    return numbers


def _limits(values: Sequence[float]) -> tuple[float, float]:
    """An axis range that holds the values with a margin of LIMIT_MARGIN of their span, within -2 to 102 percent."""
    if not values:
        return -2.0, 102.0
    low, high = min(values), max(values)
    pad = LIMIT_MARGIN * max(high - low, 1.0)
    return max(low - pad, -2.0), min(high + pad, 102.0)


def build_plane(points: Sequence[Mapping[str, object]], fmt: str) -> plt.Figure:
    """The adoption plane of one format: the points with their numbers, and the key of numbers beside the plot."""
    key_rows = len(points)
    height = max(4.4, 0.145 * key_rows + 1.3)
    figure = plt.figure(figsize=(COLUMN_WIDTH_INCHES, height), dpi=100)
    axes = figure.add_axes((0.1, 0.1, 0.5, 0.8))
    reader = sorted((p for p in points if p["gate_on_paper"] and str(p["role"]).startswith(READER_GATE)),
                    key=lambda p: p["wrong_adoption_percent"])
    original = [p for p in points if p["runs"] == "original"]
    line = sorted(reader + original, key=lambda p: p["wrong_adoption_percent"])
    if len(line) > 1:
        axes.plot([p["wrong_adoption_percent"] for p in line], [p["right_adoption_percent"] for p in line],
                  linestyle="--", color="0.45", linewidth=1, zorder=1)
    for p in points:
        marker, colour = _marker(p)
        axes.scatter(p["wrong_adoption_percent"], p["right_adoption_percent"], marker=marker, color=colour,
                     s=MARKER_AREA, zorder=2)
    axes.set_xlabel("wrong adoption, reasoned pushback (%)")
    axes.set_ylabel("right adoption, reasoned pushback (%)")
    axes.set_xlim(*_limits([p["wrong_adoption_percent"] for p in points]))
    axes.set_ylim(*_limits([p["right_adoption_percent"] for p in points]))
    axes.set_title(f"{fmt} format", fontsize=10)
    axes.grid(color="0.9", linewidth=0.5)
    handles = [Line2D([], [], marker=_marker(p)[0], color=_marker(p)[1], linestyle="", markersize=MARKER_AREA ** 0.5)
               for p in points]
    key = axes.legend(handles, [f"{i + 1}  {p['label']}" for i, p in enumerate(points)], loc="upper left",
                      bbox_to_anchor=(1.02, 1.0), fontsize=KEY_FONT_SIZE, frameon=False, labelspacing=0.35,
                      handletextpad=0.4, borderaxespad=0, title="number  run", title_fontsize=KEY_FONT_SIZE,
                      alignment="left")
    figure.canvas.draw()
    key_width = key.get_window_extent(figure.canvas.get_renderer()).width / figure.dpi
    left, right, gap, bottom, top = 0.7, 0.05, 0.1, 0.55, 0.35
    axes.set_position((left / COLUMN_WIDTH_INCHES, bottom / height,
                       (COLUMN_WIDTH_INCHES - left - right - gap - key_width) / COLUMN_WIDTH_INCHES,
                       (height - bottom - top) / height))
    place_numbers(axes, [(p["wrong_adoption_percent"], p["right_adoption_percent"]) for p in points])
    return figure


def draw_plane(points: Sequence[Mapping[str, object]], fmt: str, output: Path) -> None:
    _save(build_plane(points, fmt), output, f"adoption_plane_{fmt}", points)


def build_differences(bars: Sequence[Mapping[str, object]], fmt: str) -> plt.Figure:
    """The differences of one format: a row per run, with a panel for each rate, the figure growing with the rows."""
    figure, panels = plt.subplots(1, 2, figsize=(9, 0.32 * max(len(bars), 4) + 1.2), sharey=True)
    positions = list(range(len(bars)))[::-1]
    for panel, rate_name, marker in zip(panels, RATES, ("o", "s")):
        estimates = [b[f"{rate_name}_estimate"] for b in bars]
        below = [b[f"{rate_name}_estimate"] - b[f"{rate_name}_low"] for b in bars]
        above = [b[f"{rate_name}_high"] - b[f"{rate_name}_estimate"] for b in bars]
        panel.errorbar(estimates, positions, xerr=[below, above], fmt=marker, color="black", ecolor="0.4",
                       markersize=4, capsize=2, linewidth=1)
        panel.axvline(0, color="0.6", linewidth=0.8)
        panel.set_xlabel(f"{rate_name.replace('_', ' ')}, run minus reference (points)")
        panel.grid(axis="x", color="0.9", linewidth=0.5)
    panels[0].set_yticks(positions)
    panels[0].set_yticklabels([str(b["label"]) for b in bars], fontsize=7)
    figure.suptitle(f"{fmt} format, reasoned pushback", fontsize=10)
    return figure


def draw_differences(bars: Sequence[Mapping[str, object]], fmt: str, output: Path) -> None:
    _save(build_differences(bars, fmt), output, f"differences_from_original_{fmt}", bars)


def write_figures(tables: Path, output: Path) -> list[str]:
    """Draw both figures in each format from the tables under `tables`; returns the names written."""
    rates_rows = _read(Path(tables) / "rates.csv")
    difference_rows = _read(Path(tables) / "differences_from_original.csv")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = []
    for fmt in FORMATS:
        draw_plane(plane_points(rates_rows, fmt), fmt, output)
        draw_differences(difference_bars(difference_rows, fmt), fmt, output)
        names += [f"adoption_plane_{fmt}", f"differences_from_original_{fmt}"]
    return names


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("usage: python -m check2.evaluation.figures <tables folder> <output folder>", file=sys.stderr)
        return 2
    names = write_figures(Path(args[0]), Path(args[1]))
    print("wrote " + ", ".join(names) + f" under {args[1]}, each as PNG, PDF and CSV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
