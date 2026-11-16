"""Draw the report's main figure: where each condition lands on reasoned pushback, in each reply format.

usage: python -I make_adoption_plane.py

Reads adoption_plane_snap.csv and adoption_plane_short.csv beside this file and writes check2_adoption.png
and check2_adoption.pdf. Every coordinate is looked up by run name; none is typed here. One marker per
condition, the same in both panels, and one legend under them. Where two conditions fall on the same spot
the hollow marker is drawn over the filled one, so both can be seen; the original model's ring, the
largest marker, is drawn under all the others. A condition whose point would only
repeat another's in a panel is left out of that panel and the caption says so.
"""
import csv
import io
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
GREY = "#7a7a7a"
# legend text, run name, marker style, panels it is drawn in
CONDITIONS = [
    ("Original model", "original", dict(marker="o", ms=11, mfc="white", mec="black", mew=2.2), "LR"),
    ("System sentence", "system-prompt", dict(marker="o", ms=6.5, mfc=GREY, mec=GREY, mew=0), "LR"),
    ("Steering ×2", "steering-g2", dict(marker="^", ms=7, mfc=GREY, mec=GREY, mew=0), "LR"),
    ("Steering ×8", "steering-g8", dict(marker="v", ms=7, mfc=GREY, mec=GREY, mew=0), "LR"),
    ("Adapter only", "adapter-only", dict(marker="D", ms=9.5, mfc="none", mec="black", mew=1.6), "LR"),
    ("Check²", "check2", dict(marker="o", ms=6.5, mfc="black", mec="black", mew=0), "LR"),
    ("Check², second seed", "check2-seed1", dict(marker="o", ms=6.5, mfc="white", mec="black", mew=1.6), "LR"),
    ("Check² at half strength", "check2-alpha0.5", dict(marker="+", ms=9, mfc="black", mec="black", mew=2.0), "L"),
    ("Check² at double strength", "check2-alpha2", dict(marker="x", ms=7.5, mfc="black", mec="black", mew=2.0), "L"),
    ("Check², answer weighted†", "check2-answer-weighted", dict(marker="p", ms=8, mfc="black", mec="black", mew=0), "R"),
    ("True signal", "perfect-signal", dict(marker="*", ms=12, mfc="black", mec="black", mew=0), "LR"),
    ("Rule on the reader", "reader_user_right@0.95", dict(marker="s", ms=7, mfc="black", mec="black", mew=0), "LR"),
    ("Rule on confidence", "first_answer_probability@0.95", dict(marker="s", ms=7, mfc="white", mec=GREY, mew=1.8), "LR"),
]


def load(name):
    rows = {}
    with io.open(os.path.join(HERE, name), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows[r["label"]] = (float(r["wrong_adoption_percent"]), float(r["right_adoption_percent"]))
    return rows


def main():
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8.5, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.35))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.925, bottom=0.335, wspace=0.2)
    panels = [("L", "One-letter replies", load("adoption_plane_snap.csv")), ("R", "Short replies", load("adoption_plane_short.csv"))]
    check = []
    for ax, (key, title, rows) in zip(axes, panels):
        xs, ys = [], []
        for z, (text, run, style, where) in enumerate(CONDITIONS):
            if key not in where:
                continue
            if run not in rows:
                raise SystemExit("run %r is not in the data of panel %r; it holds: %s" % (run, title, sorted(rows)))
            x, y = rows[run]
            hollow = style.get("mfc") in ("white", "none") or style["marker"] in "+x"
            # The original model's ring is the largest marker: it goes under everything, so that a
            # smaller marker that falls inside it stays visible.
            order = 2 if run == "original" else (5 if hollow else 3)
            ax.plot([x], [y], linestyle="none", zorder=order, clip_on=False, **style)
            xs.append(x)
            ys.append(y)
            check.append("%s\t%s\t%s\t%.2f\t%.2f" % (title, text, run, x, y))
        padx = 0.07 * (max(xs) - min(xs))
        pady = 0.09 * (max(ys) - min(ys))
        ax.set_xlim(min(xs) - padx, max(xs) + padx)
        ax.set_ylim(min(ys) - pady, max(ys) + pady)
        ax.set_title(title, fontsize=9.5, pad=4)
        ax.set_xlabel("Wrong answers adopted (%)   ← better")
        ax.grid(True, color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Right answers adopted (%)   better →")
    handles = [Line2D([], [], linestyle="none", **style) for _, _, style, _ in CONDITIONS]
    fig.legend(handles, [t for t, _, _, _ in CONDITIONS], loc="lower center", ncol=4, frameon=False, fontsize=7.8,
               handletextpad=0.3, columnspacing=1.0, labelspacing=0.45, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(os.path.join(HERE, "check2_adoption.png"), dpi=300)
    fig.savefig(os.path.join(HERE, "check2_adoption.pdf"), metadata={"CreationDate": None, "Producer": None})
    io.open(os.path.join(HERE, "check2_adoption_points.txt"), "w", encoding="utf-8", newline="\n").write("\n".join(check) + "\n")
    print("drawn", len(check), "points")


if __name__ == "__main__":
    main()
