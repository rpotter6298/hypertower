"""F8 quadrant attention chart (full-image scope).

Reads the per-cell quadrant fractions produced by
``v4.figures.F8_explainability`` (see CSV at
``output/F8_quadrant_fractions.csv``) and renders a single-panel grouped bar
chart of full-image Grad-CAM intensity by optic-disc quadrant. Within-disc
and peri-disc scopes are present in the CSV but are not plotted here; see
the CSV for the per-cell numbers if you need them.

Re-run:
    python -m v4.figures.F8_quadrant_plot
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CSV  = Path(__file__).parent / "output" / "F8_quadrant_fractions.csv"
OUT  = Path(__file__).parent / "output" / "F8_quadrant_attention.png"

QUAD_ORDER = ("ST", "SN", "IT", "IN")
QUAD_LABEL = {
    "ST": "Superotemporal",
    "SN": "Superonasal",
    "IT": "Inferotemporal",
    "IN": "Inferonasal",
}

CELL_ORDER = [
    ("Normal",   "correct"),
    ("Normal",   "incorrect"),
    ("Glaucoma", "correct"),
    ("Glaucoma", "incorrect"),
]
CELL_COLOR = {
    ("Normal",   "correct"):   "#3B6FB5",
    ("Normal",   "incorrect"): "#8BB0DA",
    ("Glaucoma", "correct"):   "#c44e52",
    ("Glaucoma", "incorrect"): "#e6a3a4",
}
CELL_LABEL = {
    ("Normal",   "correct"):   "Normal correct",
    ("Normal",   "incorrect"): "Normal incorrect",
    ("Glaucoma", "correct"):   "Glaucoma correct",
    ("Glaucoma", "incorrect"): "Glaucoma incorrect",
}


def render() -> None:
    if not CSV.exists():
        raise SystemExit(
            f"CSV {CSV} not found. Run `python -m v4.figures.F8_explainability "
            "--only-gradcam --run-gradcam` first."
        )
    df = pd.read_csv(CSV)
    df_full = df[df["scope"] == "full"]

    fig, ax = plt.subplots(figsize=(9.6, 6.0))

    n_quad = len(QUAD_ORDER)
    n_cell = len(CELL_ORDER)
    bar_w  = 0.18
    x      = np.arange(n_quad, dtype=float)

    for i, cell in enumerate(CELL_ORDER):
        means, sds = [], []
        n_eyes = None
        for q in QUAD_ORDER:
            row = df_full[
                (df_full["class"]    == cell[0])
                & (df_full["outcome"] == cell[1])
                & (df_full["quadrant"] == q)
            ]
            means.append(float(row["mean"].iloc[0]) if len(row) else float("nan"))
            sds.append(float(row["sd"].iloc[0]) if len(row) else float("nan"))
            if n_eyes is None and len(row):
                n_eyes = int(row["n_eyes"].iloc[0])

        offsets = (i - (n_cell - 1) / 2.0) * bar_w
        bars = ax.bar(
            x + offsets, means, width=bar_w,
            yerr=sds, capsize=2,
            color=CELL_COLOR[cell], alpha=0.92,
            edgecolor="black", linewidth=0.6,
            label=f"{CELL_LABEL[cell]} (n = {n_eyes})",
            error_kw=dict(ecolor="#444", linewidth=0.8, capthick=0.8),
        )
        for rect, m in zip(bars, means):
            if np.isnan(m):
                continue
            ax.text(rect.get_x() + rect.get_width() / 2, m + 0.005,
                    f"{m:.2f}", ha="center", va="bottom",
                    fontsize=8.5, color="#222")

    ax.set_xticks(x)
    ax.set_xticklabels([QUAD_LABEL[q] for q in QUAD_ORDER], fontsize=11)
    ax.set_ylabel("Mean fraction of full-image Grad-CAM intensity", fontsize=11)
    ax.set_title(
        "Image-tower Grad-CAM by optic-disc quadrant (OD-oriented; centroid-split)",
        fontsize=12.5, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0, max(ax.get_ylim()[1], 0.7))
    ax.axhline(0.25, color="#888", linestyle=":", linewidth=0.8, alpha=0.6, zorder=0)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=4,
              framealpha=0.94, fontsize=10)

    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}")


if __name__ == "__main__":
    render()
