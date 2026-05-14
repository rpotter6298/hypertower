#!/usr/bin/env python3
"""
Poster-facing model comparison figure in the style of head_comparison.png.

Compares:
  - MD Only
  - Image Only
  - Ensemble Fusion

using the same run sources as the poster ROC comparison script.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from v3.scripts.output_analysis.plot_poster_roc_comparison import (
    DEFAULT_CURVES,
    REPO_ROOT,
    build_curve_summary,
    load_curve_predictions,
)


DEFAULT_OUT = REPO_ROOT / "v3" / "figures" / "explainability" / "poster_model_comparison.png"
MODEL_ORDER = ["Clinical Data Only", "Image Only", "Ensemble Fusion"]

C_NORMAL = "#4c72b0"
C_GLAUCOMA = "#c44e52"
BG = "#e8e8e8"


def _select_specs():
    selected = []
    for label in MODEL_ORDER:
        matches = [spec for spec in DEFAULT_CURVES if spec.label == label]
        if not matches:
            raise ValueError(f"Could not find default poster curve for '{label}'")
        selected.append(matches[0])
    return selected


def make_model_comparison(out_path: Path) -> None:
    specs = _select_specs()
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, len(specs), figsize=(11, 4.5), sharey=True)
    if len(specs) == 1:
        axes = [axes]
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Model Comparison — P(Glaucoma) by True Class",
        fontsize=12,
        fontweight="bold",
    )

    for ax, spec in zip(axes, specs):
        df, score_col = load_curve_predictions(spec)
        summary = build_curve_summary(df, score_col)

        ax.set_facecolor(BG)
        data_by_class = [df.loc[df["y_true"] == cls, score_col].values for cls in [0, 1]]

        vp = ax.violinplot(
            data_by_class,
            positions=[0, 1],
            widths=0.6,
            showmedians=True,
            showextrema=False,
        )
        for body, color in zip(vp["bodies"], [C_NORMAL, C_GLAUCOMA]):
            body.set_facecolor(color)
            body.set_alpha(0.35)
        vp["cmedians"].set_color("#222")
        vp["cmedians"].set_linewidth(2)

        for cls, color in zip([0, 1], [C_NORMAL, C_GLAUCOMA]):
            vals = data_by_class[cls]
            jitter = rng.uniform(-0.12, 0.12, len(vals))
            ax.scatter(
                cls + jitter,
                vals,
                color=color,
                s=4,
                alpha=0.30,
                linewidths=0,
                zorder=3,
            )

        ax.axhline(0.5, color="#888", lw=1.0, ls="--", alpha=0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Normal", "Glaucoma"], fontsize=9)
        ax.set_title(spec.label, fontsize=10, fontweight="bold", color=spec.color)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("Predicted P(Glaucoma)", fontsize=10)

        ax.text(
            0.97,
            0.04,
            f"AUC = {summary['auc_mean']:.3f} ± {summary['auc_std']:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#333",
            bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=2),
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    make_model_comparison(args.out)


if __name__ == "__main__":
    main()
