#!/usr/bin/env python3
"""
Assemble the 10×5 training dynamics panel.

Layout: 3-row × 4-col gridspec with spanning

  Top 2×2 (each cell spans 2 cols):
    row 0, cols 0-1: binary early stopping sweep
    row 0, cols 2-3: multiclass early stopping sweep
    row 1, cols 0-1: binary cost of stopping early
    row 1, cols 2-3: multiclass cost of stopping early

  Bottom row of 4 (one col each):
    row 2, col 0: binary holdout AUC by epoch
    row 2, col 1: binary val−holdout gap (val-adjusted)
    row 2, col 2: multiclass holdout AUC by epoch
    row 2, col 3: multiclass val−holdout gap (val-adjusted)

Usage
-----
    python scripts/output_analysis/visualizations/build_10x5_training_panel.py \
        --run-dir analysis_data/pipeline_10x5 \
        --out     analysis_data/pipeline_10x5/aggregate/training_dynamics_panel.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# (row, col_start, col_end, rel_path, label)
# col_end is exclusive slice — use None for single cell
CELLS = [
    # ---- top 2×2: early stopping (each spans 2 cols) ----
    (0, 0, 2, "binary/ensemble/plots/early_stopping_sweep_fused.png",
              "Binary — Early stopping sweep"),
    (0, 2, 4, "multiclass/ensemble/plots/early_stopping_sweep_fused.png",
              "Multiclass — Early stopping sweep"),
    (1, 0, 2, "binary/ensemble/plots/early_stopping_sweep_fused_inverted.png",
              "Binary — Cost of stopping early"),
    (1, 2, 4, "multiclass/ensemble/plots/early_stopping_sweep_fused_inverted_tol0.002.png",
              "Multiclass — Cost of stopping early (CI tol=0.002)"),
    # ---- bottom row of 4: holdout epoch curves (single col each) ----
    (2, 0, 1, "binary/ensemble/plots/holdout_epoch_curves_fused.png",
              "Binary — Holdout AUC by epoch"),
    (2, 1, 2, "binary/ensemble/plots/holdout_epoch_curves_fused_delta_adj.png",
              "Binary — Val−Holdout gap (val-adjusted)"),
    (2, 2, 3, "multiclass/ensemble/plots/holdout_epoch_curves_fused.png",
              "Multiclass — Holdout AUC by epoch"),
    (2, 3, 4, "multiclass/ensemble/plots/holdout_epoch_curves_fused_delta_adj.png",
              "Multiclass — Val−Holdout gap (val-adjusted)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="analysis_data/pipeline_10x5")
    ap.add_argument("--out",     default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / "aggregate" / "training_dynamics_panel.png"

    fig = plt.figure(figsize=(22, 16))
    gs  = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.75], hspace=0.08, wspace=0.04)

    for row, col_start, col_end, rel, label in CELLS:
        ax  = fig.add_subplot(gs[row, col_start:col_end])
        img = mpimg.imread(str(run_dir / rel))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(label, fontsize=11, pad=5)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
