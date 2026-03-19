#!/usr/bin/env python3
"""
Assemble a 2x2 fusion-head comparison panel from pipeline_nocrop.

Layout:
  [binary ensemble ROC]    [binary fusion-head explainability summary]
  [multiclass ensemble ROC][multiclass fusion-head explainability summary]

Usage
-----
    python scripts/output_analysis/visualizations/build_fusion_head_panel.py \
        --run-dir analysis_data/pipeline_nocrop \
        --out analysis_data/pipeline_nocrop/fusion_head_comparison_panel.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


ROC_CELLS = [
    # (row, col, rel_path, label)
    (0, 0, "binary/ensemble/plots/roc_probs_fused_mean_ovr.png",
           "Binary — Ensemble"),
    (0, 1, "binary/ensemble/plots/roc_probs_fused_head_mean_ovr.png",
           "Binary — Fusion Head"),
    (1, 0, "multiclass/ensemble/plots/roc_probs_fused_mean_ovr.png",
           "Multiclass — Ensemble"),
    (1, 1, "multiclass/ensemble/plots/roc_probs_fused_head_mean_ovr.png",
           "Multiclass — Fusion Head"),
]

EXPL_ROWS = [
    # (row_in_grid, rel_path, label)
    (2, "binary/ensemble/explainability_fusion_summary_val_fused_head.png",
        "Binary — Fusion Head events (val)"),
    (3, "multiclass/ensemble/explainability_fusion_summary_val_fused_head.png",
        "Multiclass — Fusion Head events (val)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="analysis_data/pipeline_nocrop")
    ap.add_argument("--out",     default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / "fusion_head_comparison_panel.png"

    # load all images
    roc_imgs  = {(r, c): (mpimg.imread(str(run_dir / rel)), lbl)
                 for r, c, rel, lbl in ROC_CELLS}
    expl_imgs = [(mpimg.imread(str(run_dir / rel)), lbl)
                 for _, rel, lbl in EXPL_ROWS]

    # 4-row grid: rows 0-1 are the 2×2 ROC square; rows 2-3 are full-width explainability
    fig = plt.figure(figsize=(14, 20))
    gs = fig.add_gridspec(
        4, 2,
        height_ratios=[1, 1, 0.6, 0.6],
        hspace=0.06,
        wspace=0.04,
    )

    # ROC cells (2×2)
    for row, col, _, _ in ROC_CELLS:
        ax = fig.add_subplot(gs[row, col])
        img, label = roc_imgs[(row, col)]
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(label, fontsize=11, pad=5)

    # Explainability rows (span both columns)
    for grid_row, (img, label) in zip([2, 3], expl_imgs):
        ax = fig.add_subplot(gs[grid_row, :])
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(label, fontsize=11, pad=5)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
