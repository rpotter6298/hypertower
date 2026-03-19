#!/usr/bin/env python3
"""
Assemble the 10×5 aggregate comparison panel.

Layout (3 rows × 2 cols):
  col 0 = binary,     col 1 = multiclass

  row 0: mean ROC curve
  row 1: rep stability
  row 2: holdout stability

Usage
-----
    python scripts/output_analysis/visualizations/build_10x5_aggregate_panel.py \
        --run-dir analysis_data/pipeline_10x5 \
        --out     analysis_data/pipeline_10x5/aggregate/aggregate_panel.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# (row, col, rel_path, label)
CELLS = [
    (0, 0, "aggregate/binary_roc_mean.png",           "Binary — Mean ROC"),
    (0, 1, "aggregate/multiclass_roc_mean.png",       "Multiclass — Mean ROC"),
    (1, 0, "aggregate/binary_rep_stability.png",      "Binary — Rep stability"),
    (1, 1, "aggregate/multiclass_rep_stability.png",  "Multiclass — Rep stability"),
    (2, 0, "aggregate/binary_holdout_stability.png",  "Binary — Holdout stability"),
    (2, 1, "aggregate/multiclass_holdout_stability.png", "Multiclass — Holdout stability"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="analysis_data/pipeline_10x5")
    ap.add_argument("--out",     default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / "aggregate" / "aggregate_panel.png"

    fig = plt.figure(figsize=(16, 18))
    gs  = fig.add_gridspec(3, 2, hspace=0.06, wspace=0.04)

    for row, col, rel, label in CELLS:
        ax  = fig.add_subplot(gs[row, col])
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
