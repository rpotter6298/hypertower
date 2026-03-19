#!/usr/bin/env python3
"""
Assemble a metadata-explainability comparison panel.

Layout (2 rows × 4 cols):
  row 0 = binary,     row 1 = multiclass

  col 0: single md_importance
  col 1: ensemble md_importance
  col 2: nocrop ROC
  col 3: excl_phakic_axial ROC

Usage
-----
    python scripts/output_analysis/visualizations/build_md_explainability_panel.py \
        --nocrop-dir analysis_data/pipeline_nocrop \
        --excl-dir   analysis_data/pipeline_nocrop_excl_phakic_axial \
        --out        analysis_data/pipeline_nocrop/md_explainability_panel.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# (row, col, dir_key, rel_path, label)
# dir_key: "nocrop" or "excl"
CELLS = [
    # ---- binary row (row 0) ----
    (0, 0, "nocrop", "binary/single/explainability_md_importance_summary.png",
           "Binary — Single"),
    (0, 1, "nocrop", "binary/ensemble/explainability_md_importance_summary.png",
           "Binary — Ensemble"),
    (0, 2, "nocrop", "binary/ensemble/plots/roc_probs_fused_mean_ovr.png",
           "Binary — nocrop ROC"),
    (0, 3, "excl",   "binary/ensemble/plots/roc_probs_fused_mean_ovr.png",
           "Binary — excl phakic+axial ROC"),
    # ---- multiclass row (row 1) ----
    (1, 0, "nocrop", "multiclass/single/explainability_md_importance_summary.png",
           "Multiclass — Single"),
    (1, 1, "nocrop", "multiclass/ensemble/explainability_md_importance_summary.png",
           "Multiclass — Ensemble"),
    (1, 2, "nocrop", "multiclass/ensemble/plots/roc_probs_fused_mean_ovr.png",
           "Multiclass — nocrop ROC"),
    (1, 3, "excl",   "multiclass/ensemble/plots/roc_probs_fused_mean_ovr.png",
           "Multiclass — excl phakic+axial ROC"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nocrop-dir", default="analysis_data/pipeline_nocrop")
    ap.add_argument("--excl-dir",   default="analysis_data/pipeline_nocrop_excl_phakic_axial")
    ap.add_argument("--out",        default=None)
    args = ap.parse_args()

    nocrop_dir = Path(args.nocrop_dir)
    excl_dir   = Path(args.excl_dir)
    out = Path(args.out) if args.out else nocrop_dir / "md_explainability_panel.png"

    fig = plt.figure(figsize=(24, 12))
    gs  = fig.add_gridspec(
        2, 4,
        hspace=0.08,
        wspace=0.04,
    )

    dirs = {"nocrop": nocrop_dir, "excl": excl_dir}

    for row, col, dir_key, rel, label in CELLS:
        ax  = fig.add_subplot(gs[row, col])
        img = mpimg.imread(str(dirs[dir_key] / rel))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(label, fontsize=11, pad=5)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
