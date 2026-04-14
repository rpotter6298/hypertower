"""
Phase 3 modality ablation — Image-only vs Clinical-only vs HyperTower (fused).

Pools all rep×fold predictions from the phase3/baseline run and plots
per-fold AUC for each modality as a box plot with jittered points.

Output: v3/figures/phase3_modality_ablation.png

Usage:
    python -m v3.scripts.output_analysis.plot_phase3_modality_ablation
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT    = Path(__file__).resolve().parents[3]
RESULTS_DIR  = REPO_ROOT / "v3" / "results" / "phase3" / "baseline"
FIGURES_DIR  = REPO_ROOT / "v3" / "figures"
OUT_PNG      = FIGURES_DIR / "phase3_modality_ablation.png"

C_BASELINE = "#dd8452"
C_OTHER    = "#4c72b0"
C_MEDIAN   = "#c44e52"
FSIZE      = 10

MODALITIES = [
    ("prob_img_c1",   "Image only",    C_OTHER),
    ("prob_md_c1",    "Clinical only", C_OTHER),
    ("prob_fused_c1", "HyperTower\n(fused)", C_BASELINE),
]


def load_fold_aucs() -> dict[str, list[float]]:
    aucs: dict[str, list[float]] = {col: [] for col, _, _ in MODALITIES}

    for rep_dir in sorted(RESULTS_DIR.glob("rep*")):
        fold_root = rep_dir / "binary" / "single"
        if not fold_root.exists():
            continue
        for fold_dir in sorted(fold_root.glob("fold[0-9]")):
            csv = fold_dir / "predictions_test.csv"
            if not csv.exists():
                continue
            df = pd.read_csv(csv)
            if df["y_true"].nunique() < 2:
                continue
            for col, _, _ in MODALITIES:
                if col in df.columns:
                    try:
                        aucs[col].append(roc_auc_score(df["y_true"], df[col]))
                    except Exception:
                        pass

    return aucs


def main():
    print("Loading fold AUCs ...")
    aucs = load_fold_aucs()

    n_folds = len(next(iter(aucs.values())))
    print(f"  {n_folds} folds found")
    for col, label, _ in MODALITIES:
        vals = aucs[col]
        print(f"  {label.replace(chr(10), ' '):<30}  "
              f"mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  n={len(vals)}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4.5))

    data_list = [np.array(aucs[col]) for col, _, _ in MODALITIES]
    colors     = [color for _, _, color in MODALITIES]
    labels     = [lbl   for _, lbl, _ in MODALITIES]
    x          = np.arange(len(MODALITIES))

    bp = ax.boxplot(
        data_list,
        vert=True,
        patch_artist=True,
        positions=x,
        widths=0.3,
        showfliers=True,
        flierprops=dict(marker="o", markersize=3, alpha=0.5),
        medianprops=dict(color=C_MEDIAN, linewidth=2),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xlim(-0.5, len(MODALITIES) - 0.5)
    tick_labels = [
        f"{lbl}\nAUC={np.mean(np.array(aucs[col])):.3f}"
        for col, lbl, _ in MODALITIES
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=FSIZE)
    ax.set_ylabel("AUC (ROC)", fontsize=FSIZE + 1)
    fig.suptitle(
        f"Phase 3 — Modality Ablation: Image / Clinical / Fused  ({n_folds} folds)",
        fontsize=FSIZE + 3, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
