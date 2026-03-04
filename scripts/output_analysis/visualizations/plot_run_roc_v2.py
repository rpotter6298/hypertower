#!/usr/bin/env python3
"""
Per-fold and mean OVR ROC plots for a single v2 HyperTower run.

Reads the saved .npy artifacts from a completed run and produces:
  - One per-fold ROC figure per class (all folds as individual lines)
  - One mean ± SD OVR ROC figure (all classes on the same axes)

Outputs are written to {mode_dir}/plots/.

v2 directory layout expected
-----------------------------
  {run_dir}/{eval_mode}/{tower_mode}/
    fold0/  y_true.npy  probs_fused.npy | probs_bilat.npy | probs_classic.npy | probs_fused_head.npy
    fold1/  ...

Usage examples
--------------
  # Ensemble binary run
  python scripts/output_analysis/visualizations/plot_run_roc_v2.py \\
      --run-dir analysis_data/v2_modes_full_40ep_5fold_roi_unet_perimage_refugebuild_holdout \\
      --eval-mode binary --tower-mode ensemble

  # Bilateral multiclass run
  python scripts/output_analysis/visualizations/plot_run_roc_v2.py \\
      --run-dir analysis_data/v2_modes_full_40ep_5fold_roi_unet_perimage_refugebuild_holdout \\
      --eval-mode multiclass --tower-mode bilateral

  # Fused head — explicitly select probs file
  python scripts/output_analysis/visualizations/plot_run_roc_v2.py \\
      --run-dir analysis_data/v2_ensemble_fused_binary_unet_40ep_5fold_v1 \\
      --eval-mode binary --tower-mode ensemble --probs probs_fused_head
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sk_auc


# ---------------------------------------------------------------------------
# Probs auto-detection
# ---------------------------------------------------------------------------

_PROBS_PRIORITY: dict[str, list[str]] = {
    "ensemble":  ["probs_fused_head", "probs_fused"],
    "bilateral": ["probs_bilat"],
    "single":    ["probs_classic"],
    "classic":   ["probs_classic"],
}
_ALL_PROBS = ["probs_fused_head", "probs_fused", "probs_bilat", "probs_classic"]


def detect_probs_stem(fold_dir: Path, tower_mode: str | None) -> str | None:
    priority = _PROBS_PRIORITY.get(tower_mode, _ALL_PROBS) if tower_mode else _ALL_PROBS
    for stem in priority:
        if (fold_dir / f"{stem}.npy").exists():
            return stem
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_fold_dirs(mode_dir: Path) -> list[Path]:
    return sorted(
        [p for p in mode_dir.iterdir() if p.is_dir() and p.name.startswith("fold")],
        key=lambda p: int(p.name.replace("fold", "")),
    )


def _find_y_true(fold_dir: Path) -> Path | None:
    """
    Return path to y_true.npy for this fold.  If missing from fold_dir
    (can happen for bilateral-only old runs), fall back to the same fold
    index under sibling tower-mode directories (ensemble → single → classic).
    Labels are shared across tower modes within the same fold.
    """
    local = fold_dir / "y_true.npy"
    if local.exists():
        return local
    fold_name = fold_dir.name           # e.g. "fold0"
    tower_dir = fold_dir.parent         # e.g. .../binary/bilateral
    eval_dir  = tower_dir.parent        # e.g. .../binary
    for fallback in ("ensemble", "single", "classic"):
        candidate = eval_dir / fallback / fold_name / "y_true.npy"
        if candidate.exists():
            return candidate
    return None


def load_fold(
    fold_dir: Path,
    probs_stem: str,
    eval_mode: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    y_path = _find_y_true(fold_dir)
    p_path = fold_dir / f"{probs_stem}.npy"
    if y_path is None or not p_path.exists():
        return None
    y = np.load(y_path)
    p = np.load(p_path)
    if eval_mode == "binary":
        mask = np.isin(y, [0, 1])
        y, p = y[mask], p[mask]
        if p.shape[1] > 2:
            p = p[:, :2]
    return y, p


# ---------------------------------------------------------------------------
# ROC helpers
# ---------------------------------------------------------------------------

def per_class_roc(y: np.ndarray, p: np.ndarray) -> dict[int, dict]:
    """OVR ROC for each class. Returns {k: {fpr, tpr, auc}}."""
    out: dict[int, dict] = {}
    for k in range(p.shape[1]):
        yb = (y == k).astype(np.uint8)
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        fpr, tpr, _ = roc_curve(yb, p[:, k])
        out[k] = {"fpr": fpr, "tpr": tpr, "auc": sk_auc(fpr, tpr)}
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_perfold(
    per_fold: list[tuple[int, dict]],
    out_dir: Path,
    class_names: list[str],
    probs_stem: str,
    eval_mode: str,
) -> None:
    """One figure per class: each fold as a separate line."""
    all_classes = sorted({k for _, curves in per_fold for k in curves})
    if eval_mode == "binary":
        all_classes = [k for k in all_classes if k == 1]
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in all_classes:
        cname = class_names[k] if k < len(class_names) else f"class_{k}"
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="grey")
        for fold_idx, curves in per_fold:
            if k not in curves:
                continue
            c = curves[k]
            auc_val = c["auc"]
            ax.plot(c["fpr"], c["tpr"], linewidth=1.5,
                    label=f"Fold {fold_idx}  (AUC {auc_val:.3f})")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"Per-fold ROC — {cname}  [{probs_stem}]")
        ax.legend(loc="lower right")
        fig.tight_layout()
        safe = cname.replace(" ", "_")
        fig.savefig(out_dir / f"roc_{probs_stem}_{safe}_perfold.png", dpi=160)
        plt.close(fig)
        print(f"  Saved per-fold ROC ({cname})")


def plot_mean_ovr(
    per_fold: list[tuple[int, dict]],
    out_dir: Path,
    class_names: list[str],
    probs_stem: str,
    eval_mode: str,
) -> None:
    """Mean ± SD OVR ROC — all classes on one figure."""
    all_classes = sorted({k for _, curves in per_fold for k in curves})
    if eval_mode == "binary":
        all_classes = [k for k in all_classes if k == 1]
    grid = np.linspace(0, 1, 501)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="grey")
    for k in all_classes:
        cname = class_names[k] if k < len(class_names) else f"class_{k}"
        tprs, aucs = [], []
        for _, curves in per_fold:
            if k not in curves:
                continue
            c = curves[k]
            tprs.append(np.interp(grid, c["fpr"], c["tpr"]))
            aucs.append(c["auc"])
        if not tprs:
            continue
        tprs_arr = np.vstack(tprs)
        mean = tprs_arr.mean(axis=0)
        std  = tprs_arr.std(axis=0)
        label = f"{cname}  (AUC {np.nanmean(aucs):.3f} ± {np.nanstd(aucs):.3f})"
        line, = ax.plot(grid, mean, linewidth=2, label=label)
        ax.fill_between(grid,
                        np.clip(mean - std, 0, 1),
                        np.clip(mean + std, 0, 1),
                        alpha=0.15, color=line.get_color())
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Mean OVR ROC (± 1 SD)  [{probs_stem}]")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path = out_dir / f"roc_{probs_stem}_mean_ovr.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  Saved mean OVR ROC: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-fold and mean OVR ROC plots for a single v2 run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--run-dir", required=True, type=Path,
                    help="Top-level run directory (e.g. analysis_data/v2_my_run).")
    ap.add_argument("--eval-mode", required=True, choices=["binary", "multiclass"])
    ap.add_argument("--tower-mode", required=True,
                    choices=["single", "classic", "ensemble", "bilateral"],
                    help="Tower mode subdirectory to read from.")
    ap.add_argument("--probs", default=None,
                    help="Probs file stem to use (e.g. probs_fused, probs_bilat, "
                         "probs_fused_head). Auto-detected if omitted.")
    ap.add_argument("--class-names", nargs="*",
                    default=["Healthy", "Glaucoma", "Suspect"])
    args = ap.parse_args()

    mode_dir = args.run_dir / args.eval_mode / args.tower_mode
    if not mode_dir.exists():
        raise SystemExit(f"Directory not found: {mode_dir}")

    fold_dirs = find_fold_dirs(mode_dir)
    if not fold_dirs:
        raise SystemExit(f"No fold subdirectories found in {mode_dir}")

    # Determine probs stem
    probs_stem = args.probs
    if probs_stem is None:
        for fd in fold_dirs:
            probs_stem = detect_probs_stem(fd, args.tower_mode)
            if probs_stem:
                break
    if probs_stem is None:
        raise SystemExit(f"Could not detect a probs file in {mode_dir}/fold*/")
    print(f"Using probs: {probs_stem}.npy")

    # Load all folds
    per_fold: list[tuple[int, dict]] = []
    for fd in fold_dirs:
        fold_idx = int(fd.name.replace("fold", ""))
        result = load_fold(fd, probs_stem, args.eval_mode)
        if result is None:
            print(f"  [skip] fold {fold_idx}: missing y_true or {probs_stem}.npy")
            continue
        y, p = result
        curves = per_class_roc(y, p)
        per_fold.append((fold_idx, curves))
        auc_str = "  ".join(
            f"class{k}={v['auc']:.3f}" for k, v in curves.items()
        )
        print(f"  fold {fold_idx}: {auc_str}")

    if not per_fold:
        raise SystemExit("No usable folds — nothing to plot.")

    out_dir = mode_dir / "plots"
    plot_perfold(per_fold, out_dir, args.class_names, probs_stem, args.eval_mode)
    plot_mean_ovr(per_fold, out_dir, args.class_names, probs_stem, args.eval_mode)
    print(f"\nPlots written to {out_dir}")


if __name__ == "__main__":
    main()
