#!/usr/bin/env python3
"""
Per-class ROC curves for v2 HyperTower runs.

Plots multiple runs as separate lines on the same axes — one figure per class
(multiclass) or one figure total (binary).

v2 directory layout
-------------------
  analysis_data/{run_name}/{eval_mode}/{tower_mode}/
    fold0/  y_true.npy  probs_fused.npy | probs_bilat.npy | probs_classic.npy | probs_fused_head.npy
    fold1/  ...

Usage examples
--------------
  # Compare UNet ensemble vs bilateral vs fused head (binary)
  python scripts/output_analysis/visualizations/aggregate_roc_perclass_all_models_v2.py \\
    --mode binary --tag unet_binary_comparison \\
    --runs \\
      analysis_data/v2_modes_full_40ep_5fold_roi_unet_perimage_refugebuild_holdout/binary/ensemble:"UNet Ensemble" \\
      analysis_data/v2_modes_full_40ep_5fold_roi_unet_perimage_refugebuild_holdout/binary/bilateral:"UNet Bilateral" \\
      analysis_data/v2_ensemble_fused_binary_unet_40ep_5fold_v1/binary/ensemble:"UNet Fused Head"

  Each --runs entry is  <path>:<label>  where <path> points directly to the
  {eval_mode}/{tower_mode} subdirectory and <label> is shown in the legend.

Outputs (written to --output-dir, default: analysis_data/roc_plots/)
  {tag}_binary_roc.png          (binary mode)
  {tag}_class{k}_roc.png        (multiclass mode, one file per class)
  {tag}_roc_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sk_auc


# ---------------------------------------------------------------------------
# Probs filename auto-detection priority per tower mode
# ---------------------------------------------------------------------------
_PROBS_PRIORITY: dict[str, list[str]] = {
    "ensemble":  ["probs_fused_head", "probs_fused"],
    "bilateral": ["probs_bilat"],
    "single":    ["probs_classic"],
    "classic":   ["probs_classic"],
}
_ALL_PROBS = ["probs_fused_head", "probs_fused", "probs_bilat", "probs_classic"]


def _detect_probs_stem(fold_dir: Path, tower_mode: str | None) -> str | None:
    priority = _PROBS_PRIORITY.get(tower_mode, _ALL_PROBS) if tower_mode else _ALL_PROBS
    for stem in priority:
        if (fold_dir / f"{stem}.npy").exists():
            return stem
    return None


# ---------------------------------------------------------------------------
# Fold discovery and loading
# ---------------------------------------------------------------------------

def find_fold_dirs(mode_dir: Path) -> list[Path]:
    return sorted(
        [p for p in mode_dir.iterdir() if p.is_dir() and p.name.startswith("fold")],
        key=lambda p: int(p.name.replace("fold", "")),
    )


def load_fold(fold_dir: Path, probs_stem: str) -> tuple[np.ndarray, np.ndarray] | None:
    y_path = fold_dir / "y_true.npy"
    p_path = fold_dir / f"{probs_stem}.npy"
    if not y_path.exists() or not p_path.exists():
        return None
    return np.load(y_path), np.load(p_path)


# ---------------------------------------------------------------------------
# Per-class ROC helpers
# ---------------------------------------------------------------------------

def per_class_roc(y: np.ndarray, p: np.ndarray) -> dict[int, tuple]:
    K = p.shape[1]
    out: dict[int, tuple] = {}
    for k in range(K):
        yb = (y == k).astype(np.uint8)
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        fpr, tpr, _ = roc_curve(yb, p[:, k])
        out[k] = (fpr, tpr, sk_auc(fpr, tpr))
    return out


def build_mean_curve(mode_dir: Path, tower_mode: str | None, mode: str) -> dict | None:
    fold_dirs = find_fold_dirs(mode_dir)
    if not fold_dirs:
        return None

    probs_stem: str | None = None
    for fd in fold_dirs:
        probs_stem = _detect_probs_stem(fd, tower_mode)
        if probs_stem:
            break
    if probs_stem is None:
        return None

    grid = np.linspace(0, 1, 501)
    per_fold: list[dict] = []
    for fd in fold_dirs:
        result = load_fold(fd, probs_stem)
        if result is None:
            continue
        y, p = result
        if mode == "binary":
            mask = np.isin(y, [0, 1])
            y, p = y[mask], p[mask]
            if p.shape[1] > 2:
                p = p[:, :2]
        per_fold.append(per_class_roc(y, p))

    if not per_fold:
        return None

    K = max(max(d.keys()) for d in per_fold) + 1
    class_curves: dict[int, dict] = {}
    for k in range(K):
        tprs, aucs = [], []
        for d in per_fold:
            if k not in d:
                continue
            fpr, tpr, a = d[k]
            tprs.append(np.interp(grid, fpr, tpr))
            aucs.append(a)
        if not tprs:
            continue
        tprs_arr = np.vstack(tprs)
        class_curves[k] = {
            "fpr":      grid,
            "tpr_mean": tprs_arr.mean(axis=0),
            "tpr_std":  tprs_arr.std(axis=0),
            "auc_mean": float(np.nanmean(aucs)),
            "auc_std":  float(np.nanstd(aucs)),
        }

    return {"probs_stem": probs_stem, "class_curves": class_curves}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_run_entry(entry: str) -> tuple[Path, str]:
    """Parse  path:label  or  path  (label defaults to last two dir components)."""
    if ":" in entry:
        raw_path, label = entry.rsplit(":", 1)
    else:
        raw_path = entry
        p = Path(entry)
        label = f"{p.parent.name}/{p.name}"
    return Path(raw_path), label


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-class ROC curves comparing multiple v2 HyperTower runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--runs", nargs="+", required=True, metavar="PATH[:LABEL]",
        help=(
            "Mode-level directories to compare, each optionally followed by :label. "
            "Path should point to the {eval_mode}/{tower_mode} subdirectory."
        ),
    )
    ap.add_argument("--mode", required=True, choices=["binary", "multiclass"])
    ap.add_argument("--tag", required=True, help="Output filename prefix.")
    ap.add_argument(
        "--output-dir", default="analysis_data/roc_plots",
        help="Directory for PNG and JSON output (default: analysis_data/roc_plots).",
    )
    ap.add_argument(
        "--class-names", nargs="*", default=["Healthy", "Glaucoma", "Suspect"],
    )
    ap.add_argument("--shade", action="store_true", help="Shade ±1 SD bands.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_model: list[tuple[str, dict]] = []
    for entry in args.runs:
        mode_dir, label = parse_run_entry(entry)
        if not mode_dir.exists():
            print(f"  WARNING: {mode_dir} not found — skipping.")
            continue
        tower_mode = mode_dir.name
        result = build_mean_curve(mode_dir, tower_mode, args.mode)
        if result is None:
            print(f"  WARNING: no usable folds in {mode_dir} — skipping.")
            continue
        auc_str = "  ".join(
            f"class{k} AUC={v['auc_mean']:.3f}±{v['auc_std']:.3f}"
            for k, v in result["class_curves"].items()
        )
        print(f"  {label}  [{result['probs_stem']}]  {auc_str}")
        per_model.append((label, result["class_curves"]))

    if not per_model:
        raise SystemExit("No usable runs — nothing to plot.")

    if args.mode == "binary":
        classes_to_plot = [1]
        out_names = [f"{args.tag}_binary_roc.png"]
        titles = ["Binary — Glaucoma (positive class)"]
    else:
        max_k = max(max(curves.keys()) for _, curves in per_model)
        classes_to_plot = list(range(min(3, max_k + 1)))
        out_names = [f"{args.tag}_class{k}_roc.png" for k in classes_to_plot]
        titles = [
            f"Multiclass OVR — "
            f"{args.class_names[k] if k < len(args.class_names) else f'class {k}'}"
            for k in classes_to_plot
        ]

    out_json: dict = {"tag": args.tag, "mode": args.mode, "figures": []}

    for k, out_name, title in zip(classes_to_plot, out_names, titles):
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="grey")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{title}\n{args.tag}")

        entries = []
        for label, curves in per_model:
            if k not in curves:
                continue
            c = curves[k]
            ax.plot(
                c["fpr"], c["tpr_mean"], linewidth=2,
                label=f"{label}  (AUC {c['auc_mean']:.3f} ± {c['auc_std']:.3f})",
            )
            if args.shade:
                ax.fill_between(
                    c["fpr"],
                    np.clip(c["tpr_mean"] - c["tpr_std"], 0, 1),
                    np.clip(c["tpr_mean"] + c["tpr_std"], 0, 1),
                    alpha=0.10,
                )
            entries.append({
                "label": label,
                "auc_mean": c["auc_mean"],
                "auc_std": c["auc_std"],
            })

        ax.legend(loc="lower right")
        fig.tight_layout()
        out_path = out_dir / out_name
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"  Saved: {out_path}")

        out_json["figures"].append({
            "class_index": k,
            "output_png": str(out_path),
            "models": entries,
        })

    summary_path = out_dir / f"{args.tag}_roc_summary.json"
    summary_path.write_text(json.dumps(out_json, indent=2))
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
