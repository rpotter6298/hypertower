#!/usr/bin/env python3
"""Aggregate per-fold metrics across runs and visualize AUC vs accuracy.

The script scans every `summary.json` under the provided analysis directory,
loads the per-fold macro AUC values, and combines them with per-fold
predictions to compute accuracy. Two scatter plots are produced:

1. AUC vs. fold index (with jitter) coloured by fold.
2. Accuracy (x-axis) vs. AUC (y-axis) coloured by fold.

This helps identify folds that persistently underperform across experiments.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap
    from matplotlib.lines import Line2D
except ImportError as exc:  # pragma: no cover - forward-friendly error for runtime
    raise SystemExit("matplotlib is required to run this script") from exc


@dataclass
class FoldMetric:
    run_id: str
    fold: int
    auc: float
    accuracy: float
    summary_path: Path
    fusion_mode: Optional[str]
    plot_head: str


HEAD_SUFFIX = {
    "fused": "fused",
    "metadata": "md",
    "metadata_only": "md",
    "image": "img",
    "image_only": "img",
    "img": "img",
    "md": "md",
}


def infer_head(summary: Dict[str, object]) -> str:
    """Return the prediction head name used for evaluation."""
    plot_head = summary.get("plot_head")
    if isinstance(plot_head, str) and plot_head:
        key = plot_head.lower()
        if key in HEAD_SUFFIX:
            return key
    fusion_mode = summary.get("fusion_mode")
    if isinstance(fusion_mode, str):
        key = fusion_mode.lower()
        if key in HEAD_SUFFIX:
            return key
    # Fall back to fused head if nothing else matches
    return "fused"


def prediction_suffix(head: str) -> str:
    key = head.lower()
    if key in {"metadata", "metadata_only", "md"}:
        return "md"
    if key in {"image", "image_only", "img"}:
        return "img"
    return "fused"


def compute_accuracy(probs: np.ndarray, y_true: np.ndarray) -> float:
    if probs.ndim == 1:
        preds = (probs >= 0.5).astype(int)
    else:
        preds = np.argmax(probs, axis=1)
    y_int = y_true.astype(int)
    return float((preds == y_int).mean()) if y_int.size else np.nan


def load_summary(path: Path) -> Optional[Dict[str, object]]:
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[warn] Could not parse {path}: {exc}", file=sys.stderr)
        return None


def collect_metrics(summary_path: Path) -> Iterable[FoldMetric]:
    summary = load_summary(summary_path)
    if not summary:
        return []
    # Only keep multiclass experiments (num_classes > 2 or eval_mode explicitly multiclass)
    num_classes = summary.get("num_classes")
    eval_mode = summary.get("eval_mode")
    if (isinstance(num_classes, int) and num_classes <= 2) or (isinstance(eval_mode, str) and eval_mode.lower() == "binary"):
        return []

    head = infer_head(summary)
    per_fold_auc = summary.get("per_fold_macro_ovr_auc") or summary.get("per_fold_auc")
    if not isinstance(per_fold_auc, list):
        # Fallback for summaries that only store fold_metrics[*].stats.
        metric_key = f"auc_{prediction_suffix(head)}"
        fold_metrics = summary.get("fold_metrics")
        if not isinstance(fold_metrics, list):
            return []
        per_fold_auc = []
        for entry in fold_metrics:
            if not isinstance(entry, dict):
                return []
            stats = entry.get("stats")
            if not isinstance(stats, dict):
                return []
            auc_val = stats.get(metric_key)
            try:
                per_fold_auc.append(float(auc_val))
            except (TypeError, ValueError):
                return []

    suffix = prediction_suffix(head)
    run_id = summary.get("run_id", summary_path.parent.name)
    fusion_mode = summary.get("fusion_mode")

    for fold_idx, auc_val in enumerate(per_fold_auc):
        try:
            auc = float(auc_val)
        except (TypeError, ValueError):
            continue

        base = summary_path.parent
        probs_path = base / f"fold{fold_idx}_probs_{suffix}.npy"
        y_true_path = base / f"fold{fold_idx}_y_true.npy"
        if not probs_path.exists() or not y_true_path.exists():
            # fall back: if fused missing for metadata mode (or vice versa), try md or img
            if suffix != "fused":
                alt_probs_path = base / f"fold{fold_idx}_probs_fused.npy"
                if alt_probs_path.exists():
                    probs_path = alt_probs_path
            if not probs_path.exists():
                print(
                    f"[warn] Missing predictions for fold {fold_idx} in {base}; skipped",
                    file=sys.stderr,
                )
                continue
        try:
            probs = np.load(probs_path)
            y_true = np.load(y_true_path)
        except Exception as exc:
            print(f"[warn] Failed loading predictions for {base}: {exc}", file=sys.stderr)
            continue
        accuracy = compute_accuracy(probs, y_true)
        yield FoldMetric(
            run_id=str(run_id),
            fold=fold_idx,
            auc=auc,
            accuracy=accuracy,
            summary_path=summary_path,
            fusion_mode=fusion_mode if isinstance(fusion_mode, str) else None,
            plot_head=head,
        )


def build_plot(metrics: List[FoldMetric], output: Path, jitter: float, seed: int, show: bool) -> None:
    rng = np.random.default_rng(seed)
    folds = sorted({m.fold for m in metrics})
    fold_to_color: Dict[int, tuple] = {}
    cmap = get_cmap("tab10", max(len(folds), 1))
    for idx, fold in enumerate(folds):
        fold_to_color[fold] = cmap(idx)

    # Prepare arrays for plotting
    aucs = np.array([m.auc for m in metrics])
    accs = np.array([m.accuracy for m in metrics])
    fold_indices = np.array([m.fold for m in metrics])
    colors = [fold_to_color[m.fold] for m in metrics]
    jitter_offsets = rng.uniform(-jitter, jitter, size=len(metrics))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    # Panel 1: Fold vs AUC scatter with jitter
    ax0 = axes[0]
    ax0.scatter(fold_indices + 1 + jitter_offsets, aucs, c=colors, edgecolor="k", linewidth=0.4, alpha=0.85)
    ax0.set_xticks([f + 1 for f in folds])
    ax0.set_xlabel("Fold index")
    ax0.set_ylabel("Macro AUC")
    ax0.set_title("Per-fold AUC across runs")
    ax0.grid(True, linestyle=":", linewidth=0.5, alpha=0.4)

    # Panel 2: Accuracy vs AUC scatter
    ax1 = axes[1]
    ax1.scatter(accs, aucs, c=colors, edgecolor="k", linewidth=0.4, alpha=0.85)
    ax1.set_xlabel("Accuracy")
    ax1.set_ylabel("Macro AUC")
    ax1.set_title("Accuracy vs AUC by fold")
    ax1.grid(True, linestyle=":", linewidth=0.5, alpha=0.4)

    # Shared legend
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"Fold {fold + 1}",
            markerfacecolor=fold_to_color[fold],
            markeredgecolor="k",
            markersize=8,
        )
        for fold in folds
    ]
    for ax in axes:
        ax.legend(handles=legend_handles, frameon=False, loc="lower right")

    fig.suptitle("Fold-level performance across experiments", fontsize=14)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"Saved plot to {output}")

    if show:
        plt.show()
    plt.close(fig)


def print_summary(metrics: List[FoldMetric]) -> None:
    total_runs = len({m.run_id for m in metrics})
    print(f"Collected {len(metrics)} fold metrics from {total_runs} runs.")
    by_fold: Dict[int, List[FoldMetric]] = {}
    for metric in metrics:
        by_fold.setdefault(metric.fold, []).append(metric)
    for fold, entries in sorted(by_fold.items()):
        aucs = np.array([m.auc for m in entries])
        accs = np.array([m.accuracy for m in entries])
        print(
            f"  Fold {fold + 1}: AUC {aucs.mean():.3f} ± {aucs.std(ddof=0):.3f} | "
            f"Accuracy {accs.mean():.3f} ± {accs.std(ddof=0):.3f} (n={len(entries)})"
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Plot per-fold AUCs and accuracies across runs.")
    parser.add_argument(
        "--analysis-root",
        default="analysis_data",
        help="Root directory that contains run folders with summary.json files (default: analysis_data)",
    )
    parser.add_argument(
        "--output",
        default="analysis_data/fold_auc_vs_accuracy.png",
        help="Where to save the generated figure (default: analysis_data/fold_auc_vs_accuracy.png)",
    )
    parser.add_argument("--jitter", type=float, default=0.08, help="Horizontal jitter for fold scatter plot")
    parser.add_argument("--seed", type=int, default=17, help="Random seed for jitter replication")
    parser.add_argument("--show", action="store_true", help="Display the plot interactively after saving")
    args = parser.parse_args(argv)

    analysis_root = Path(args.analysis_root)
    if not analysis_root.exists():
        raise SystemExit(f"Analysis root {analysis_root} does not exist")

    summary_files = sorted(analysis_root.rglob("summary.json"))
    if not summary_files:
        raise SystemExit(f"No summary.json files found under {analysis_root}")

    metrics: List[FoldMetric] = []
    for summary_path in summary_files:
        metrics.extend(collect_metrics(summary_path))

    if not metrics:
        raise SystemExit("No fold metrics collected. Check that prediction files are present.")

    print_summary(metrics)
    build_plot(metrics, Path(args.output), jitter=args.jitter, seed=args.seed, show=args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
