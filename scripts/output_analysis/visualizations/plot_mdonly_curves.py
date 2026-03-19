#!/usr/bin/env python3
"""
Per-epoch learning curves for mdonly runs.

Reads epoch_log.csv from each fold dir and plots val_auc, val_acc,
hld_auc, hld_acc — one figure per metric, all folds as individual lines.

Usage:
    python scripts/output_analysis/visualizations/plot_mdonly_curves.py \
        --run-dirs analysis_data/pipeline_mdonly_50ep \
                   analysis_data/pipeline_mdonly_200ep \
                   analysis_data/pipeline_mdonly_500ep \
        --eval-mode binary
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = [
    ("val_auc", "Val AUC"),
    ("val_acc", "Val Accuracy"),
    ("hld_auc", "Holdout AUC"),
    ("hld_acc", "Holdout Accuracy"),
]

PHASE_SHADING = {
    "tower_warmup": "#d0e8ff",
    "fused_warmup": "#d0ffe8",
}


def _load_folds(mode_dir: Path) -> list[pd.DataFrame]:
    fold_dirs = sorted(
        [p for p in mode_dir.glob("fold*") if p.is_dir()],
        key=lambda p: int(p.name.replace("fold", "")),
    )
    frames = []
    for fd in fold_dirs:
        csv = fd / "epoch_log.csv"
        if not csv.exists():
            print(f"  [warn] {csv} not found, skipping")
            continue
        df = pd.read_csv(csv)
        df["_fold"] = int(fd.name.replace("fold", ""))
        frames.append(df)
    return frames


def _shade_warmup(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Shade warmup phase regions based on first fold's phase column."""
    if "phase" not in df.columns:
        return
    prev_phase = None
    start = None
    for _, row in df.iterrows():
        phase = row["phase"]
        ep = row["epoch"]
        if phase != prev_phase:
            if prev_phase in PHASE_SHADING and start is not None:
                ax.axvspan(start - 0.5, ep - 0.5, color=PHASE_SHADING[prev_phase],
                           alpha=0.35, zorder=0, label=f"{prev_phase.replace('_', ' ')}")
            start = ep
            prev_phase = phase
    # close last span
    if prev_phase in PHASE_SHADING and start is not None:
        ax.axvspan(start - 0.5, df["epoch"].max() + 0.5,
                   color=PHASE_SHADING[prev_phase], alpha=0.35, zorder=0)


def plot_curves(
    run_dirs: list[Path],
    eval_mode: str,
    tower_mode: str,
    out_dir: Path | None,
) -> None:
    # Collect (label, frames) pairs
    datasets: list[tuple[str, list[pd.DataFrame]]] = []
    for rd in run_dirs:
        mode_dir = rd / eval_mode / tower_mode
        if not mode_dir.exists():
            print(f"  [skip] {mode_dir} not found")
            continue
        frames = _load_folds(mode_dir)
        if not frames:
            print(f"  [skip] no epoch_log.csv found under {mode_dir}")
            continue
        datasets.append((rd.name, frames))

    if not datasets:
        print("No data found — nothing to plot.")
        return

    # One figure per metric
    for metric_key, metric_label in METRICS:
        # Check any fold actually has this metric with non-nan values
        has_data = any(
            not frames[0][metric_key].isna().all()
            for _, frames in datasets
            if frames and metric_key in frames[0].columns
        )
        if not has_data:
            continue

        n_runs = len(datasets)
        fig, axes = plt.subplots(1, n_runs, figsize=(5 * n_runs, 4.5), squeeze=False)

        for col_idx, (run_label, frames) in enumerate(datasets):
            ax = axes[0][col_idx]
            if frames and metric_key in frames[0].columns:
                _shade_warmup(ax, frames[0])

            colours = plt.cm.tab10(np.linspace(0, 0.9, len(frames)))
            for frame, colour in zip(frames, colours):
                if metric_key not in frame.columns:
                    continue
                vals = frame[metric_key].values
                epochs = frame["epoch"].values
                mask = ~np.isnan(vals.astype(float))
                if mask.sum() == 0:
                    continue
                ax.plot(epochs[mask], vals[mask],
                        linewidth=1.4, color=colour,
                        label=f"fold {frame['_fold'].iloc[0]}")

            ax.set_title(run_label, fontsize=10)
            ax.set_xlabel("Epoch")
            if col_idx == 0:
                ax.set_ylabel(metric_label)
            ax.legend(fontsize=7, loc="lower right")
            ax.grid(True, linewidth=0.4, alpha=0.5)

        fig.suptitle(f"{metric_label}  [{eval_mode} / {tower_mode}]", fontsize=12)
        fig.tight_layout()

        dest = out_dir or (run_dirs[0].parent / "mdonly_plots")
        dest.mkdir(parents=True, exist_ok=True)
        fname = f"mdonly_{metric_key}_{eval_mode}_{tower_mode}.png"
        fig.savefig(dest / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {dest / fname}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dirs", nargs="+", required=True,
                    help="One or more run directories (e.g. analysis_data/pipeline_mdonly_50ep).")
    ap.add_argument("--eval-mode", default="binary", choices=["binary", "multiclass"])
    ap.add_argument("--tower-mode", default="single", choices=["single", "ensemble"])
    ap.add_argument("--out", default=None,
                    help="Output directory for plots (default: {first_run_dir}/../mdonly_plots).")
    args = ap.parse_args()

    run_dirs = [Path(d) for d in args.run_dirs]
    out_dir = Path(args.out) if args.out else None
    plot_curves(run_dirs, args.eval_mode, args.tower_mode, out_dir)


if __name__ == "__main__":
    main()
