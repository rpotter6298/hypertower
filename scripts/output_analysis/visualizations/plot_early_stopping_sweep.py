#!/usr/bin/env python3
"""
Simulate early stopping at each epoch N and show what val/holdout AUC
you would have gotten if you stopped there.

For each fold and each candidate stopping epoch N:
  - Find the epoch <= N with the highest val AUC  (checkpoint selection)
  - Record the val AUC and holdout AUC at that epoch

Then plot mean ± std across all folds as a function of N.

Usage
-----
    python scripts/output_analysis/visualizations/plot_early_stopping_sweep.py \
        --run-dir analysis_data/pipeline_10x5 \
        --eval-mode binary \
        --tower-mode ensemble \
        --head fused
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HEAD_COL = {
    "fused":   ("ensemble_val_auc",     "ensemble_holdout_auc"),
    "img":     ("ensemble_val_auc_img", "ensemble_holdout_auc_img"),
    "md":      ("ensemble_val_auc_md",  "ensemble_holdout_auc_md"),
    "classic": ("classic_val_auc",      "classic_holdout_auc"),
}


def load_fold_logs(run_dir: Path, eval_mode: str, tower_mode: str,
                   val_col: str, hld_col: str):
    logs = []
    for rep_dir in sorted(run_dir.glob("rep*")):
        mode_dir = rep_dir / eval_mode / tower_mode
        if not mode_dir.exists():
            continue
        for fd in sorted(
            [d for d in mode_dir.iterdir() if d.is_dir() and d.name.startswith("fold")],
            key=lambda p: int(p.name.replace("fold", "")),
        ):
            log = fd / "epoch_log.csv"
            if not log.exists():
                continue
            df = pd.read_csv(log)
            if val_col not in df.columns or hld_col not in df.columns:
                continue
            df = df[["epoch", val_col, hld_col]].dropna()
            logs.append(df.reset_index(drop=True))
    return logs


def sweep(logs: list[pd.DataFrame], val_col: str, hld_col: str):
    max_epoch = max(df["epoch"].max() for df in logs)
    epochs = np.arange(1, int(max_epoch) + 1)

    val_mat = np.full((len(logs), len(epochs)), np.nan)
    hld_mat = np.full((len(logs), len(epochs)), np.nan)

    for i, df in enumerate(logs):
        for j, n in enumerate(epochs):
            window = df[df["epoch"] <= n]
            if window.empty:
                continue
            best_idx = window[val_col].idxmax()
            val_mat[i, j] = window.loc[best_idx, val_col]
            hld_mat[i, j] = window.loc[best_idx, hld_col]

    return epochs, val_mat, hld_mat


def plot(epochs, val_mat, hld_mat, out_path: Path, title: str, inverted: bool = False, ci_tol: float = 0.0):
    val_mean = np.nanmean(val_mat, axis=0)
    val_std  = np.nanstd(val_mat,  axis=0)
    hld_mean = np.nanmean(hld_mat, axis=0)
    hld_std  = np.nanstd(hld_mat,  axis=0)

    if inverted:
        # compute cost per fold, then aggregate — avoids max-of-mean bias
        best_val_per_fold = np.nanmax(val_mat, axis=1, keepdims=True)  # (n_folds, 1)
        best_hld_per_fold = np.nanmax(hld_mat, axis=1, keepdims=True)
        delta_val = best_val_per_fold - val_mat   # (n_folds, n_epochs)
        delta_hld = best_hld_per_fold - hld_mat
        y_val  = np.nanmean(delta_val, axis=0)
        y_hld  = np.nanmean(delta_hld, axis=0)
        sy_val = np.nanstd(delta_val,  axis=0)
        sy_hld = np.nanstd(delta_hld,  axis=0)
    else:
        y_val, y_hld   = val_mean, hld_mean
        sy_val, sy_hld = val_std,  hld_std

    fig, ax = plt.subplots(figsize=(11, 5))

    if inverted:
        # faint per-fold lines
        for i in range(delta_val.shape[0]):
            ax.plot(epochs, delta_val[i], color="steelblue", linewidth=0.6, alpha=0.18)
            ax.plot(epochs, delta_hld[i], color="firebrick", linewidth=0.6, alpha=0.18)

    ax.plot(epochs, y_val, color="steelblue", linewidth=2.0,
            label="Best val − val@N  (val cost of stopping early)" if inverted
                  else "Val AUC (best ckpt up to N)")
    ax.fill_between(epochs, y_val - sy_val, y_val + sy_val, color="steelblue", alpha=0.15)

    ax.plot(epochs, y_hld, color="firebrick", linewidth=2.0,
            label="Best hld − hld@N  (holdout cost of stopping early)" if inverted
                  else "Holdout AUC (at best val ckpt)")
    ax.fill_between(epochs, y_hld - sy_hld, y_hld + sy_hld, color="firebrick", alpha=0.15)

    if inverted:
        ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.4)

        # CI-crosses-zero regions (with optional tolerance)
        val_ci_zero  = (y_val - sy_val) <= ci_tol
        hld_ci_zero  = (y_hld - sy_hld) <= ci_tol
        both_ci_zero = val_ci_zero & hld_ci_zero

        ymax = max(np.nanmax(y_val), np.nanmax(y_hld)) * 1.15
        ax.fill_between(epochs, 0, ymax, where=val_ci_zero,
                        color="steelblue", alpha=0.12, label="val CI ≤ 0")
        ax.fill_between(epochs, 0, ymax, where=hld_ci_zero,
                        color="firebrick", alpha=0.12, label="holdout CI ≤ 0")
        ax.fill_between(epochs, 0, ymax, where=both_ci_zero,
                        color="purple", alpha=0.20, label="both CI ≤ 0")

        ax.set_ylabel("AUC lost vs best achievable")
        ax.set_ylim(-0.05, ymax)
        legend_loc = "upper right"
    else:
        # gap curve on twin axis
        gap_mean = val_mean - hld_mean
        ax2 = ax.twinx()
        ax2.plot(epochs, gap_mean, color="darkorange", linewidth=1.5,
                 linestyle="--", alpha=0.7, label="Val−Holdout gap")
        ax2.set_ylabel("Val − Holdout gap", color="darkorange", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="darkorange")
        ax2.set_ylim(-0.1, 0.4)
        lines2, labels2 = ax2.get_legend_handles_labels()

        best_hld_ep  = epochs[np.nanargmax(hld_mean)]
        best_hld_val = hld_mean[np.nanargmax(hld_mean)]
        ax.axvline(best_hld_ep, color="firebrick", linewidth=1.2, linestyle=":",
                   alpha=0.8, label=f"peak holdout @ epoch {best_hld_ep} ({best_hld_val:.3f})")
        stable = epochs >= 3
        min_gap_ep = epochs[stable][np.nanargmin(gap_mean[stable])]
        ax.axvline(min_gap_ep, color="darkorange", linewidth=1.2, linestyle=":",
                   alpha=0.8, label=f"min gap @ epoch {min_gap_ep}")
        ax.set_ylabel("AUC")
        ax.set_ylim(0.5, 1.05)
        legend_loc = "lower right"

    ax.set_xlabel("Stopping epoch N")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    lines1, labels1 = ax.get_legend_handles_labels()
    if not inverted:
        lines1 += lines2; labels1 += labels2
    ax.legend(lines1, labels1, fontsize=8, loc=legend_loc)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir",    default="analysis_data/pipeline_10x5")
    ap.add_argument("--eval-mode",  default="binary")
    ap.add_argument("--tower-mode", default="ensemble")
    ap.add_argument("--head",       default="fused", choices=list(HEAD_COL))
    ap.add_argument("--out",        default=None)
    ap.add_argument("--inverted",   action="store_true",
                    help="Plot best-achievable minus current (cost of stopping early)")
    ap.add_argument("--ci-tol",     type=float, default=0.0,
                    help="Tolerance for CI-crosses-zero shading (default 0.0)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    val_col, hld_col = HEAD_COL[args.head]

    logs = load_fold_logs(run_dir, args.eval_mode, args.tower_mode, val_col, hld_col)
    if not logs:
        print("No epoch_log.csv files found.")
        return
    print(f"Loaded {len(logs)} fold logs")

    epochs, val_mat, hld_mat = sweep(logs, val_col, hld_col)

    tol_tag = f"_tol{args.ci_tol}" if args.ci_tol else ""
    suffix = f"_inverted{tol_tag}" if args.inverted else ""
    out = Path(args.out) if args.out else (
        run_dir / args.eval_mode / args.tower_mode / "plots" /
        f"early_stopping_sweep_{args.head}{suffix}.png"
    )
    title = ("Simulated early stopping — cost of stopping at epoch N\n"
             if args.inverted else
             "Simulated early stopping sweep\n")
    title += f"{run_dir.name} · {args.eval_mode}/{args.tower_mode} · head={args.head}"
    if args.inverted and args.ci_tol:
        title += f"  (CI tol={args.ci_tol})"
    plot(epochs, val_mat, hld_mat, out, title, inverted=args.inverted, ci_tol=args.ci_tol)


if __name__ == "__main__":
    main()
