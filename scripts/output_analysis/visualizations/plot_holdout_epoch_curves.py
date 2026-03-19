#!/usr/bin/env python3
"""
Plot per-epoch holdout metrics across all folds in a 10x5 (or any multi-rep) run.

Each fold gets its own line. Lines are coloured by rep.

Modes
-----
  holdout   — raw holdout AUC per epoch  (original plot)
  delta     — val_auc - holdout_auc per epoch (generalization gap;
               closer to 0 = val most faithfully reflects holdout)

Usage
-----
    python scripts/output_analysis/visualizations/plot_holdout_epoch_curves.py \
        --run-dir analysis_data/pipeline_10x5 \
        --eval-mode binary \
        --tower-mode ensemble \
        --head fused \
        --mode delta
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


def load_curves(run_dir: Path, eval_mode: str, tower_mode: str,
                val_col: str, hld_col: str, mode: str):
    """
    Returns list of (rep, fold, epochs_array, values_array).
    mode='holdout' → values = holdout_auc
    mode='delta'   → values = val_auc - holdout_auc
    """
    curves = []
    for rep_dir in sorted(run_dir.glob("rep*")):
        mode_dir = rep_dir / eval_mode / tower_mode
        if not mode_dir.exists():
            continue
        fold_dirs = sorted(
            [d for d in mode_dir.iterdir() if d.is_dir() and d.name.startswith("fold")],
            key=lambda p: int(p.name.replace("fold", "")),
        )
        for fd in fold_dirs:
            log = fd / "epoch_log.csv"
            if not log.exists():
                continue
            df = pd.read_csv(log)
            needed = [hld_col] if mode == "holdout" else [val_col, hld_col]
            if any(c not in df.columns for c in needed):
                continue
            df = df.dropna(subset=needed)
            if mode == "holdout":
                values = df[hld_col].to_numpy()
            elif mode == "delta":
                values = (df[val_col] - df[hld_col]).to_numpy()
            else:  # delta_adj
                val_arr = df[val_col].to_numpy()
                hld_arr = df[hld_col].to_numpy()
                val_best = np.nanmax(val_arr)
                penalty  = val_best - val_arr   # 0 when val is at its peak
                values   = (val_arr - hld_arr) + penalty
            curves.append((rep_dir.name, fd.name, df["epoch"].to_numpy(), values))
    return curves


def _build_mean_matrix(curves):
    all_ep = max(len(e) for _, _, e, _ in curves)
    mat = np.full((len(curves), all_ep), np.nan)
    for i, (_, _, e, a) in enumerate(curves):
        mat[i, :len(a)] = a
    return mat, np.arange(1, all_ep + 1)


def plot(curves, mode: str, out_path: Path, title: str):
    reps = sorted(set(r for r, _, _, _ in curves))
    cmap = matplotlib.colormaps.get_cmap("tab10")
    rep_color = {r: cmap(i / max(len(reps) - 1, 1)) for i, r in enumerate(reps)}

    fig, ax = plt.subplots(figsize=(12, 6))

    for rep, fold, epochs, vals in curves:
        ax.plot(epochs, vals, color=rep_color[rep], alpha=0.3, linewidth=0.9)

    # per-rep mean
    for rep in reps:
        rep_curves = [(e, a) for r, _, e, a in curves if r == rep]
        max_ep = max(len(e) for e, _ in rep_curves)
        mat = np.full((len(rep_curves), max_ep), np.nan)
        for i, (e, a) in enumerate(rep_curves):
            mat[i, :len(a)] = a
        mean_curve = np.nanmean(mat, axis=0)
        ax.plot(np.arange(1, max_ep + 1), mean_curve,
                color=rep_color[rep], linewidth=1.8, alpha=0.85, label=rep)

    # global mean ± std
    all_mat, ep_axis = _build_mean_matrix(curves)
    global_mean = np.nanmean(all_mat, axis=0)
    global_std  = np.nanstd(all_mat, axis=0)
    ax.plot(ep_axis, global_mean, color="black", linewidth=2.5, zorder=5, label="global mean")
    ax.fill_between(ep_axis, global_mean - global_std, global_mean + global_std,
                    color="black", alpha=0.12, zorder=4)

    if mode in ("delta", "delta_adj"):
        ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
        if mode == "delta":
            ax.set_ylabel("Val AUC − Holdout AUC  (gap)")
        else:
            ax.set_ylabel("(Val − Holdout) + (ValBest − Val)  (adjusted gap)")
        min_ep = int(ep_axis[np.nanargmin(global_mean)])
        min_val = global_mean[np.nanargmin(global_mean)]
        ax.axvline(min_ep, color="red", linewidth=1.2, linestyle=":", alpha=0.7,
                   label=f"min adjusted gap @ epoch {min_ep} ({min_val:+.3f})")
    else:
        ax.set_ylabel("Holdout AUC")
        ax.set_ylim(0, 1.05)

    ax.set_xlabel("Epoch")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, loc="upper right" if mode == "delta" else "lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

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
    ap.add_argument("--mode",       default="holdout", choices=["holdout", "delta", "delta_adj"])
    ap.add_argument("--out",        default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    val_col, hld_col = HEAD_COL[args.head]
    curves = load_curves(run_dir, args.eval_mode, args.tower_mode,
                         val_col, hld_col, args.mode)

    if not curves:
        print("No epoch_log.csv files found — check --run-dir / --eval-mode / --tower-mode")
        return

    print(f"Loaded {len(curves)} fold curves, up to {max(len(e) for _,_,e,_ in curves)} epochs each")

    out = Path(args.out) if args.out else (
        run_dir / args.eval_mode / args.tower_mode / "plots" /
        f"holdout_epoch_curves_{args.head}_{args.mode}.png"
    )
    label = {"holdout": "holdout AUC", "delta": "val−holdout gap", "delta_adj": "val−holdout gap (val-adjusted)"}[args.mode]
    title = (f"Per-fold {label} by epoch\n"
             f"{run_dir.name} · {args.eval_mode}/{args.tower_mode} · head={args.head}")
    plot(curves, args.mode, out, title)


if __name__ == "__main__":
    main()
