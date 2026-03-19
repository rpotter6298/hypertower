#!/usr/bin/env python3
"""
Visualise aggregated GradCAM heatmaps produced by aggregate_gradcam.py.

Produces two figures:

Figure 1 — Mean heatmaps grid
  Rows: classes (e.g. Normal, Glaucoma)
  Cols: OD_all | OS_all | OD_correct | OD_incorrect | OS_correct | OS_incorrect

Figure 2 — Attention stats
  Panel A: disc_frac distribution per class (violin/box), OD and OS side by side
  Panel B: entropy distribution per class
  Panel C: disc_frac correct vs incorrect per class (scatter means + error bars)

Figure 3 — Disc attention vs correct confidence
  Scatter of disc_frac vs correct_conf (confidence if correct, 1-confidence if wrong)
  One panel per class, OD and OS overlaid, Pearson r annotated

Usage
-----
    python scripts/output_analysis/explainability/plot_gradcam_aggregate.py \
        --agg-dir analysis_data/pipeline_nocrop/binary/single/gradcam_aggregate
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


def load_agg(agg_dir: Path):
    npz  = np.load(agg_dir / "mean_heatmaps.npz")
    stats = pd.read_csv(agg_dir / "attention_stats.csv")
    return npz, stats


def _classes_from_npz(npz) -> list[str]:
    classes = []
    for key in npz.files:
        parts = key.split("_")
        # key format: {EYE}_{ClassName}_{split}_{stat}
        # ClassName may be multi-word (e.g. "Glaucoma", "Normal", "Suspect")
        if parts[-1] == "mean" and parts[-2] == "all" and parts[0] == "OD":
            classes.append(parts[1])
    return sorted(set(classes))


def plot_mean_heatmaps(npz, classes: list[str], out_path: Path):
    eyes   = ["OD", "OS"]
    splits = ["all", "correct", "incorrect"]
    cols   = [(e, s) for e in eyes for s in splits]   # 6 columns

    n_rows = len(classes)
    n_cols = len(cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.8, n_rows * 2.8))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for r, cls in enumerate(classes):
        for c, (eye, split) in enumerate(cols):
            ax  = axes[r, c]
            key = f"{eye}_{cls}_{split}_mean"
            if key not in npz:
                ax.axis("off")
                ax.set_title(f"{eye} {split}\n(no data)", fontsize=7)
                continue
            cam   = npz[key]
            count = int(npz.get(f"{eye}_{cls}_{split}_count", np.array(0)))
            ax.imshow(cam, cmap="jet", vmin=0, vmax=1)
            ax.axis("off")
            title = f"{cls} | {eye} {split}\n(N={count})"
            ax.set_title(title, fontsize=7)

    fig.suptitle("Mean GradCAM heatmaps by class / eye / outcome", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_attention_stats(stats: pd.DataFrame, classes: list[str], out_path: Path):
    eyes = ["OD", "OS"]
    cmap = plt.get_cmap("tab10")
    class_colors = {cls: cmap(i) for i, cls in enumerate(classes)}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ---- Panel A: disc_frac per class × eye ----
    ax = axes[0]
    positions = []
    labels    = []
    data_viol = []
    tick_pos  = []
    pos = 0
    for cls in classes:
        for eye in eyes:
            sub = stats[(stats["true_name"] == cls) & (stats["eye"] == eye)]["disc_frac"].dropna()
            data_viol.append(sub.values)
            positions.append(pos)
            labels.append(f"{cls[:3]}\n{eye}")
            tick_pos.append(pos)
            pos += 1
        pos += 0.5   # gap between classes

    vp = ax.violinplot(data_viol, positions=positions, showmedians=True, widths=0.7)
    for i, (pc, cls) in enumerate(zip(vp["bodies"], [c for c in classes for _ in eyes])):
        pc.set_facecolor(class_colors[cls])
        pc.set_alpha(0.65)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Disc fraction (attention mass within GT disc mask)")
    ax.set_title("Disc attention by class")
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # ---- Panel B: entropy per class × eye (same layout) ----
    ax = axes[1]
    data_ent = []
    for cls in classes:
        for eye in eyes:
            sub = stats[(stats["true_name"] == cls) & (stats["eye"] == eye)]["entropy"].dropna()
            data_ent.append(sub.values)

    vp2 = ax.violinplot(data_ent, positions=positions, showmedians=True, widths=0.7)
    for pc, cls in zip(vp2["bodies"], [c for c in classes for _ in eyes]):
        pc.set_facecolor(class_colors[cls])
        pc.set_alpha(0.65)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Attention entropy (higher = more diffuse)")
    ax.set_title("Attention entropy by class")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # ---- Panel C: disc_frac correct vs incorrect, mean ± std ----
    ax = axes[2]
    x_ticks  = []
    x_labels = []
    pos = 0
    for cls in classes:
        for eye in eyes:
            for split, marker, ls in [("correct", "o", "-"), ("incorrect", "X", "--")]:
                sub = stats[
                    (stats["true_name"] == cls) &
                    (stats["eye"] == eye) &
                    (stats["correct"] == (split == "correct"))
                ]["disc_frac"].dropna()
                if len(sub) == 0:
                    continue
                ax.errorbar(
                    pos, sub.mean(), yerr=sub.std(),
                    fmt=marker, color=class_colors[cls], linestyle=ls,
                    capsize=4, markersize=7, alpha=0.85,
                    label=f"{cls[:3]} {eye} {split}" if pos < 4 else "_",
                )
                pos += 1
            x_ticks.append(pos - 1.5)
            x_labels.append(f"{cls[:3]}\n{eye}")
        pos += 0.5

    ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_ylabel("Disc fraction")
    ax.set_title("Disc fraction: correct vs incorrect\n(circle=correct, X=incorrect)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # legend: one patch per class
    patches = [mpatches.Patch(color=class_colors[c], label=c) for c in classes]
    patches += [
        plt.Line2D([0], [0], marker="o", color="grey", label="correct",   linestyle="none"),
        plt.Line2D([0], [0], marker="X", color="grey", label="incorrect", linestyle="none"),
    ]
    ax.legend(handles=patches, fontsize=7, loc="lower right")

    fig.suptitle("GradCAM attention statistics", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_disc_attention_correlation(stats: pd.DataFrame, classes: list[str], out_path: Path):
    """
    Scatter disc_frac vs correct_conf per class.

    correct_conf = confidence  if correct
                 = 1 - confidence  if incorrect

    This asks: does focusing attention on the disc region correlate with
    the model being more confident about the right answer?
    """
    import scipy.stats as scipy_stats

    stats = stats.copy()
    stats["correct_conf"] = np.where(
        stats["correct"],
        stats["confidence"],
        1.0 - stats["confidence"],
    )

    corr_colors  = {True: "steelblue", False: "tomato"}
    corr_labels  = {True: "Correct", False: "Incorrect"}
    is_binary    = len(classes) == 2

    n_cls = len(classes)
    fig, axes = plt.subplots(1, n_cls, figsize=(5 * n_cls, 5), sharey=True)
    if n_cls == 1:
        axes = [axes]

    for ax, cls in zip(axes, classes):
        sub = stats[stats["true_name"] == cls]
        x_all, y_all = [], []

        for correct_val, color in corr_colors.items():
            csub = sub[sub["correct"] == correct_val]
            x = csub["disc_frac"].values
            y = csub["correct_conf"].values
            ax.scatter(x, y, marker="o", color=color,
                       alpha=0.75, s=30,
                       label=corr_labels[correct_val],
                       edgecolors="none")
            x_all.extend(x.tolist())
            y_all.extend(y.tolist())

        # pooled regression line
        x_arr = np.array(x_all)
        y_arr = np.array(y_all)
        if len(x_arr) >= 3:
            r, p = scipy_stats.pearsonr(x_arr, y_arr)
            m, b = np.polyfit(x_arr, y_arr, 1)
            xs = np.linspace(0, 1, 100)
            ax.plot(xs, m * xs + b, color="black", linewidth=1.5, linestyle="--", alpha=0.7)
            p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
            ax.annotate(f"r={r:+.3f}\n{p_str}", xy=(0.05, 0.93), xycoords="axes fraction",
                        fontsize=9, va="top",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

        if is_binary:
            ax.axhline(0.5, color="red", linewidth=1.0, linestyle=":",
                       alpha=0.7, label="Decision boundary (0.50)")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Disc fraction\n(attention mass within GT disc mask)", fontsize=9)
        ax.set_title(cls, fontsize=11)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(linestyle="--", alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

    axes[0].set_ylabel("Correct-class confidence\n(conf if correct, 1−conf if wrong)", fontsize=9)
    fig.suptitle("Disc attention vs correct-class confidence", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg-dir", required=True,
                    help="Directory produced by aggregate_gradcam.py")
    ap.add_argument("--out-heatmaps", default=None)
    ap.add_argument("--out-stats",    default=None)
    ap.add_argument("--out-corr",     default=None)
    args = ap.parse_args()

    agg_dir  = Path(args.agg_dir)
    out_hm   = Path(args.out_heatmaps) if args.out_heatmaps else agg_dir / "mean_heatmaps_plot.png"
    out_st   = Path(args.out_stats)    if args.out_stats    else agg_dir / "attention_stats_plot.png"
    out_corr = Path(args.out_corr)     if args.out_corr     else agg_dir / "disc_attention_correlation.png"

    npz, stats = load_agg(agg_dir)
    classes    = _classes_from_npz(npz)
    print(f"Classes found: {classes}")
    print(f"Total eye records in stats: {len(stats)}")

    plot_mean_heatmaps(npz, classes, out_hm)
    plot_attention_stats(stats, classes, out_st)
    plot_disc_attention_correlation(stats, classes, out_corr)


if __name__ == "__main__":
    main()
