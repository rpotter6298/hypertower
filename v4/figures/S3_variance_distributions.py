"""S3 - Centered-offset distributions of architectural vs fold-rep variance.

Companion to S2, presenting the same variance decomposition as two
overlapping KDE curves on a common centered axis.

For each of the 200 (fold-rep, bridge) AUC observations, compute two
mean-centered offsets:

    architectural offset = AUC - mean(AUC over the 4 bridges in that fold-rep)
    fold-rep offset      = AUC - mean(AUC over the 50 fold-reps for that bridge)

Both sets have 200 values, both are centered at 0 by construction, and the
spread of each distribution corresponds directly to one of the two SDs in
the variance decomposition. Plotted as KDE curves on a shared x-axis, the
ratio of their widths is the variance ratio reported in S2.

Re-run:
    python -m v4.figures.S3_variance_distributions
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from v4.figures.util.loaders import RESULTS_ROOT


OUT = Path(__file__).parent / "output" / "S3_variance_distributions.png"

BRIDGES = [
    ("Concat",   "phase3_v4/single_bcd_concat"),
    ("Pairwise", "phase3_v4/single_bcd_pairwise"),
    ("Gated",    "phase3_v4/single_bcd_gated"),
    ("Hadamard", "refuge_v2m_baseline/ensemble_single_refugelike"),
]
STAGE_KEY = "nt_test_auc"

C_ARCH = "#222"                # dark grey for the architectural curve
BRIDGE_COLORS = {
    "Concat":   "#7f7f7f",     # grey  (underperformer)
    "Pairwise": "#ff7f0e",     # orange
    "Gated":    "#2ca02c",     # green
    "Hadamard": "#1f77b4",     # blue (default)
}


def collect() -> pd.DataFrame:
    rows = []
    for label, rel in BRIDGES:
        root = RESULTS_ROOT / rel
        for s in sorted(root.glob("rep*/binary/summary.json")):
            rep = int(s.parents[1].name.replace("rep", ""))
            d = json.loads(s.read_text())
            for fr in d.get("fold_results", []):
                v = fr.get(STAGE_KEY)
                if v is None or not np.isfinite(v):
                    continue
                rows.append({
                    "bridge": label,
                    "rep":    rep,
                    "fold":   int(fr["fold"]),
                    "auc":    float(v),
                })
    return pd.DataFrame(rows)


def render() -> None:
    df = collect()
    if df.empty:
        print("No data collected.")
        return

    # Compute centering offsets per (fold-rep, bridge) cell
    fold_rep_mean = df.groupby(["rep", "fold"])["auc"].transform("mean")
    bridge_mean   = df.groupby("bridge")["auc"].transform("mean")
    df["arch_offset"] = df["auc"] - fold_rep_mean
    df["fold_offset"] = df["auc"] - bridge_mean

    # Use the same SD formula as S2 (within-group SD averaged across groups)
    # so the two figures report identical pooled numbers.
    cell_var   = df.groupby(["rep", "fold"])["auc"].var(ddof=1)
    bridge_var = df.groupby("bridge")["auc"].var(ddof=1)
    arch_sd_pooled = float(np.sqrt(cell_var.mean()))
    fold_sd_pooled = float(np.sqrt(bridge_var.mean()))
    ratio = fold_sd_pooled / arch_sd_pooled if arch_sd_pooled > 0 else float("inf")

    # Per-bridge fold-rep SDs (50 fold-reps per bridge)
    per_bridge_sd = {b: float(np.sqrt(v)) for b, v in bridge_var.items()}

    print(f"n cells = {len(df)}")
    print(f"architectural SD (pooled, within-fold-rep avg) = {arch_sd_pooled:.4f}")
    print(f"fold-rep SD      (pooled, within-bridge avg)   = {fold_sd_pooled:.4f}")
    print(f"ratio fold-SD / arch-SD                         = {ratio:.2f}")
    print("Per-bridge fold-rep SDs:")
    for b in [name for name, _ in BRIDGES]:
        print(f"  {b:<10s} SD = {per_bridge_sd[b]:.4f}")

    # KDE x-axis: cover the union of all offset ranges
    all_offsets = np.concatenate([df["arch_offset"].values, df["fold_offset"].values])
    x_max = float(np.abs(all_offsets).max()) * 1.10
    xs = np.linspace(-x_max, x_max, 600)

    fig, ax = plt.subplots(figsize=(9.4, 5.6))

    # Per-bridge fold-rep offset curves (4 curves, 50 values each)
    bridge_order = [name for name, _ in BRIDGES]
    for b in bridge_order:
        offs = df.loc[df["bridge"] == b, "fold_offset"].values
        kde  = gaussian_kde(offs)
        y    = kde(xs)
        sd   = per_bridge_sd[b]
        ax.plot(xs, y, color=BRIDGE_COLORS[b], linewidth=1.6, alpha=0.92,
                zorder=3,
                label=f"{b}  fold-rep SD = {sd:.4f}")

    # Architectural offset curve (200 values pooled across cells)
    arch_offsets = df["arch_offset"].values
    kde_arch = gaussian_kde(arch_offsets)
    y_arch   = kde_arch(xs)
    ax.fill_between(xs, y_arch, color=C_ARCH, alpha=0.18, zorder=2)
    ax.plot(xs, y_arch, color=C_ARCH, linewidth=2.2, linestyle="--", zorder=4,
            label=f"Architectural (pooled)  SD = {arch_sd_pooled:.4f}")

    # Mean line at 0 (every distribution is centered there)
    ax.axvline(0, color="#666", linewidth=0.8, linestyle=":", alpha=0.6, zorder=0)

    ax.set_xlabel("AUC offset from grouping mean", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_xlim(-x_max, x_max)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.92)

    # Top-right annotation with the variance ratio
    txt = f"variance ratio  fold-SD / arch-SD = {ratio:.2f}"
    ax.text(
        0.985, 0.975, txt, transform=ax.transAxes,
        ha="right", va="top", fontsize=10.5, family="monospace",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                  edgecolor="#888", alpha=0.92),
    )

    fig.suptitle(
        "Architectural vs fold-rep variance: centered-offset distributions",
        fontsize=12.5, fontweight="bold", y=0.995,
    )
    fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}")


if __name__ == "__main__":
    render()
