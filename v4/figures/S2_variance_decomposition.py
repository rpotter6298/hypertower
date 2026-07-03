"""S2 - Variance decomposition of the four-bridge L1 fusion comparison.

Two-panel supplementary figure summarising the variance decomposition
reported alongside section 3.2 of the manuscript.

Panel A (left): paired-line plot.
    x-axis : the 4 bridge variants (Concat, Pairwise, Gated, Hadamard)
    y-axis : eye-level test AUC
    each line : one fold-rep, connecting that fold-rep's 4 bridge AUCs
    overlay : per-bridge boxplot showing the marginal AUC distribution
    annotation : pooled across-architecture and across-fold-rep SDs,
                 and the SD ratio

Panel B (right): centered-offset KDEs.
    x-axis : AUC offset from grouping mean (centered at 0)
    y-axis : density
    four coloured curves : per-bridge fold-rep distributions (50 fold-reps
                           per bridge, centered by subtracting each bridge's
                           own mean)
    dashed dark curve     : architectural offset distribution (200 values,
                           centered by subtracting each fold-rep's mean)
    annotation : variance ratio

Read directly from v4/results/experiments/{phase3_v4,refuge_v2m_baseline}.

Re-run:
    python -m v4.figures.S2_variance_decomposition
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


OUT = Path(__file__).parent / "output" / "S2_variance_decomposition.png"


# Bridge label, results dir relative to experiments/, and stage key
BRIDGES = [
    ("Concat",   "phase3_v4/single_bcd_concat"),
    ("Pairwise", "phase3_v4/single_bcd_pairwise"),
    ("Gated",    "phase3_v4/single_bcd_gated"),
    ("Hadamard", "refuge_v2m_baseline/ensemble_single_refugelike"),
]
STAGE_KEY = "nt_test_auc"


# Panel-A colour palette (paired lines + boxplots)
C_LINE        = "#1f6fb0"    # single blue for all paired-cell lines
C_LINE_ALPHA  = 0.22
C_MARKER      = "#1f6fb0"
C_MEDIAN      = "#c44e52"    # red box median line
C_BOX_FILL    = "#dbe6f0"    # pale blue box fill

# Panel-B colour palette (per-bridge KDEs)
C_ARCH        = "#222"       # dark grey for the architectural curve
BRIDGE_COLORS = {
    "Concat":   "#7f7f7f",
    "Pairwise": "#ff7f0e",
    "Gated":    "#2ca02c",
    "Hadamard": "#1f77b4",
}


# ── Data loading + decomposition ────────────────────────────────────────────

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


def decompose(df: pd.DataFrame) -> tuple[float, float, float, dict[str, float]]:
    """Return (arch_sd_pooled, fold_sd_pooled, ratio, per_bridge_sd)."""
    cell_var   = df.groupby(["rep", "fold"])["auc"].var(ddof=1)
    bridge_var = df.groupby("bridge")["auc"].var(ddof=1)
    arch_sd = float(np.sqrt(cell_var.mean()))
    fold_sd = float(np.sqrt(bridge_var.mean()))
    ratio   = fold_sd / arch_sd if arch_sd > 0 else float("inf")
    per_bridge_sd = {b: float(np.sqrt(v)) for b, v in bridge_var.items()}
    return arch_sd, fold_sd, ratio, per_bridge_sd


# ── Panel A: paired-line plot ───────────────────────────────────────────────

def draw_panel_a(ax, df: pd.DataFrame,
                 arch_sd: float, fold_sd: float, ratio: float,
                 n_cells: int) -> None:
    bridge_order = [b for b, _ in BRIDGES]
    pivot = df.pivot_table(
        index=["rep", "fold"], columns="bridge", values="auc"
    )[bridge_order]
    x_positions = np.arange(len(bridge_order), dtype=float)

    # Paired cell lines: one per fold-rep
    for _, row in pivot.iterrows():
        if row.isna().any():
            continue
        ax.plot(
            x_positions, row.values, color=C_LINE, alpha=C_LINE_ALPHA,
            linewidth=0.9, marker="o", markersize=2.0,
            markerfacecolor=C_MARKER, markeredgecolor="none", zorder=2,
        )

    # Per-bridge boxplot
    box_data = [pivot[b].dropna().values for b in bridge_order]
    ax.boxplot(
        box_data,
        positions=x_positions,
        widths=0.32,
        patch_artist=True,
        manage_ticks=False,
        zorder=3,
        boxprops=dict(facecolor=C_BOX_FILL, edgecolor="black",
                      linewidth=1.0, alpha=0.85),
        whiskerprops=dict(color="black", linewidth=0.9),
        capprops=dict(color="black", linewidth=0.9),
        medianprops=dict(color=C_MEDIAN, linewidth=1.8),
        flierprops=dict(marker="", markersize=0),
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(bridge_order, fontsize=11)
    ax.set_xlabel("L1 fusion bridge", fontsize=11)
    ax.set_ylabel("Eye-level test AUC", fontsize=11)
    ax.set_xlim(-0.5, len(bridge_order) - 0.5)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    txt = (
        f"n = {n_cells} fold-rep AUC values | variance ratio {ratio:.2f}\n"
        f"  across-architecture SD = {arch_sd:.3f}\n"
        f"  across-fold-rep SD     = {fold_sd:.3f}"
    )
    ax.text(
        0.985, 0.025, txt, transform=ax.transAxes,
        ha="right", va="bottom", fontsize=9.5, family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#888", alpha=0.92),
    )


# ── Panel B: centered-offset KDE curves ─────────────────────────────────────

def draw_panel_b(ax, df: pd.DataFrame,
                 arch_sd: float, fold_sd: float, ratio: float,
                 per_bridge_sd: dict[str, float]) -> None:
    df = df.copy()
    fold_rep_mean = df.groupby(["rep", "fold"])["auc"].transform("mean")
    bridge_mean   = df.groupby("bridge")["auc"].transform("mean")
    df["arch_offset"] = df["auc"] - fold_rep_mean
    df["fold_offset"] = df["auc"] - bridge_mean

    all_offsets = np.concatenate(
        [df["arch_offset"].values, df["fold_offset"].values]
    )
    x_max = float(np.abs(all_offsets).max()) * 1.10
    xs = np.linspace(-x_max, x_max, 600)

    # Per-bridge fold-rep offset curves
    for b in [name for name, _ in BRIDGES]:
        offs = df.loc[df["bridge"] == b, "fold_offset"].values
        kde  = gaussian_kde(offs)
        y    = kde(xs)
        sd   = per_bridge_sd[b]
        ax.plot(xs, y, color=BRIDGE_COLORS[b], linewidth=1.6, alpha=0.92,
                zorder=3,
                label=f"{b}  SD = {sd:.3f}")

    # Architectural offset curve (pooled)
    arch_offsets = df["arch_offset"].values
    kde_arch = gaussian_kde(arch_offsets)
    y_arch   = kde_arch(xs)
    ax.fill_between(xs, y_arch, color=C_ARCH, alpha=0.18, zorder=2)
    ax.plot(xs, y_arch, color=C_ARCH, linewidth=2.2, linestyle="--", zorder=4,
            label=f"Architectural  SD = {arch_sd:.3f}")

    ax.axvline(0, color="#666", linewidth=0.8, linestyle=":",
               alpha=0.6, zorder=0)
    ax.set_xlabel("AUC offset from grouping mean", fontsize=11)
    ax.set_ylabel("Probability density", fontsize=11)
    ax.set_xlim(-x_max, x_max)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", which="both", left=True, labelleft=False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    # Headroom on the y-axis so the variance-ratio box does not crowd the
    # architectural-curve peak.
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax * 1.10)

    # Legend below the top so it clears the variance-ratio annotation.
    ax.legend(
        loc="upper left", bbox_to_anchor=(0.0, 0.82),
        fontsize=9, framealpha=0.92,
    )

    txt = f"variance ratio  fold-SD / arch-SD = {ratio:.2f}"
    ax.text(
        0.985, 0.975, txt, transform=ax.transAxes,
        ha="right", va="top", fontsize=10.5, family="monospace",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                  edgecolor="#888", alpha=0.92),
    )


# ── Combined render ─────────────────────────────────────────────────────────

def render() -> None:
    df = collect()
    if df.empty:
        print("No data collected; check the source paths.")
        return

    n_cells = df.groupby(["rep", "fold"]).ngroups
    n_bridges = df["bridge"].nunique()
    arch_sd, fold_sd, ratio, per_bridge_sd = decompose(df)

    print(f"Collected {len(df)} observations  ({n_bridges} bridges x {n_cells} cells)")
    print(f"  across-architecture SD (within cell)  : {arch_sd:.4f}")
    print(f"  across-fold-rep SD (within bridge)    : {fold_sd:.4f}")
    print(f"  ratio fold-SD / arch-SD               : {ratio:.2f}x")
    print("Per-bridge fold-rep SDs:")
    for b in [name for name, _ in BRIDGES]:
        print(f"  {b:<10s} SD = {per_bridge_sd[b]:.4f}")

    fig, (axA, axB) = plt.subplots(
        nrows=1, ncols=2, figsize=(16.0, 6.0),
        gridspec_kw=dict(wspace=0.22),
    )

    draw_panel_a(axA, df, arch_sd, fold_sd, ratio, n_cells)
    draw_panel_b(axB, df, arch_sd, fold_sd, ratio, per_bridge_sd)

    # Subfigure labels
    for ax, label in ((axA, "A"), (axB, "B")):
        ax.text(
            -0.07, 1.03, label, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=15, fontweight="bold",
        )

    fig.suptitle(
        "Variance decomposition: fold-assignment noise vs L1 bridge choice",
        fontsize=13.0, fontweight="bold", y=1.00,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}")


if __name__ == "__main__":
    render()
