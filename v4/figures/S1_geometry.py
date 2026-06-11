"""F5 — Adding geometry as a third information source.

Box plot in the F2 style (black-bordered boxes, red median lines, baseline
median reference, Wilcoxon p-values). Two sections separated by a divider:

  Section A — Vector injection (compact 5-dim structured features)
      baseline (no geom)    | image+clinical ensemble, no geometry stream
      unet vector           | + 5-dim geometry features from UNet seg (auto)
      gt vector             | + 5-dim geometry features from GT contours (human)

  Section B — geometry network (a parallel CNN on segmentation maps)
      solo                  | geometry network alone, no img/cd
      unet fusion           | tritower img+cd+geom, UNet seg (auto)
      gt fusion             | tritower img+cd+geom, GT contours (human)

Baseline reference for both sections = "no geometry" ensemble. The figure
shows that:
  * geometry features carry signal alone (solo > chance)
  * a compact vector of GT-derived features modestly helps (+0.014)
  * UNet-derived features (auto) don't help meaningfully
  * a full geometry network doesn't help beyond what the img backbone has

Re-run after data lands:
    python -m v4.figures.S1_geometry
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from v4.figures.util.loaders import RESULTS_ROOT

OUT = Path(__file__).parent / "output" / "S1_geometry.png"

# ── Style (mirrors F2) ───────────────────────────────────────────────────────
C_VAR     = "#4c72b0"   # blue   — variant boxes
C_BASE    = "#dd8452"   # orange — baseline reference box
C_MEDIAN  = "#c44e52"   # red    — median line
ALPHA     = 0.82


def _wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    diffs = a - b
    if len(diffs) < 5 or np.all(diffs == 0):
        return float("nan")
    try:
        return float(wilcoxon(diffs, alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def load_fold_aucs(run_dir: Path) -> np.ndarray:
    """Aggregate test AUC across all rep × fold, picking the run's eval_stage."""
    import json
    if not run_dir.exists():
        return np.array([])
    out: list[float] = []
    for rep in sorted(run_dir.glob("rep*")):
        s = next(iter(rep.rglob("summary.json")), None)
        if s is None: continue
        d = json.loads(s.read_text())
        eval_stage = d.get("eval_stage", "hb")
        key = f"{eval_stage}_test_auc"
        for fr in d.get("fold_results", []):
            v = fr.get(key)
            if v is not None and np.isfinite(v):
                out.append(float(v))
    return np.array(out)


# ── Per-section data definitions ─────────────────────────────────────────────

# Baseline (used in both sections as reference)
BASELINE_LABEL = "baseline\n(no geometry)"
BASELINE_RUN   = RESULTS_ROOT / "ensemble_fused" / "no_geom"

# Section A — Vector injection variants (image+clinical ensemble, +EPC geom)
VECTOR_VARIANTS = [
    ("U-Net vector", RESULTS_ROOT / "tri_v1" / "geom_vec_unet"),     # currently 3 reps; 10-rep bump queued
    ("GT vector",    RESULTS_ROOT / "ensemble_fused" / "geom_gt"),
]

# Section B — network variants (CNN over segmentation maps)
NETWORK_VARIANTS = [
    ("solo (geom network alone)", RESULTS_ROOT / "tri_v1" / "baseline_solo"),
    ("U-Net fusion",            RESULTS_ROOT / "tri_v1" / "baseline_tri"),
    ("GT fusion",               RESULTS_ROOT / "phase6_v4" / "tritower_geom_gt"),
]


def render() -> None:
    base_aucs = load_fold_aucs(BASELINE_RUN)
    vec_data  = [(lbl, load_fold_aucs(p)) for lbl, p in VECTOR_VARIANTS]
    network_data = [(lbl, load_fold_aucs(p)) for lbl, p in NETWORK_VARIANTS]

    print(f"Baseline (no geometry): n={len(base_aucs):>3d}  "
          f"mean={base_aucs.mean():.3f}±{base_aucs.std():.3f}"
          if len(base_aucs) else "Baseline: no data")
    print("Vector injection variants:")
    for lbl, a in vec_data:
        print(f"  {lbl:<28s}  n={len(a):>3d}  mean={a.mean():.3f}±{a.std():.3f}"
              if len(a) else f"  {lbl:<28s}  pending")
    print("Network variants:")
    for lbl, a in network_data:
        print(f"  {lbl:<28s}  n={len(a):>3d}  mean={a.mean():.3f}±{a.std():.3f}"
              if len(a) else f"  {lbl:<28s}  pending")

    # Layout positions
    box_w        = 0.55
    inner_gap    = 0.50
    section_gap  = 0.95

    # Section A: baseline | unet vector | gt vector
    section_a_labels = [BASELINE_LABEL] + [l for l, _ in vec_data]
    section_a_data   = [base_aucs] + [a for _, a in vec_data]
    section_a_colors = [C_BASE] + [C_VAR] * len(vec_data)

    # Section B: solo | unet fusion | gt fusion
    section_b_labels = [l for l, _ in network_data]
    section_b_data   = [a for _, a in network_data]
    section_b_colors = [C_VAR] * len(network_data)

    positions: list[float] = []
    p = 0.0
    for _ in section_a_labels:
        positions.append(p); p += box_w + inner_gap
    section_a_right = positions[-1] + box_w / 2
    p = positions[-1] + box_w + section_gap
    section_b_left = p
    for _ in section_b_labels:
        positions.append(p); p += box_w + inner_gap

    all_labels = section_a_labels + section_b_labels
    all_data   = section_a_data   + section_b_data
    all_colors = section_a_colors + section_b_colors

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    fig.suptitle("Geometry Integration", fontsize=13, fontweight="bold")

    boxprops_kw  = dict(linewidth=1.2, edgecolor="black")
    medianprops  = dict(color=C_MEDIAN, linewidth=2)
    whiskerprops = dict(color="black", linewidth=1.0)
    capprops     = dict(color="black", linewidth=1.0)
    flierprops   = dict(marker="o", markersize=3, alpha=0.55,
                        markerfacecolor="#888", markeredgecolor="#444")

    for x, aucs, color in zip(positions, all_data, all_colors):
        if not len(aucs):
            continue
        ax.boxplot(
            aucs, positions=[x], widths=box_w, patch_artist=True, manage_ticks=False,
            boxprops=dict(facecolor=color, alpha=ALPHA, **boxprops_kw),
            medianprops=medianprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            flierprops=flierprops,
        )

    # Baseline median reference line across the whole plot
    if len(base_aucs):
        ax.axhline(np.median(base_aucs), color=C_BASE,
                   linewidth=1.2, linestyle="--", alpha=0.55,
                   label="Baseline median  (no geometry)")

    # Section dividers
    div_x = (section_a_right + section_b_left - box_w / 2) / 2
    ax.axvline(div_x, color="#aaa", linewidth=0.7, alpha=0.6, linestyle="-")

    # Section headers
    sec_a_cx = (positions[0] + positions[len(section_a_labels) - 1]) / 2
    sec_b_cx = (positions[len(section_a_labels)] + positions[-1]) / 2
    ax.text(sec_a_cx, 1.02, "Vector injection (5-dim structured features)",
            ha="center", va="bottom", fontsize=11, fontweight="bold", color="#333",
            transform=ax.get_xaxis_transform())
    ax.text(sec_b_cx, 1.02, "Geometry network (CNN over segmentation map)",
            ha="center", va="bottom", fontsize=11, fontweight="bold", color="#333",
            transform=ax.get_xaxis_transform())

    # X-tick labels with Wilcoxon p-values vs baseline for non-baseline boxes
    tick_lbls = []
    for lbl, aucs in zip(all_labels, all_data):
        if lbl == BASELINE_LABEL or not len(aucs) or not len(base_aucs):
            tick_lbls.append(lbl); continue
        n = min(len(aucs), len(base_aucs))
        p_val = _wilcoxon_p(aucs[:n], base_aucs[:n])
        ps = f"p={p_val:.3f}" if not np.isnan(p_val) else "p=n/a"
        tick_lbls.append(f"{lbl}\n{ps}")
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_lbls, fontsize=9.5)

    ax.set_xlim(positions[0] - box_w, positions[-1] + box_w + 0.3)
    ax.set_ylim(0.55, 1.0)
    ax.set_ylabel("Test AUC", fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.92)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}")


if __name__ == "__main__":
    render()
