"""F7 — Backbone upgrade: refugelike → refuge V2-M.

Four architecture configurations ordered from least to most complex, each
shown as a paired box plot (refugelike vs refuge_efficientnet_v2_m). Same
F2-style: black-bordered boxes, red median lines, baseline median dashed
reference. Within each group a Wilcoxon p-value compares V2-M to refugelike.

Configs (left → right, increasing architectural complexity):
    1. Single-eye img+cd ensemble  (no bilateral aggregation)
    2. Bilateral img only          (bilateral hb, single tower)
    3. Bilateral img+cd ensemble   (production architecture)
    4. Bilateral tritower          (img+cd+geom)

Re-run anytime:
    python -m v4.figures.X1_v2m_punch
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import wilcoxon

from v4.figures.util.loaders import RESULTS_ROOT

OUT = Path(__file__).parent / "output" / "X1_v2m_backbone.png"

# ── Style ───────────────────────────────────────────────────────────────────
C_REFUGELIKE = "#dd8452"   # orange — the older fundus-pretrained baseline
C_REFUGE_V2M = "#4c72b0"   # blue   — the upgraded fundus-pretrained backbone
C_MEDIAN     = "#c44e52"   # red    — median line
ALPHA        = 0.82

# (config_label, refugelike_run_path, refuge_v2m_run_path)
CONFIGS = [
    (
        "Single\nimg only",
        RESULTS_ROOT / "refuge_v2m_baseline" / "img_solo_single_refugelike",
        RESULTS_ROOT / "refuge_v2m_baseline" / "img_solo_single_refuge_v2m",
    ),
    (
        "Single ensemble\n(img+cd)",
        RESULTS_ROOT / "refuge_v2m_baseline" / "ensemble_single_refugelike",
        RESULTS_ROOT / "refuge_v2m_baseline" / "ensemble_single_refuge_v2m",
    ),
    (
        "Bilateral ensemble\n(img+cd)",
        RESULTS_ROOT / "tri_v1" / "baseline_ensemble",
        RESULTS_ROOT / "efficientnet" / "refuge_efficientnetv2_m",
    ),
    (
        "3-way fusion\n(img+cd+geom)",
        RESULTS_ROOT / "tri_v1" / "baseline_tri",
        RESULTS_ROOT / "refuge_v2m_baseline" / "tritower",
    ),
]


def _wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    diffs = a - b
    if len(diffs) < 5 or np.all(diffs == 0):
        return float("nan")
    try:
        return float(wilcoxon(diffs, alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def load_fold_aucs(run_dir: Path) -> np.ndarray:
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


def render() -> None:
    data = []
    for label, refg_path, v2m_path in CONFIGS:
        refg = load_fold_aucs(refg_path)
        v2m  = load_fold_aucs(v2m_path)
        data.append((label, refg, v2m))
        print(f"  {label.replace(chr(10), ' '):<32s}  refg n={len(refg):>3d} {refg.mean():.3f}±{refg.std():.3f}   "
              f"v2m n={len(v2m):>3d} {v2m.mean():.3f}±{v2m.std():.3f}"
              if (len(refg) and len(v2m)) else f"  {label} pending")

    # Layout: 4 groups of 2 boxes
    box_w        = 0.46
    pair_gap     = 0.10
    group_gap    = 0.85
    group_width  = 2 * box_w + pair_gap
    positions: list[tuple[float, float]] = []
    p = 0.0
    for _ in CONFIGS:
        positions.append((p, p + box_w + pair_gap))
        p += group_width + group_gap

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    fig.suptitle("Backbone Upgrade — refugelike → refuge V2-M",
                 fontsize=13, fontweight="bold")

    boxprops_kw  = dict(linewidth=1.2, edgecolor="black")
    medianprops  = dict(color=C_MEDIAN, linewidth=2)
    whiskerprops = dict(color="black", linewidth=1.0)
    capprops     = dict(color="black", linewidth=1.0)
    flierprops   = dict(marker="o", markersize=3, alpha=0.55,
                        markerfacecolor="#888", markeredgecolor="#444")

    for (label, refg, v2m), (xr, xv) in zip(data, positions):
        if len(refg):
            ax.boxplot(refg, positions=[xr], widths=box_w, patch_artist=True,
                       manage_ticks=False,
                       boxprops=dict(facecolor=C_REFUGELIKE, alpha=ALPHA, **boxprops_kw),
                       medianprops=medianprops, whiskerprops=whiskerprops,
                       capprops=capprops, flierprops=flierprops)
        if len(v2m):
            ax.boxplot(v2m, positions=[xv], widths=box_w, patch_artist=True,
                       manage_ticks=False,
                       boxprops=dict(facecolor=C_REFUGE_V2M, alpha=ALPHA, **boxprops_kw),
                       medianprops=medianprops, whiskerprops=whiskerprops,
                       capprops=capprops, flierprops=flierprops)


    # Group tick labels (config name + Wilcoxon p between paired boxes)
    tick_x  = [(xr + xv) / 2 for xr, xv in positions]
    tick_lb = []
    for (label, refg, v2m), _ in zip(data, positions):
        if len(refg) and len(v2m):
            n = min(len(refg), len(v2m))
            p_val = _wilcoxon_p(v2m[:n], refg[:n])
            ps = f"p={p_val:.3f}" if not np.isnan(p_val) else "p=n/a"
            tick_lb.append(f"{label}\n{ps}")
        else:
            tick_lb.append(label)
    ax.set_xticks(tick_x)
    ax.set_xticklabels(tick_lb, fontsize=9.5)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=C_REFUGELIKE, edgecolor="black",
                       alpha=ALPHA, label="refugelike (ResNet50 + REFUGE)"),
        mpatches.Patch(facecolor=C_REFUGE_V2M, edgecolor="black",
                       alpha=ALPHA, label="refuge V2-M (EfficientNetV2-M + REFUGE)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9, framealpha=0.92)

    # Limits and grid
    xmin = positions[0][0] - box_w
    xmax = positions[-1][1] + box_w
    ax.set_xlim(xmin - 0.3, xmax + 0.3)
    ax.set_ylim(0.55, 1.0)
    ax.set_ylabel("Test AUC (10 reps × 5 folds)", fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}")


if __name__ == "__main__":
    render()
