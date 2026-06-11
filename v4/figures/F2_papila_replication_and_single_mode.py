"""F2 — Backbone selection panel.

Box plot in the style of v3/figures/phase2_analysis.png (black-bordered boxes,
red median lines, baseline median reference). Three left-to-right sections:

  Block 1 (blue) — Basic backbones (img-only, single-eye, ImageNet pretraining):
      VGG16, MobileNetV2, DenseNet121, InceptionV3, ResNet50
      Sourced from v3 phase 1 / phase 2 fold AUCs. Will be refined with v4
      10x5 runs later; means should not move much.

  Block 2 (blue) — ResNet50 preprocessing/CV variations:
      leaky CV, GT crop, U-Net crop  (all 2.5x scale; 1.1x dropped from labels)
      Sourced from v3 phase 2 'classic_test_auc' (single-mode image-only).

  Block 3 (orange) — Baseline reference:
      "Baseline (fine-tuned ResNet50)" — what we previously called refugelike.
      Sourced from v3 phase 2 imageonly_refugelike_proper.

Each non-baseline box is labelled with a Wilcoxon two-sided p-value comparing
its fold AUCs to the baseline.

Re-run anytime:
    python -m v4.figures.F2_papila_replication_and_single_mode
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from v4.figures.util.loaders import REPO_ROOT

OUT = Path(__file__).parent / "output" / "F2_backbones.png"

# ── Colors / styling (mirrors v3 phase2_analysis) ────────────────────────────
C_VAR     = "#4c72b0"   # blue   — non-baseline boxes (basic backbones + variants)
C_BASE    = "#dd8452"   # orange — baseline reference box
C_MEDIAN  = "#c44e52"   # red    — median line inside boxes
ALPHA     = 0.82

V3_PHASE1_DIR = REPO_ROOT / "v3" / "results" / "phase1"
V3_PHASE2_DIR = REPO_ROOT / "v3" / "results" / "phase2"


def _wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    diffs = a - b
    if len(diffs) < 5 or np.all(diffs == 0):
        return float("nan")
    try:
        return float(wilcoxon(diffs, alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def _load_phase1_fold_aucs(subdir: str) -> np.ndarray:
    fp = V3_PHASE1_DIR / subdir / "fold_metrics.csv"
    if not fp.exists():
        return np.array([])
    df = pd.read_csv(fp)
    return df["auc"].dropna().astype(float).values


def _load_phase2_classic_aucs(run_name: str) -> np.ndarray:
    """Collect classic_test_auc across all rep×fold for a phase 2 run folder."""
    root = V3_PHASE2_DIR / run_name
    if not root.exists():
        return np.array([])
    out: list[float] = []
    for rep in sorted(root.glob("rep*")):
        fp = rep / "binary" / "single" / "fold_results.csv"
        if not fp.exists(): continue
        df = pd.read_csv(fp)
        if "classic_test_auc" not in df.columns: continue
        out.extend(df["classic_test_auc"].dropna().astype(float).tolist())
    return np.array(out)


# ── Per-section data definitions ─────────────────────────────────────────────
#   Each entry: (label, loader_fn, *args)

BASIC_BACKBONES = [
    ("VGG16",       _load_phase1_fold_aucs, "cnn_vgg16"),
    ("MobileNetV2", _load_phase1_fold_aucs, "cnn_mobilenet_v2"),
    ("DenseNet121", _load_phase1_fold_aucs, "cnn_densenet121"),
    ("InceptionV3", _load_phase1_fold_aucs, "cnn_inception_v3"),
    # Use phase 2 ResNet50 (50 fold AUCs) for tighter statistics on the
    # backbone that we sweep variations of in block 2.
    ("ResNet50",    _load_phase2_classic_aucs, "imageonly_resnet50_proper"),
]

RESNET_VARIATIONS = [
    ("leaky CV",   _load_phase2_classic_aucs, "imageonly_resnet50_leaky"),
    ("GT crop",    _load_phase2_classic_aucs, "imageonly_resnet50_gtcrop_2.5"),
    ("U-Net crop", _load_phase2_classic_aucs, "imageonly_resnet50_unetcrop_2.5"),
]

BASELINE_LABEL = "baseline\n(fine-tuned ResNet50)"
BASELINE_DATA  = (_load_phase2_classic_aucs, "imageonly_refugelike_proper")


def render() -> None:
    # Load everything
    block1 = [(lbl, fn(arg)) for lbl, fn, arg in BASIC_BACKBONES]
    block2 = [(lbl, fn(arg)) for lbl, fn, arg in RESNET_VARIATIONS]
    base_fn, base_arg = BASELINE_DATA
    base_aucs = base_fn(base_arg)

    print("Block 1 — Basic backbones:")
    for lbl, a in block1:
        print(f"  {lbl:<14s} n={len(a):>3d}  mean={a.mean():.3f}±{a.std():.3f}" if len(a) else f"  {lbl:<14s} no data")
    print("Block 2 — ResNet50 variations:")
    for lbl, a in block2:
        print(f"  {lbl:<14s} n={len(a):>3d}  mean={a.mean():.3f}±{a.std():.3f}" if len(a) else f"  {lbl:<14s} no data")
    print(f"Block 3 — Baseline: n={len(base_aucs)}  "
          f"mean={base_aucs.mean():.3f}±{base_aucs.std():.3f}" if len(base_aucs) else "Block 3 — no baseline data")

    # Lay out positions
    gap = 0.7
    pos: list[float] = []
    p = 0.0
    for _ in block1:
        pos.append(p); p += 1.0
    section1_right = p - 1.0
    p += gap
    section2_left = p
    for _ in block2:
        pos.append(p); p += 1.0
    section2_right = p - 1.0
    p += gap
    section3_left = p
    pos.append(p)
    section3_right = p
    total_w = p + 0.6

    fig, ax = plt.subplots(figsize=(13, 5.8))
    fig.suptitle("Backbone Selection", fontsize=13, fontweight="bold")

    box_w = 0.55
    boxprops_kw = dict(linewidth=1.2, edgecolor="black")
    medianprops = dict(color=C_MEDIAN, linewidth=2)
    whiskerprops = dict(color="black", linewidth=1.0)
    capprops    = dict(color="black", linewidth=1.0)
    flierprops  = dict(marker="o", markersize=3, alpha=0.55,
                       markerfacecolor="#888", markeredgecolor="#444")

    all_aucs: list[np.ndarray] = []
    all_labels: list[str] = []
    all_colors: list[str] = []

    for lbl, a in block1 + block2:
        all_labels.append(lbl); all_aucs.append(a); all_colors.append(C_VAR)
    all_labels.append(BASELINE_LABEL); all_aucs.append(base_aucs); all_colors.append(C_BASE)

    # Draw boxes
    for x, aucs, color in zip(pos, all_aucs, all_colors):
        if not len(aucs): continue
        bp = ax.boxplot(
            aucs, positions=[x], widths=box_w, patch_artist=True, manage_ticks=False,
            boxprops=dict(facecolor=color, alpha=ALPHA, **boxprops_kw),
            medianprops=medianprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            flierprops=flierprops,
        )

    # Baseline median reference line spanning the variant blocks
    if len(base_aucs):
        ax.axhline(np.median(base_aucs),
                   color=C_BASE, linewidth=1.2, linestyle="--", alpha=0.55,
                   label="Baseline median")

    # Dividers between sections (vertical light lines)
    div1 = (section1_right + section2_left) / 2
    div2 = (section2_right + section3_left) / 2
    for d in (div1, div2):
        ax.axvline(d, color="#aaa", linewidth=0.7, alpha=0.65, linestyle="-")

    # Section labels just above each block
    y_band = 1.02
    section_centers = [
        ((pos[0] + section1_right) / 2,                "Basic backbones (img-only, single)"),
        ((section2_left + section2_right) / 2,          "ResNet50 variations"),
        ((section3_left + section3_right) / 2,          "Baseline"),
    ]
    for cx, txt in section_centers:
        ax.text(cx, y_band, txt, ha="center", va="bottom",
                fontsize=10, color="#333", fontweight="bold",
                transform=ax.get_xaxis_transform())

    # X-tick labels (with p-values vs baseline beneath each variant box)
    tick_labels = []
    for lbl, aucs, color in zip(all_labels, all_aucs, all_colors):
        if color == C_BASE or not len(aucs) or not len(base_aucs):
            tick_labels.append(lbl); continue
        n = min(len(aucs), len(base_aucs))
        p_val = _wilcoxon_p(aucs[:n], base_aucs[:n])
        ps = f"p={p_val:.3f}" if not np.isnan(p_val) else "p=n/a"
        tick_labels.append(f"{lbl}\n{ps}")
    ax.set_xticks(pos)
    ax.set_xticklabels(tick_labels, fontsize=9.5)

    ax.set_xlim(-0.6, section3_right + 0.7)
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
