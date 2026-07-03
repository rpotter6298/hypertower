"""F2 - Backbone selection panel.

Box plot in the style of v3/figures/phase2_analysis.png (black-bordered boxes,
red median lines, baseline median reference). Three left-to-right sections:

  Block 1 (blue) -- Basic backbones (img-only, single-eye, ImageNet pretraining):
      VGG16, MobileNetV2, DenseNet121, InceptionV3, ResNet50
      Sourced from v4 experiments/backbone_replication/basic_* (10x5 = 50 fold-rep
      AUCs each, img-only single-eye, patient-grouped CV).

  Block 2 (blue) -- ResNet50 preprocessing/CV variations:
      Anonymous CV, GT crop, U-Net crop  (disc crops use margin 2.5x)
      "Anonymous CV" = patient-identity-agnostic cross-validation: fold
      assignment ignores PAPILA's patient IDs, allowing the same patient's
      OD/OS pair to be split across train and test. Reflects the standard
      protocol in benchmark reports that do not have patient-level labels
      (or do not respect them).
      Sourced from v4 experiments/backbone_replication/{anonymous_cv,gtcrop,
      unetcrop}_refugelike.

  Block 3 (orange) -- Baseline reference:
      "Baseline (fine-tuned ResNet50)" -- REFUGE-pretrained R50 image-only,
      sourced from v4 experiments/refuge_v2m_baseline/img_solo_single_refugelike.

Each non-baseline box is labelled with a Wilcoxon two-sided p-value comparing
its fold-rep AUCs to the baseline fold-rep AUCs.

Re-run anytime:
    python -m v4.figures.F2_papila_replication_and_single_mode
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from v4.figures.util.loaders import RESULTS_ROOT

OUT = Path(__file__).parent / "output" / "F2_backbones.png"

# Colors / styling (mirrors v3 phase2_analysis)
C_VAR     = "#4c72b0"   # blue   - non-baseline boxes
C_BASE    = "#dd8452"   # orange - baseline reference box
C_MEDIAN  = "#c44e52"   # red    - median line inside boxes
ALPHA     = 0.82

# Stage key for img-only single-eye fusion
STAGE_KEY = "img_fuse_test_auc"


def _wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    diffs = a - b
    if len(diffs) < 5 or np.all(diffs == 0):
        return float("nan")
    try:
        return float(wilcoxon(diffs, alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def _load_fold_aucs(rel: str) -> np.ndarray:
    """Collect STAGE_KEY across all rep x fold for a v4 results subdirectory."""
    root = RESULTS_ROOT / rel
    if not root.exists():
        return np.array([])
    out: list[float] = []
    for s in sorted(root.glob("rep*/binary/summary.json")):
        d = json.loads(s.read_text())
        for fr in d.get("fold_results", []):
            v = fr.get(STAGE_KEY)
            if v is None or not np.isfinite(v):
                continue
            out.append(float(v))
    return np.array(out)


# Per-section data definitions (label, results-subdir under RESULTS_ROOT)

BASIC_BACKBONES = [
    ("VGG16",       "backbone_replication/basic_vgg16"),
    ("MobileNetV2", "backbone_replication/basic_mobilenet_v2"),
    ("DenseNet121", "backbone_replication/basic_densenet121"),
    ("InceptionV3", "backbone_replication/basic_inception_v3"),
    ("ResNet50",    "backbone_replication/basic_resnet50"),
]

RESNET_VARIATIONS = [
    ("Anonymous CV", "backbone_replication/anonymous_cv_refugelike"),
    ("GT crop",      "backbone_replication/gtcrop_refugelike"),
    ("U-Net crop",   "backbone_replication/unetcrop_refugelike"),
]

BASELINE_LABEL = "baseline\n(fine-tuned ResNet50)"
BASELINE_REL   = "refuge_v2m_baseline/img_solo_single_refugelike"


def render() -> None:
    block1 = [(lbl, _load_fold_aucs(rel)) for lbl, rel in BASIC_BACKBONES]
    block2 = [(lbl, _load_fold_aucs(rel)) for lbl, rel in RESNET_VARIATIONS]
    base_aucs = _load_fold_aucs(BASELINE_REL)

    print("Block 1 - Basic backbones (ImageNet pretraining):")
    for lbl, a in block1:
        if len(a):
            print(f"  {lbl:<14s} n={len(a):>3d}  mean={a.mean():.4f} +/- {a.std():.4f}")
        else:
            print(f"  {lbl:<14s} no data")
    print("Block 2 - ResNet50 (REFUGE) variations:")
    for lbl, a in block2:
        if len(a):
            print(f"  {lbl:<14s} n={len(a):>3d}  mean={a.mean():.4f} +/- {a.std():.4f}")
        else:
            print(f"  {lbl:<14s} no data")
    if len(base_aucs):
        print(f"Block 3 - Baseline: n={len(base_aucs)}  "
              f"mean={base_aucs.mean():.4f} +/- {base_aucs.std():.4f}")
    else:
        print("Block 3 - no baseline data")

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

    for x, aucs, color in zip(pos, all_aucs, all_colors):
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

    if len(base_aucs):
        ax.axhline(np.median(base_aucs),
                   color=C_BASE, linewidth=1.2, linestyle="--", alpha=0.55,
                   label="Baseline median")

    div1 = (section1_right + section2_left) / 2
    div2 = (section2_right + section3_left) / 2
    for d in (div1, div2):
        ax.axvline(d, color="#aaa", linewidth=0.7, alpha=0.65, linestyle="-")

    y_band = 1.02
    section_centers = [
        ((pos[0] + section1_right) / 2,        "Basic backbones (img-only, single)"),
        ((section2_left + section2_right) / 2, "ResNet50 variations"),
        ((section3_left + section3_right) / 2, "Baseline"),
    ]
    for cx, txt in section_centers:
        ax.text(cx, y_band, txt, ha="center", va="bottom",
                fontsize=10, color="#333", fontweight="bold",
                transform=ax.get_xaxis_transform())

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
