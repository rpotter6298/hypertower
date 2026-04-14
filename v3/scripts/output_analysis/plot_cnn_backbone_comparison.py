#!/usr/bin/env python
"""
Bar plot comparing CNN standalone vs HyperTower image_only AUC per backbone,
with PAPILA paper reference lines.

Usage:
  python -m v3.scripts.output_analysis.plot_cnn_backbone_comparison \
      --results-dir v3/results/phase1 \
      --output v3/results/phase1/cnn_backbone_comparison.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


BACKBONES = ["densenet121", "vgg16", "mobilenet_v2", "inception_v3", "resnet50"]
BACKBONE_LABELS = {
    "densenet121":  "DenseNet121",
    "vgg16":        "VGG16",
    "mobilenet_v2": "MobileNetV2",
    "inception_v3": "Inception V3",
    "resnet50":     "ResNet50",
}

# Per-backbone paper AUCs (binary, Test #2, PAPILA 2022)
PAPER_AUC = {
    "densenet121":  0.80,
    "vgg16":        0.84,
    "mobilenet_v2": 0.75,
    "inception_v3": 0.78,
    "resnet50":     0.78,
}
PAPER_STD = {
    "densenet121":  0.05,
    "vgg16":        0.02,
    "mobilenet_v2": 0.06,
    "inception_v3": 0.08,
    "resnet50":     0.07,
}

COLOURS = {
    "cnn":       "#4878CF",
    "ht":        "#D65F5F",
    "paper_ref": "black",
}


def load_cnn_fold_aucs(results_dir: Path, backbone: str) -> list[float]:
    fpath = results_dir / f"cnn_{backbone}" / "fold_metrics.csv"
    if not fpath.exists():
        print(f"  WARNING: missing {fpath}")
        return []
    df = pd.read_csv(fpath)
    return df["auc"].tolist()


def load_ht_fold_aucs(results_dir: Path, backbone: str, n_folds: int = 5) -> list[float]:
    aucs = []
    for fold in range(n_folds):
        fold_dir = results_dir / "imageonly_ht" / backbone / "binary" / "single" / f"fold{fold}"
        y_path  = fold_dir / "test_y_true.npy"
        p_path  = fold_dir / "test_probs_fused.npy"
        if not (y_path.exists() and p_path.exists()):
            print(f"  WARNING: missing predictions for {backbone} fold{fold}")
            continue
        y  = np.load(y_path)
        pr = np.load(p_path)
        if len(np.unique(y)) < 2:
            print(f"  WARNING: single-class test set for {backbone} fold{fold}, skipping")
            continue
        aucs.append(float(roc_auc_score(y, pr[:, 1])))
    return aucs


def plot(cnn_data: dict, ht_data: dict, output: Path):
    n = len(BACKBONES)
    x = np.arange(n)
    group_width = 0.7
    bar_w = group_width / 2 * 0.88
    offsets = [-group_width / 4, group_width / 4]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for bi, backbone in enumerate(BACKBONES):
        for si, (tag, data, colour) in enumerate([
            ("CNN standalone", cnn_data, COLOURS["cnn"]),
            ("HyperTower (image only)", ht_data, COLOURS["ht"]),
        ]):
            aucs = data.get(backbone, [])
            if not aucs:
                continue
            xpos = bi + offsets[si]
            mean, std = np.mean(aucs), np.std(aucs)
            ax.bar(
                xpos, mean, width=bar_w,
                color=colour, alpha=0.80,
                label=tag if bi == 0 else "_nolegend_",
            )
            ax.errorbar(
                xpos, mean, yerr=std,
                fmt="none", color="black", capsize=4, linewidth=1.2,
            )

        # Paper reference line spanning this backbone's group
        paper_val = PAPER_AUC.get(backbone)
        if paper_val is not None:
            lw = group_width / 2 + bar_w / 2
            label = "PAPILA paper" if bi == 0 else "_nolegend_"
            ax.hlines(
                paper_val,
                bi - group_width / 2, bi + group_width / 2,
                colors=COLOURS["paper_ref"], linestyles=":", linewidths=1.8,
                label=label,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([BACKBONE_LABELS[b] for b in BACKBONES], fontsize=11)
    ax.set_ylabel("AUC (ROC)", fontsize=11)
    ax.set_title("Phase 1: CNN backbone AUC — standalone vs HyperTower (image only)", fontsize=12)
    ax.set_ylim(0.45, 1.02)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Saved: {output}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="v3/results/phase1")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    output = Path(args.output) if args.output else results_dir / "cnn_backbone_comparison.png"

    cnn_data = {b: load_cnn_fold_aucs(results_dir, b) for b in BACKBONES}
    ht_data  = {b: load_ht_fold_aucs(results_dir, b)  for b in BACKBONES}

    plot(cnn_data, ht_data, output)

    # Summary table
    print(f"\n{'Backbone':<16}  {'CNN standalone':>18}  {'HT image_only':>18}  {'Paper':>12}")
    print("-" * 72)
    for b in BACKBONES:
        cnn_aucs = cnn_data[b]
        ht_aucs  = ht_data[b]
        cnn_str  = f"{np.mean(cnn_aucs):.3f} ± {np.std(cnn_aucs):.3f}" if cnn_aucs else "—"
        ht_str   = f"{np.mean(ht_aucs):.3f} ± {np.std(ht_aucs):.3f}"  if ht_aucs  else "—"
        p_str    = f"{PAPER_AUC[b]:.2f} ± {PAPER_STD[b]:.2f}"
        print(f"{BACKBONE_LABELS[b]:<16}  {cnn_str:>18}  {ht_str:>18}  {p_str:>12}")


if __name__ == "__main__":
    main()
