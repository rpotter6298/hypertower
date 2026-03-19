#!/usr/bin/env python
"""
Plot fold-level AUC box plots for all phase 1 comparison variants.

Reads fold_metrics.csv files produced by phase_1_papila_reproduce.py
and generates a grouped box plot: one group per classifier, one box
per variant (tag).

Usage:
  python -m v3.scripts.output_analysis.plot_phase1_boxplots \
      --results-dir v3/results/phase1 \
      --tags paper_matched no_leakage hypertower_loader \
      --output v3/results/phase1/auc_boxplot_comparison.png
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

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


TAG_LABELS = {
    "paper_matched":      "Paper-matched\n(eye-level CV)",
    "no_leakage":         "No leakage\n(patient-level CV)",
    "hypertower_loader":  "HyperTower loader\n(patient-level CV)",
}

CLASSIFIER_ORDER = ["KNN", "Random Forest", "SVM", "Logistic Regression"]
CLASSIFIER_SHORT = {
    "KNN": "KNN",
    "Random Forest": "RF",
    "SVM": "SVM",
    "Logistic Regression": "LR",
}

# Colours per variant
VARIANT_COLOURS = [
    "#4878CF",   # blue  — paper_matched
    "#6ACC65",   # green — no_leakage
    "#D65F5F",   # red   — hypertower_loader
]


def load_fold_aucs(results_dir: Path, tags: list[str]) -> dict:
    """
    Returns {tag: {classifier_name: [fold_auc, ...]}}
    """
    data: dict = {}
    for tag in tags:
        tag_dir = results_dir / tag
        data[tag] = {}
        for clf in CLASSIFIER_ORDER:
            fpath = tag_dir / clf / "fold_metrics.csv"
            if fpath.exists():
                df = pd.read_csv(fpath)
                data[tag][clf] = df["auc"].tolist()
            else:
                print(f"  WARNING: missing {fpath}")
                data[tag][clf] = []
    return data


def plot_boxplots(data: dict, tags: list[str], output: Path, paper_aucs: dict | None = None):
    n_clf  = len(CLASSIFIER_ORDER)
    n_tags = len(tags)
    group_width = 0.8
    box_width   = group_width / n_tags * 0.85
    offsets     = np.linspace(-group_width / 2 + box_width / 2,
                               group_width / 2 - box_width / 2, n_tags)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for ti, tag in enumerate(tags):
        colour = VARIANT_COLOURS[ti % len(VARIANT_COLOURS)]
        label  = TAG_LABELS.get(tag, tag)
        first  = True
        for ci, clf in enumerate(CLASSIFIER_ORDER):
            aucs = data[tag].get(clf, [])
            if not aucs:
                continue
            x = ci + offsets[ti]
            bp = ax.boxplot(
                aucs,
                positions=[x],
                widths=box_width,
                patch_artist=True,
                boxprops=dict(facecolor=colour, alpha=0.75),
                medianprops=dict(color="black", linewidth=1.8),
                whiskerprops=dict(color=colour, linewidth=1.2),
                capprops=dict(color=colour, linewidth=1.2),
                flierprops=dict(marker="o", markersize=4,
                                markerfacecolor=colour, alpha=0.6),
                manage_ticks=False,
            )
            if first:
                bp["boxes"][0].set_label(label)
                first = False

    # Paper reference lines (dashed, per-classifier)
    if paper_aucs:
        for ci, clf in enumerate(CLASSIFIER_ORDER):
            if clf in paper_aucs:
                ax.hlines(paper_aucs[clf], ci - group_width / 2, ci + group_width / 2,
                          colors="black", linestyles=":", linewidths=1.2,
                          label="Paper (PAPILA)" if ci == 0 else "_nolegend_")

    ax.set_xticks(range(n_clf))
    ax.set_xticklabels([CLASSIFIER_SHORT[c] for c in CLASSIFIER_ORDER], fontsize=12)
    ax.set_ylabel("AUC (ROC)", fontsize=11)
    ax.set_title("Phase 1: Clinical-only classifier AUC by CV strategy", fontsize=12)
    ax.set_ylim(0.45, 1.02)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Saved: {output}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="v3/results/phase1")
    ap.add_argument("--tags", nargs="+",
                    default=["paper_matched", "no_leakage", "hypertower_loader"])
    ap.add_argument("--output", default=None,
                    help="Output PNG path. Default: <results-dir>/auc_boxplot_comparison.png")
    ap.add_argument("--no-paper-lines", action="store_true",
                    help="Omit the dotted paper-reported AUC reference lines.")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    output = Path(args.output) if args.output else results_dir / "auc_boxplot_comparison.png"

    data = load_fold_aucs(results_dir, args.tags)

    # PAPILA paper reported AUCs
    paper_aucs = None if args.no_paper_lines else {
        "KNN":                 0.75,
        "Random Forest":       0.64,
        "SVM":                 0.75,
        "Logistic Regression": 0.70,
    }

    plot_boxplots(data, args.tags, output, paper_aucs=paper_aucs)

    # Print summary table
    print(f"\n{'Classifier':<20}", end="")
    for tag in args.tags:
        label = tag.replace("_", " ")
        print(f"  {label:>22}", end="")
    print()
    print("-" * (20 + 24 * len(args.tags)))
    for clf in CLASSIFIER_ORDER:
        print(f"{CLASSIFIER_SHORT[clf]:<20}", end="")
        for tag in args.tags:
            aucs = data[tag].get(clf, [])
            if aucs:
                print(f"  {np.mean(aucs):.3f} ± {np.std(aucs):.3f}     ", end="")
            else:
                print(f"  {'—':>22}", end="")
        print()


if __name__ == "__main__":
    main()
