"""
Phase 5 comparison panel — ROC curves + fusion event summaries.

Layout:
  Top row    (1 × 3)  — ROC curves: Single HyperTower | Bilateral Ensemble | Fused Head
  Bottom rows (2 × 1)  — Fusion event summary (full-width) for Ensemble then Fused Head
                          (Single mode has fused-only bridge; no meaningful fusion events)

All data derived from predictions_test.csv — no checkpoints required.

Usage:
    python -m v3.scripts.output_analysis.explainability.comparison_panel_phase5
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

REPO_ROOT    = Path(__file__).resolve().parents[4]
RESULTS_ROOT = REPO_ROOT / "v3" / "results"
FIGURES_ROOT = REPO_ROOT / "v3" / "figures" / "explainability"
CLINICAL_DIR = REPO_ROOT / "Papila" / "ClinicalData"

RUNS = [
    {"label": "Single HyperTower", "run": "phase5/single_fused",
     "tower_path": "binary/single",   "is_single": False},
    {"label": "Bilateral Ensemble", "run": "phase5/ensemble_fused",
     "tower_path": "binary/ensemble", "is_single": False},
    {"label": "Fused Head",         "run": "phase5/logit_mlp_head",
     "tower_path": "binary/ensemble", "is_single": False},
]

# Event taxonomy (img=image tower, md=clinical tower)
_EVENT_KEYS = [
    "full_correction", "img_assist", "md_assist",
    "full_error",      "img_drag",   "md_drag",
    "concordant_correct", "concordant_wrong",
]
_EVENT_COLORS = [
    "#2ca02c", "#98df8a", "#b5cf6b",   # positive
    "#d62728", "#ff9896", "#ffbb78",   # negative
    "#aec7e8", "#c5b0d5",              # concordant
]
_POSITIVE_KEYS = _EVENT_KEYS[:3]
_NEGATIVE_KEYS = _EVENT_KEYS[3:6]
_DISAGREE_KEYS = _POSITIVE_KEYS + _NEGATIVE_KEYS   # exclude concordant


# ── Data loading ──────────────────────────────────────────────────────────────

def load_pooled(run: str, tower_path: str) -> pd.DataFrame:
    run_dir = RESULTS_ROOT / run
    rows = []
    for rep in sorted(run_dir.glob("rep*")):
        tm = rep / tower_path
        if not tm.exists():
            continue
        for fold in sorted(tm.glob("fold[0-9]")):
            csv = fold / "predictions_test.csv"
            if csv.exists():
                df = pd.read_csv(csv)
                df["rep"]  = rep.name
                df["fold"] = fold.name
                rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No predictions found under {run_dir}/{tower_path}")
    return pd.concat(rows, ignore_index=True)


def classify_events(df: pd.DataFrame) -> pd.DataFrame:
    """Add event_type column based on pred_fused/pred_img/pred_md vs y_true."""
    df = df.copy()
    y  = df["y_true"].values
    pf = df["pred_fused"].values
    pi = df["pred_img"].values
    pm = df["pred_md"].values

    fused_ok = pf == y
    img_ok   = pi == y
    md_ok    = pm == y

    def _classify(fo, io, mo):
        if fo and io and mo:   return "concordant_correct"
        if not fo and not io and not mo: return "concordant_wrong"
        if fo and not io and not mo:     return "full_correction"
        if fo and io and not mo:         return "img_assist"
        if fo and not io and mo:         return "md_assist"
        if not fo and io and mo:         return "full_error"
        if not fo and not io and mo:     return "img_drag"
        if not fo and io and not mo:     return "md_drag"
        return "other"

    df["event_type"]  = [_classify(fo, io, mo)
                         for fo, io, mo in zip(fused_ok, img_ok, md_ok)]
    # conf_delta: fused prob minus average of img/md
    df["conf_fused"]  = df["prob_fused_c1"]
    df["conf_img"]    = df["prob_img_c1"]
    df["conf_md"]     = df["prob_md_c1"]
    df["conf_delta"]  = df["conf_fused"] - 0.5 * (df["conf_img"] + df["conf_md"])
    return df


# ── ROC panel ─────────────────────────────────────────────────────────────────

def _draw_roc(ax, df: pd.DataFrame, label: str, color: str) -> None:
    """Draw per-fold ROC curves (faint) + mean ROC (bold) on ax."""
    fold_aucs = []
    for (rep, fold), grp in df.groupby(["rep", "fold"]):
        if grp["y_true"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(grp["y_true"], grp["prob_fused_c1"])
        ax.plot(fpr, tpr, color=color, alpha=0.12, lw=0.8)
        fold_aucs.append(roc_auc_score(grp["y_true"], grp["prob_fused_c1"]))

    # Mean ROC via interpolation
    mean_fpr = np.linspace(0, 1, 200)
    tprs = []
    for (rep, fold), grp in df.groupby(["rep", "fold"]):
        if grp["y_true"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(grp["y_true"], grp["prob_fused_c1"])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = np.mean(fold_aucs)
    std_auc  = np.std(fold_aucs)
    ax.plot(mean_fpr, mean_tpr, color=color, lw=2.2,
            label=f"Mean AUC = {mean_auc:.3f} ± {std_auc:.3f}")
    ax.fill_between(mean_fpr,
                    np.percentile(tprs, 25, axis=0),
                    np.percentile(tprs, 75, axis=0),
                    color=color, alpha=0.12)
    ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.5)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False Positive Rate", fontsize=9)
    ax.set_ylabel("True Positive Rate", fontsize=9)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)


# ── Fusion summary panel ──────────────────────────────────────────────────────

def _draw_fusion_summary(axes_row, df: pd.DataFrame, label: str) -> None:
    """Draw 3-panel fusion summary (disagreement events only) on axes_row (list of 3 axes)."""
    event_color = dict(zip(_EVENT_KEYS, _EVENT_COLORS))
    event_labels = {
        "full_correction": "Full correction\n(both wrong → right)",
        "img_assist":      "Img assist\n(img✓ md✗ → right)",
        "md_assist":       "MD assist\n(md✓ img✗ → right)",
        "full_error":      "Full error\n(both right → wrong)",
        "img_drag":        "Img drag\n(img✗ md✓ → wrong)",
        "md_drag":         "MD drag\n(md✗ img✓ → wrong)",
    }

    # Only count disagreement events (exclude concordant)
    counts = {k: (df["event_type"] == k).sum() for k in _DISAGREE_KEYS}

    # Panel 0: totals bar (positive vs negative)
    ax = axes_row[0]
    for bar_x, keys in ((0, _POSITIVE_KEYS), (1, _NEGATIVE_KEYS)):
        bot = 0
        for k in keys:
            c = int(counts[k])
            ax.bar(bar_x, c, bottom=bot, color=event_color[k], width=0.5)
            if c > 0:
                ax.text(bar_x, bot + c / 2, str(c), ha="center", va="center",
                        fontsize=8, fontweight="bold")
            bot += c
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Positive\nevents", "Negative\nevents"])
    ax.set_ylabel("Count (all folds)")
    patches = [mpatches.Patch(color=event_color[k], label=event_labels[k].split("\n")[0])
               for k in _DISAGREE_KEYS if counts[k] > 0]
    ax.legend(handles=patches, fontsize=6, loc="upper right")
    ax.set_title(f"{label}\nDisagreement event totals", fontsize=9)

    # Panel 1: per-fold stacked bar (disagreement events only)
    ax = axes_row[1]
    fold_groups = sorted(df.groupby(["rep", "fold"]), key=lambda x: x[0])
    x = np.arange(len(fold_groups))
    pos_bot = np.zeros(len(fold_groups))
    neg_bot = np.zeros(len(fold_groups))
    for k, color in zip(_POSITIVE_KEYS, _EVENT_COLORS[:3]):
        vals = np.array([(g["event_type"] == k).sum() for _, g in fold_groups], dtype=float)
        ax.bar(x, vals, bottom=pos_bot, color=color, width=0.6)
        pos_bot += vals
    for k, color in zip(_NEGATIVE_KEYS, _EVENT_COLORS[3:6]):
        vals = np.array([(g["event_type"] == k).sum() for _, g in fold_groups], dtype=float)
        ax.bar(x + 0.65, vals, bottom=neg_bot, color=color, width=0.6)
        neg_bot += vals
    ax.set_xticks([])
    ax.set_xlabel("Fold", fontsize=8)
    ax.set_ylabel("Count"); ax.set_title("Per-fold breakdown\n(left=positive, right=negative)", fontsize=9)

    # Panel 2: img vs md confidence scatter (disagreement events only)
    ax = axes_row[2]
    for k in _DISAGREE_KEYS:
        sub = df[df["event_type"] == k]
        if len(sub) == 0:
            continue
        ax.scatter(sub["conf_img"], sub["conf_md"], c=event_color[k],
                   alpha=0.65, s=30, edgecolors="none",
                   label=event_labels[k].split("\n")[0])
    ax.plot([0, 1], [0, 1], "k--", lw=0.5, alpha=0.4)
    ax.set_xlabel("P(Glaucoma) — Image head"); ax.set_ylabel("P(Glaucoma) — MD head")
    ax.legend(fontsize=5.5, loc="lower right")
    ax.set_title("Tower confidence space\n(disagreement events only)", fontsize=9)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ROC_COLORS = ["#4c72b0", "#dd8452", "#55a868"]

    print("Loading predictions ...")
    datasets = []
    for cfg, color in zip(RUNS, ROC_COLORS):
        df = load_pooled(cfg["run"], cfg["tower_path"])
        df = classify_events(df)
        datasets.append((cfg, df, color))

    fusion_runs = [(cfg, df, color) for cfg, df, color in datasets]

    # ── Layout ────────────────────────────────────────────────────────────────
    # Row 0:        3 ROC axes
    # Rows 1,2:     2 fusion summary strips (5 axes each, spanning full width)
    n_fusion = len(fusion_runs)
    fig = plt.figure(figsize=(20, 6 + 4.5 * n_fusion))
    gs = fig.add_gridspec(
        1 + n_fusion, 1,
        height_ratios=[5] + [4.5] * n_fusion,
        hspace=0.35,
    )

    # ROC row — subdivide into 3
    roc_gs = gs[0].subgridspec(1, 3, wspace=0.28)
    for i, (cfg, df, color) in enumerate(datasets):
        ax = fig.add_subplot(roc_gs[i])
        _draw_roc(ax, df, cfg["label"], color)

    # Fusion rows (3 panels each)
    for fi, (cfg, df, color) in enumerate(fusion_runs):
        fus_gs = gs[1 + fi].subgridspec(1, 3, wspace=0.32)
        axes_row = [fig.add_subplot(fus_gs[j]) for j in range(3)]
        _draw_fusion_summary(axes_row, df, cfg["label"])

    fig.suptitle("Phase 5 — Model Comparison: ROC Curves & Fusion Event Analysis",
                 fontsize=13, fontweight="bold", y=1.01)

    out = FIGURES_ROOT / "comparison_panel_phase5.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
