"""
Phase 1–5 analysis — ablation plots + pairwise Wilcoxon tests.

Usage:
    python -m v3.scripts.output_analysis.analyze_phases          # all phases
    python -m v3.scripts.output_analysis.analyze_phases --phase 3
    python -m v3.scripts.output_analysis.analyze_phases --out figures/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

RESULTS_ROOT = Path(__file__).resolve().parents[3] / "v3" / "results"

# Shared style constants
C_BASELINE = "#dd8452"
C_OTHER    = "#4c72b0"
C_MEDIAN   = "#c44e52"
FSIZE      = 10


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_rep_aucs(run_path: Path, test_key: str = "classic_test") -> np.ndarray:
    """Return array of per-rep AUC means for a run directory."""
    aucs = []
    for rep_dir in sorted(run_path.glob("rep*")):
        for summary in rep_dir.rglob("summary.json"):
            txt = summary.read_text().strip()
            if not txt:
                continue
            d = json.loads(txt)
            auc = d.get("mode_summary", {}).get(test_key, {}).get("auc_mean")
            if auc is not None:
                aucs.append(auc)
                break  # one summary per rep
    return np.array(aucs)


def wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value; returns nan if underpowered."""
    diffs = a - b
    if np.all(diffs == 0) or len(diffs) < 5:
        return float("nan")
    try:
        return wilcoxon(diffs, alternative="two-sided").pvalue
    except Exception:
        return float("nan")


def stars(p: float) -> str:
    if np.isnan(p):   return ""
    if p < 0.001:     return "***"
    if p < 0.01:      return "**"
    if p < 0.05:      return "*"
    return "ns"


def paired_matrix(runs: list[str], aucs_dict: dict[str, np.ndarray],
                  fdr: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return (p_matrix, corrected_p_matrix) shape (n, n)."""
    n = len(runs)
    raw = np.full((n, n), np.nan)
    for i, a in enumerate(runs):
        for j, b in enumerate(runs):
            if i != j and a in aucs_dict and b in aucs_dict:
                ai, bi = aucs_dict[a], aucs_dict[b]
                min_n = min(len(ai), len(bi))
                if min_n >= 5:
                    raw[i, j] = wilcoxon_p(ai[:min_n], bi[:min_n])
    if fdr:
        mask = ~np.isnan(raw)
        if mask.sum() > 0:
            flat = raw[mask]
            _, corrected, _, _ = multipletests(flat, method="fdr_bh")
            corr = raw.copy()
            corr[mask] = corrected
            return raw, corr
    return raw, raw.copy()


def boxplot_panel(ax, data_list, labels, base_idx, title="", ylabel="Test AUC",
                  base_aucs=None, all_aucs_by_label=None):
    """Vertical box plot with p-value vs baseline under each tick label."""
    n = len(labels)
    x = np.arange(n)
    colors = [C_BASELINE if i == base_idx else C_OTHER for i in range(n)]

    bp = ax.boxplot(data_list, vert=True, patch_artist=True, positions=x,
                    widths=0.3, showfliers=True,
                    flierprops=dict(marker="o", markersize=3, alpha=0.5),
                    medianprops=dict(color=C_MEDIAN, linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    if base_aucs is not None:
        ax.axhline(np.median(base_aucs), color=C_BASELINE, linewidth=1,
                   linestyle="--", alpha=0.5, label="Baseline median")
        ax.legend(fontsize=FSIZE - 1)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylabel(ylabel, fontsize=FSIZE + 1)
    if title:
        ax.set_title(title, fontsize=FSIZE + 1, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    tick_labels = []
    for i, lbl in enumerate(labels):
        if i == base_idx or base_aucs is None:
            tick_labels.append(lbl)
            continue
        a = (all_aucs_by_label or {}).get(lbl, data_list[i])
        min_n = min(len(a), len(base_aucs))
        p = wilcoxon_p(a[:min_n], base_aucs[:min_n])
        p_str = f"p={p:.3f}" if not np.isnan(p) else "p=n/a"
        tick_labels.append(f"{lbl}\n{p_str}")

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=FSIZE)


def bar_plot(ax, labels, means, stds, baseline_idx, title, ylabel="AUC",
             baseline_aucs=None, all_aucs=None):
    """Horizontal bar chart with baseline highlighted and p-value annotations."""
    n = len(labels)
    colors = ["#4c72b0" if i != baseline_idx else "#dd8452" for i in range(n)]
    y = np.arange(n)
    bars = ax.barh(y, means, xerr=stds, color=colors, alpha=0.85,
                   height=0.6, capsize=3, error_kw=dict(linewidth=1))
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(ylabel)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.axvline(means[baseline_idx], color="#dd8452", linewidth=1, linestyle="--", alpha=0.6)

    # Annotate with p-value stars vs baseline
    if baseline_aucs is not None and all_aucs is not None:
        x_max = max(means) + max(stds) + 0.005
        for i, lbl in enumerate(labels):
            if i == baseline_idx:
                continue
            a = all_aucs.get(lbl)
            if a is None:
                continue
            min_n = min(len(a), len(baseline_aucs))
            p = wilcoxon_p(a[:min_n], baseline_aucs[:min_n])
            s = stars(p)
            if s:
                ax.text(x_max, i, s, va="center", fontsize=7,
                        color="black" if s != "ns" else "gray")


def pairwise_heatmap(ax, runs, p_matrix, title):
    """Lower-triangle heatmap of corrected p-values."""
    n = len(runs)
    display = np.full_like(p_matrix, np.nan)
    for i in range(n):
        for j in range(i):
            display[i, j] = p_matrix[i, j]

    im = ax.imshow(display, vmin=0, vmax=0.1, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(runs, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(runs, fontsize=7)
    ax.set_title(title, fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, label="p-value (FDR)")

    for i in range(n):
        for j in range(i):
            p = display[i, j]
            if not np.isnan(p):
                ax.text(j, i, f"{p:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if p < 0.05 else "black")


# ── Phase 1 ──────────────────────────────────────────────────────────────────

P1_CLF_ORDER  = ["KNN", "Random Forest", "SVM", "Logistic Regression"]
P1_CLF_LABELS = {"KNN": "KNN", "Random Forest": "RF", "SVM": "SVM", "Logistic Regression": "LR"}
P1_TAGS = [
    ("no_leakage",        "baseline"),
    ("hypertower_loader", "HT loader"),
]
P1_BASELINE_TAG = "no_leakage"
P1_PAPER_AUC = {"KNN": 0.75, "Random Forest": 0.64, "SVM": 0.75, "Logistic Regression": 0.70}

P1_BACKBONES = ["densenet121", "vgg16", "mobilenet_v2", "inception_v3", "resnet50"]
P1_BACKBONE_LABELS = {
    "densenet121":  "DenseNet121",
    "vgg16":        "VGG16",
    "mobilenet_v2": "MobileNetV2",
    "inception_v3": "InceptionV3",
    "resnet50":     "ResNet50",
}
P1_PAPER_CNN = {
    "densenet121": (0.80, 0.05), "vgg16": (0.84, 0.02),
    "mobilenet_v2": (0.75, 0.06), "inception_v3": (0.78, 0.08), "resnet50": (0.78, 0.07),
}


def _load_p1_clf_aucs(phase1_dir: Path) -> dict:
    """Load per-fold AUCs for each tag × classifier."""
    data = {}
    for tag, _ in P1_TAGS:
        data[tag] = {}
        for clf in P1_CLF_ORDER:
            fpath = phase1_dir / tag / clf / "fold_metrics.csv"
            if fpath.exists():
                data[tag][clf] = pd.read_csv(fpath)["auc"].tolist()
    return data


def _load_p1_cnn_aucs(phase1_dir: Path) -> tuple[dict, dict]:
    cnn, ht = {}, {}
    for b in P1_BACKBONES:
        fpath = phase1_dir / f"cnn_{b}" / "fold_metrics.csv"
        cnn[b] = pd.read_csv(fpath)["auc"].tolist() if fpath.exists() else []
        aucs = []
        for fold in range(5):
            yp = phase1_dir / "imageonly_ht" / b / "binary" / "single" / f"fold{fold}" / "test_y_true.npy"
            pp = phase1_dir / "imageonly_ht" / b / "binary" / "single" / f"fold{fold}" / "test_probs_fused.npy"
            if yp.exists() and pp.exists():
                y, pr = np.load(yp), np.load(pp)
                if len(np.unique(y)) >= 2:
                    aucs.append(float(roc_auc_score(y, pr[:, 1])))
        ht[b] = aucs
    return cnn, ht


def analyze_phase1(out_dir: Path):
    print("\n=== Phase 1 ===")
    phase1_dir = RESULTS_ROOT / "phase1"

    # ── Clinical classifiers ────────────────────────────────────────────────
    clf_data = _load_p1_clf_aucs(phase1_dir)
    n_clf  = len(P1_CLF_ORDER)
    n_tags = len(P1_TAGS)
    group_w = 0.7
    box_w   = group_w / n_tags * 0.85
    offsets = np.linspace(-group_w / 2 + box_w / 2, group_w / 2 - box_w / 2, n_tags)
    tag_colors = [C_BASELINE if t == P1_BASELINE_TAG else C_OTHER for t, _ in P1_TAGS]

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    fig1.suptitle("Phase 1 — Clinical-only classifiers: CV strategy comparison",
                  fontsize=FSIZE + 2, fontweight="bold")

    for ti, (tag, lbl) in enumerate(P1_TAGS):
        color = tag_colors[ti]
        first = True
        for ci, clf in enumerate(P1_CLF_ORDER):
            aucs = clf_data.get(tag, {}).get(clf, [])
            if not aucs:
                continue
            bp = ax1.boxplot(aucs, positions=[ci + offsets[ti]], widths=box_w,
                             patch_artist=True, manage_ticks=False,
                             boxprops=dict(facecolor=color, alpha=0.8),
                             medianprops=dict(color=C_MEDIAN, linewidth=2),
                             whiskerprops=dict(color=color, linewidth=1.2),
                             capprops=dict(color=color, linewidth=1.2),
                             flierprops=dict(marker="o", markersize=3, alpha=0.5))
            if first:
                bp["boxes"][0].set_label(lbl)
                first = False

    for ci, clf in enumerate(P1_CLF_ORDER):
        if clf in P1_PAPER_AUC:
            ax1.hlines(P1_PAPER_AUC[clf], ci - group_w / 2, ci + group_w / 2,
                       colors="black", linestyles=":", linewidths=1.5,
                       label="PAPILA paper" if ci == 0 else "_nolegend_")

    ax1.set_xticks(range(n_clf))
    ax1.set_xticklabels([P1_CLF_LABELS[c] for c in P1_CLF_ORDER], fontsize=FSIZE + 1)
    ax1.set_ylabel("Test AUC", fontsize=FSIZE + 1)
    ax1.set_ylim(0.45, 1.02)
    ax1.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.4)
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(fontsize=FSIZE, loc="lower right", framealpha=0.9)
    fig1.tight_layout()
    p1 = out_dir / "phase1_clinical_classifiers.png"
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved: {p1}")

    # ── CNN backbones ───────────────────────────────────────────────────────
    cnn_data, ht_data = _load_p1_cnn_aucs(phase1_dir)
    n_b = len(P1_BACKBONES)
    offsets2 = [-group_w / 4, group_w / 4]
    method_colors = [C_OTHER, C_BASELINE]

    fig2, ax2 = plt.subplots(figsize=(11, 5))
    fig2.suptitle("Phase 1 — CNN backbone: standalone vs HyperTower (image only)",
                  fontsize=FSIZE + 2, fontweight="bold")

    for bi, backbone in enumerate(P1_BACKBONES):
        for si, (lbl, data, color) in enumerate([
            ("CNN standalone", cnn_data, method_colors[0]),
            ("HyperTower",     ht_data,  method_colors[1]),
        ]):
            aucs = data.get(backbone, [])
            if not aucs:
                continue
            bp = ax2.boxplot(aucs, positions=[bi + offsets2[si]], widths=box_w,
                             patch_artist=True, manage_ticks=False,
                             boxprops=dict(facecolor=color, alpha=0.8),
                             medianprops=dict(color=C_MEDIAN, linewidth=2),
                             whiskerprops=dict(color=color, linewidth=1.2),
                             capprops=dict(color=color, linewidth=1.2),
                             flierprops=dict(marker="o", markersize=3, alpha=0.5))
            if bi == 0:
                bp["boxes"][0].set_label(lbl)

        if backbone in P1_PAPER_CNN:
            mean_p, _ = P1_PAPER_CNN[backbone]
            ax2.hlines(mean_p, bi - group_w / 2, bi + group_w / 2,
                       colors="black", linestyles=":", linewidths=1.5,
                       label="PAPILA paper" if bi == 0 else "_nolegend_")

    ax2.set_xticks(range(n_b))
    ax2.set_xticklabels([P1_BACKBONE_LABELS[b] for b in P1_BACKBONES], fontsize=FSIZE)
    ax2.set_ylabel("Test AUC", fontsize=FSIZE + 1)
    ax2.set_ylim(0.45, 1.02)
    ax2.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.4)
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(fontsize=FSIZE, loc="lower right", framealpha=0.9)
    fig2.tight_layout()
    p2 = out_dir / "phase1_cnn_backbones.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {p2}")


# ── Phase 2 ──────────────────────────────────────────────────────────────────

PHASE2_RUNS = [
    ("imageonly_resnet50_leaky",       "classic_test", "leaky CV"),
    ("imageonly_resnet50_proper",      "classic_test", "baseline"),
    ("imageonly_refugelike_proper",    "classic_test", "pretrained"),
    ("imageonly_resnet50_gtcrop_1.1",  "classic_test", "GT crop 1.1x"),
    ("imageonly_resnet50_gtcrop_2.5",  "classic_test", "GT crop 2.5x"),
    ("imageonly_resnet50_unetcrop_1.1","classic_test", "UNet crop 1.1x"),
    ("imageonly_resnet50_unetcrop_2.5","classic_test", "UNet crop 2.5x"),
]
PHASE2_BASELINE = "imageonly_resnet50_proper"
PHASE2_GROUPS = {
    "Backbone": ["imageonly_resnet50_proper", "imageonly_refugelike_proper"],
    "GT crop": ["imageonly_resnet50_proper", "imageonly_resnet50_gtcrop_1.1", "imageonly_resnet50_gtcrop_2.5"],
    "UNet crop": ["imageonly_resnet50_proper", "imageonly_resnet50_unetcrop_1.1", "imageonly_resnet50_unetcrop_2.5"],
    "Data leakage": ["imageonly_resnet50_leaky", "imageonly_resnet50_proper"],
}


def analyze_phase2(out_dir: Path):
    print("\n=== Phase 2 ===")
    aucs = {}
    for run, key, _ in PHASE2_RUNS:
        a = load_rep_aucs(RESULTS_ROOT / "phase2" / run, key)
        aucs[run] = a
        print(f"  {run:40s}  AUC={np.mean(a):.3f}±{np.std(a):.3f}  n={len(a)}")

    # Leakage impact
    leaky = aucs.get("imageonly_resnet50_leaky", np.array([]))
    proper = aucs.get("imageonly_resnet50_proper", np.array([]))
    if len(leaky) and len(proper):
        min_n = min(len(leaky), len(proper))
        p = wilcoxon_p(leaky[:min_n], proper[:min_n])
        delta = np.mean(leaky) - np.mean(proper)
        print(f"\n  Data leakage inflates AUC by {delta:+.3f}  (Wilcoxon p={p:.4f})")

    # Build display order: baseline first, then non-baseline sorted by mean AUC descending
    base_run = PHASE2_BASELINE
    base_label = next(lbl for r, _, lbl in PHASE2_RUNS if r == base_run)
    others = [(r, lbl) for r, _, lbl in PHASE2_RUNS if r != base_run]
    others.sort(key=lambda x: -np.mean(aucs[x[0]]) if len(aucs.get(x[0], [])) else float("inf"))
    ordered = [(base_run, base_label)] + others

    run_names = [r for r, _ in ordered]
    labels    = [lbl for _, lbl in ordered]
    base_idx  = 0
    base_aucs = aucs[base_run]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("Phase 2 — ResNet50: Backbone & Preprocessing Comparison", fontsize=12, fontweight="bold")

    data    = [aucs[r] for r in run_names]
    colors  = ["#dd8452" if r == base_run else "#4c72b0" for r in run_names]
    x       = np.arange(len(run_names))

    bp = ax.boxplot(data, vert=True, patch_artist=True, positions=x,
                    widths=0.3, showfliers=True,
                    flierprops=dict(marker="o", markersize=3, alpha=0.5),
                    medianprops=dict(color="#c44e52", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xlim(-0.5, len(run_names) - 0.5)
    ax.set_ylabel("Test AUC", fontsize=11)
    ax.axhline(np.median(base_aucs), color="#dd8452", linewidth=1,
               linestyle="--", alpha=0.5, label="Baseline median")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Build x-tick labels with p-value on a second line underneath
    tick_labels = []
    for i, run in enumerate(run_names):
        if run == base_run:
            tick_labels.append(labels[i])
            continue
        a = aucs[run]
        min_n = min(len(a), len(base_aucs))
        p = wilcoxon_p(a[:min_n], base_aucs[:min_n])
        p_str = f"p={p:.3f}" if not np.isnan(p) else "p=n/a"
        tick_labels.append(f"{labels[i]}\n{p_str}")

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=10)

    fig.tight_layout()
    path = out_dir / "phase2_analysis.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Phase 3 ──────────────────────────────────────────────────────────────────

PHASE3_GROUPS = {
    "Loss function": {
        "baseline":    "Baseline (BCE fused)",
        "loss_all":    "All losses",
        "loss_bcd_p03":"BCD p=0.3",
        "loss_bcd_p07":"BCD p=0.7",
    },
    "SE attention": {
        "baseline":      "Baseline",
        "se_img_tower":  "SE img tower",
        "se_cd_tower":   "SE cd tower",
        "se_bridge":     "SE bridge",
        "se_all":        "SE all",
    },
    "IOP correction": {
        "baseline":          "Baseline (none)",
        "iop_ratio":         "Ratio",
        "iop_ratio_drop_raw":"Ratio + drop raw",
        "iop_ols":           "OLS",
        "iop_lad":           "LAD",
        "iop_multi":         "Multi",
    },
    "Feature ablation": {
        "baseline":        "Baseline (all)",
        "excl_iop":        "Excl IOP",
        "excl_age":        "Excl age",
        "excl_axial_length":"Excl axial length",
        "excl_refractive": "Excl refractive",
    },
    "Network dims": {
        "baseline":       "Baseline",
        "cd_hidden_64":   "CD hidden=64",
        "cd_hidden_256":  "CD hidden=256",
        "fusion_dim_128": "Fusion dim=128",
        "fusion_dim_512": "Fusion dim=512",
    },
    "Dropout": {
        "baseline":         "Baseline (0.5)",
        "bridge_dropout_03":"Bridge drop=0.3",
        "bridge_dropout_07":"Bridge drop=0.7",
        "cd_dropout_03":    "CD drop=0.3",
    },
    "Backbone freezing": {
        "baseline":  "Baseline (75%)",
        "freeze_25": "Freeze 25%",
        "freeze_50": "Freeze 50%",
    },
    "Warmup": {
        "baseline":             "Baseline (cd40+twr3+fus3)",
        "warmup_no_cd":         "No CD warmup",
        "warmup_tower5_fused5": "Tower5+Fused5",
    },
    "Sampling": {
        "baseline":           "Baseline",
        "balanced_sampling":  "Balanced sampling",
    },
    "Epoch length": {
        "epochs_1":  "1 epoch",
        "epochs_5":  "5 epochs",
        "epochs_10": "10 epochs",
        "epochs_20": "20 epochs",
        "epochs_30": "30 epochs (baseline)",
        "epochs_50": "50 epochs",
    },
    "Learning rate": {
        "baseline": "Baseline (1e-4)",
        "lr_3e4":   "3e-4",
        "lr_1e3":   "1e-3",
        "lr_1e5":   "1e-5",
    },
}
PHASE3_BASELINE = "baseline"


def analyze_phase3(out_dir: Path):
    print("\n=== Phase 3 ===")
    all_runs = set()
    for group in PHASE3_GROUPS.values():
        all_runs.update(group.keys())
    aucs = {}
    for run in all_runs:
        a = load_rep_aucs(RESULTS_ROOT / "phase3" / run, "classic_test")
        aucs[run] = a
    base_aucs = aucs[PHASE3_BASELINE]
    print(f"  Baseline AUC: {np.mean(base_aucs):.3f}±{np.std(base_aucs):.3f}")

    # Build display order: baseline first, then each group (non-baseline, sorted desc)
    GAP = 1.2   # extra space between groups
    pos = 0.0
    positions, box_data, tick_labels, colors, is_sig = [], [], [], [], []
    group_spans = []  # (x_mid, group_name) for title annotations

    # Baseline box
    positions.append(pos)
    box_data.append(base_aucs)
    tick_labels.append("baseline")
    colors.append(C_BASELINE)
    is_sig.append(False)
    pos += 1 + GAP

    def _group_max_median(group_runs):
        vals = [np.median(aucs[r]) for r in group_runs if r != PHASE3_BASELINE and len(aucs.get(r, []))]
        return max(vals) if vals else 0.0

    sorted_groups = sorted(PHASE3_GROUPS.items(), key=lambda x: -_group_max_median(x[1]))

    for shade_idx, (group_name, group_runs) in enumerate(sorted_groups):
        non_base = [(r, lbl) for r, lbl in group_runs.items() if r != PHASE3_BASELINE]
        non_base.sort(key=lambda x: -np.mean(aucs[x[0]]) if len(aucs.get(x[0], [])) else float("inf"))

        group_start = pos
        for run, lbl in non_base:
            a = aucs.get(run, np.array([]))
            positions.append(pos)
            box_data.append(a)
            # p-value label under name
            min_n = min(len(a), len(base_aucs))
            p = wilcoxon_p(a[:min_n], base_aucs[:min_n]) if min_n >= 5 else float("nan")
            sig = not np.isnan(p) and p < 0.05
            if sig:
                tick_labels.append(f"* {lbl}\np={p:.3f}")
            else:
                tick_labels.append(lbl)
            colors.append(C_OTHER)
            is_sig.append(sig)
            pos += 1
        group_spans.append(((group_start + pos - 1) / 2, group_name, group_start, pos - 1, shade_idx))
        pos += GAP

    fig, ax = plt.subplots(figsize=(9, 22))
    fig.suptitle("Phase 3 — Clinical Fusion Ablations (Single-Eye)",
                 fontsize=FSIZE + 3, fontweight="bold")
    fig.subplots_adjust(top=0.97, left=0.38)

    bp = ax.boxplot(box_data, vert=False, patch_artist=True, positions=positions,
                    widths=0.5, showfliers=True,
                    flierprops=dict(marker="o", markersize=3, alpha=0.5),
                    medianprops=dict(color=C_MEDIAN, linewidth=2),
                    manage_ticks=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    # Alternating shaded group backgrounds
    for x_mid, gname, g_start, g_end, shade_idx in group_spans:
        if shade_idx % 2 == 0:
            ax.axhspan(g_start - 0.5, g_end + 0.5, color="gray", alpha=0.07, zorder=0)
        # Group title to the left of the y-tick labels
        ax.text(-0.42, x_mid, gname, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=FSIZE - 1, fontweight="bold", color="#444444")

    ax.axvline(np.median(base_aucs), color=C_BASELINE, linewidth=1,
               linestyle="--", alpha=0.5, label="Baseline median")
    ax.set_yticks(positions)
    ax.set_yticklabels(tick_labels, fontsize=FSIZE - 1)
    for tick_lbl, sig in zip(ax.get_yticklabels(), is_sig):
        if sig:
            tick_lbl.set_fontweight("bold")
    ax.set_xlabel("Test AUC", fontsize=FSIZE + 1)
    ax.set_xlim(0.65, None)
    ax.set_ylim(-0.7, pos - GAP + 0.7)
    ax.invert_yaxis()  # baseline at top
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=FSIZE, loc="lower right")
    path = out_dir / "phase3_single_mode_ablations.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # Print top winners vs baseline
    print("\n  Top movers vs baseline (Wilcoxon, uncorrected):")
    deltas = []
    for run in aucs:
        if run == PHASE3_BASELINE:
            continue
        a = aucs[run]
        min_n = min(len(a), len(base_aucs))
        if min_n < 5:
            continue
        delta = np.mean(a) - np.mean(base_aucs)
        p = wilcoxon_p(a[:min_n], base_aucs[:min_n])
        deltas.append((run, delta, p))
    deltas.sort(key=lambda x: -x[1])
    for run, delta, p in deltas[:8]:
        print(f"    {run:30s}  {delta:+.3f}  p={p:.4f} {stars(p)}")

    # Pairwise table — IOP correction group (FDR-corrected Wilcoxon p-values)
    iop_runs   = list(PHASE3_GROUPS["IOP correction"].keys())
    iop_labels = list(PHASE3_GROUPS["IOP correction"].values())
    _, corr = paired_matrix(iop_runs, aucs)
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    import csv
    path2 = reports_dir / "phase3_iop_pairwise.csv"
    with open(path2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + iop_labels)
        for i, row_lbl in enumerate(iop_labels):
            cells = [row_lbl]
            for j in range(len(iop_labels)):
                p = corr[i, j]
                cells.append(f"{p:.4f} {stars(p)}" if not np.isnan(p) else "—")
            w.writerow(cells)
    print(f"  Saved: {path2}")


# ── Phase 4 ──────────────────────────────────────────────────────────────────

PHASE4_RUNS = {
    "single":             ("classic_test",  "Single-eye\n(baseline)"),
    "ensemble":           ("ensemble_test", "Ensemble\n(indep OD+OS)"),
    "bilateral":          ("bilat_test",    "BilateralHT\n(shared+concat)"),
    "siamese":            ("bilat_test",    "SiameseHT\n(mean+delta)"),
    "bilateral_loss_all": ("bilat_test",    "BilateralHT\nall-losses"),
    "siamese_loss_all":   ("bilat_test",    "SiameseHT\nall-losses"),
}
PHASE4_BASELINE = "single"


def analyze_phase4(out_dir: Path):
    print("\n=== Phase 4 ===")
    aucs = {}
    for run, (key, _) in PHASE4_RUNS.items():
        a = load_rep_aucs(RESULTS_ROOT / "phase4" / run, key)
        aucs[run] = a
        print(f"  {run:25s}  AUC={np.mean(a):.3f}±{np.std(a):.3f}  n={len(a)}")

    base_run  = PHASE4_BASELINE
    base_aucs = aucs[base_run]
    others = [(r, PHASE4_RUNS[r][1]) for r in PHASE4_RUNS if r != base_run]
    others.sort(key=lambda x: -np.mean(aucs[x[0]]) if len(aucs.get(x[0], [])) else float("inf"))
    ordered   = [(base_run, PHASE4_RUNS[base_run][1])] + others
    run_keys  = [r for r, _ in ordered]
    labels    = [lbl for _, lbl in ordered]
    data_list = [aucs[r] for r in run_keys]
    aucs_by_label = {lbl: aucs[r] for r, lbl in ordered}

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle("Phase 4 — Bilateral Architecture Comparison (Image Only)",
                 fontsize=FSIZE + 2, fontweight="bold")
    boxplot_panel(ax, data_list, labels, base_idx=0,
                  base_aucs=base_aucs, all_aucs_by_label=aucs_by_label)
    ax.set_ylim(0.75, None)

    fig.tight_layout()
    path = out_dir / "phase4_analysis.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Phase 5 ──────────────────────────────────────────────────────────────────

PHASE5_RUNS = {
    # "single_fused":        ("classic_test",   "Single-eye (baseline)"),
    "ensemble_fused":      ("ensemble_test",  "Ensemble (baseline)"),
    "bilateral_fused":     ("bilat_test",     "BilateralHT"),
    "siamese_fused":       ("bilat_test",     "SiameseHT"),
    # "ensemble_fused_head": ("ensemble_test",  "Ensemble\n+clinical+head"),
    "logit_mlp_head": ("ensemble_test",  "Ensemble\n+Fusion head"),
}
PHASE5_BASELINE = "ensemble_fused"


def analyze_phase5(out_dir: Path):
    print("\n=== Phase 5 ===")
    aucs = {}
    for run, (key, _) in PHASE5_RUNS.items():
        a = load_rep_aucs(RESULTS_ROOT / "phase5" / run, key)
        aucs[run] = a
        print(f"  {run:25s}  AUC={np.mean(a):.3f}±{np.std(a):.3f}  n={len(a)}")

    base_run  = PHASE5_BASELINE
    base_aucs = aucs[base_run]
    others = [(r, PHASE5_RUNS[r][1]) for r in PHASE5_RUNS if r != base_run]
    others.sort(key=lambda x: -np.mean(aucs[x[0]]) if len(aucs.get(x[0], [])) else float("inf"))
    ordered   = [(base_run, PHASE5_RUNS[base_run][1])] + others
    run_keys  = [r for r, _ in ordered]
    labels    = [lbl for _, lbl in ordered]
    data_list = [aucs[r] for r in run_keys]
    aucs_by_label = {lbl: aucs[r] for r, lbl in ordered}

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle("Phase 5 — Full HyperTower: Bilateral + Clinical",
                 fontsize=FSIZE + 2, fontweight="bold")
    boxplot_panel(ax, data_list, labels, base_idx=0,
                  base_aucs=base_aucs, all_aucs_by_label=aucs_by_label)
    ax.set_ylim(0.78, None)

    fig.tight_layout()
    path = out_dir / "phase5_analysis.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Cross-phase summary ───────────────────────────────────────────────────────

def analyze_cross_phase(out_dir: Path):
    """Single figure tracing the best model from each phase."""
    print("\n=== Cross-phase progression ===")
    trajectory = [
        ("Phase 2\nresnet50 proper",   "phase2", "imageonly_resnet50_proper",   "classic_test"),
        ("Phase 2\nrefugelike proper", "phase2", "imageonly_refugelike_proper",  "classic_test"),
        ("Phase 3\n+IOP ratio\n+drop raw", "phase3", "iop_ratio_drop_raw",     "classic_test"),
        ("Phase 4\nensemble\n(image only)", "phase4", "ensemble",               "ensemble_test"),
        ("Phase 5\nensemble\n+clinical",    "phase5", "ensemble_fused",         "ensemble_test"),
        ("Phase 5\nensemble\n+clinical+head","phase5","ensemble_fused_head",    "ensemble_test"),
    ]

    labels, means, stds, all_aucs = [], [], [], []
    for lbl, phase, run, key in trajectory:
        a = load_rep_aucs(RESULTS_ROOT / phase / run, key)
        labels.append(lbl)
        means.append(np.mean(a) if len(a) else np.nan)
        stds.append(np.std(a) if len(a) else np.nan)
        all_aucs.append(a)
        print(f"  {lbl.replace(chr(10),' '):35s}  AUC={means[-1]:.3f}±{stds[-1]:.3f}  n={len(a)}")

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(labels))
    ax.errorbar(x, means, yerr=stds, fmt="o-", linewidth=2, markersize=7,
                capsize=4, color="#4c72b0")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Test AUC (10-rep mean ± std)")
    ax.set_title("HyperTower — Model Progression Across Phases", fontsize=12, fontweight="bold")
    ax.set_ylim(0.75, 0.95)
    ax.axhline(means[0], color="gray", linewidth=1, linestyle=":", alpha=0.5, label="Phase 2 baseline")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = out_dir / "cross_phase_progression.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5],
                    help="Run only this phase (default: all)")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[3] / "v3" / "figures",
                    help="Output directory for figures")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    run_all = args.phase is None
    if run_all or args.phase == 1:
        analyze_phase1(args.out)
    if run_all or args.phase == 2:
        analyze_phase2(args.out)
    if run_all or args.phase == 3:
        analyze_phase3(args.out)
    if run_all or args.phase == 4:
        analyze_phase4(args.out)
    if run_all or args.phase == 5:
        analyze_phase5(args.out)
    if run_all:
        analyze_cross_phase(args.out)


if __name__ == "__main__":
    main()
