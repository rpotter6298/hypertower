"""F3b - Hadamard L1 fusion ROC: patient-grouped vs anonymous CV (overlay).

Single panel overlaying the ROC of the same single-eye img+cd Hadamard L1
fusion configuration evaluated under patient-grouped 5-fold CV (the headline
protocol) and patient-anonymous 5-fold CV (the prevailing benchmark protocol).
Both runs use matched seeds, matched backbones, matched training schedules;
only the fold-grouping rule changes.

For each configuration the figure shows:
  - per-fold-rep ROC curves as faint coloured lines (50 curves per condition)
  - mean ROC across fold-reps with a shaded SD band
  - the rep-mean test AUC ± SD in a corner annotation

Re-run:
    python -m v4.figures.F3b_anonymous_cv_roc
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from v4.figures.util.loaders import RESULTS_ROOT


OUT = Path(__file__).parent / "output" / "F3b_anonymous_cv_roc.png"

CONDITIONS = [
    ("Patient-grouped CV",
     RESULTS_ROOT / "refuge_v2m_baseline" / "ensemble_single_refugelike",
     "nt", "#1f6fb0"),
    ("Anonymous CV",
     RESULTS_ROOT / "backbone_replication" / "anonymous_cv_ensemble_single_refugelike",
     "nt", "#c44e52"),
]

# Common FPR grid for per-fold-rep ROC interpolation
FPR_GRID = np.linspace(0.0, 1.0, 201)


def collect_per_foldrep(run_dir: Path, eval_stage: str):
    """Return list of (y_true, y_score) tuples, one per (rep, fold)."""
    per: list[tuple[np.ndarray, np.ndarray]] = []
    for rep in sorted(run_dir.glob("rep*")):
        fp = next(iter(rep.rglob("predictions.h5")), None)
        if fp is None:
            continue
        with h5py.File(fp, "r") as f:
            if eval_stage not in f:
                continue
            grp = f[eval_stage]
            logits = grp["logits"][:]
            y_true = grp["y_true"][:].astype(int)
            split = grp["split"][:]
            n_folds, n_epochs, _, n_heads, n_outputs = logits.shape
            if n_outputs != 2:
                continue
            ep, head = n_epochs - 1, n_heads - 1
            for fold in range(n_folds):
                labels = np.array(
                    [s.decode() if isinstance(s, bytes) else str(s) for s in split[fold]]
                )
                test_mask = labels == "test"
                if not test_mask.any():
                    continue
                lg = logits[fold, ep, test_mask, head, :]
                e = np.exp(lg - lg.max(axis=1, keepdims=True))
                p = e / e.sum(axis=1, keepdims=True)
                y = y_true[test_mask]
                s = p[:, 1]
                per.append((y, s))
    return per


def interp_tpr(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    if len(np.unique(y)) < 2:
        return np.full_like(FPR_GRID, np.nan, dtype=float)
    fpr, tpr, _ = roc_curve(y, s)
    return np.interp(FPR_GRID, fpr, tpr)


def render() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 6.4))

    # Diagonal reference first so it sits behind everything
    ax.plot([0, 1], [0, 1], color="#aaa", linewidth=0.8, linestyle=":", zorder=1)

    annotations = []

    for label, run_dir, stage, color in CONDITIONS:
        per = collect_per_foldrep(run_dir, stage)
        if not per:
            continue

        tprs = np.array([interp_tpr(y, s) for (y, s) in per])
        valid = ~np.isnan(tprs).any(axis=1)
        tprs = tprs[valid]

        per_aucs = np.array(
            [roc_auc_score(y, s) for (y, s) in per if len(np.unique(y)) >= 2]
        )
        rep_means = (per_aucs.reshape(-1, 5).mean(axis=1)
                     if len(per_aucs) % 5 == 0 else per_aucs)

        # Per-fold-rep curves
        for (y, s) in per:
            if len(np.unique(y)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(fpr, tpr, color=color, linewidth=0.4, alpha=0.13, zorder=2)

        # Mean ± SD band
        mean_tpr = tprs.mean(axis=0)
        sd_tpr = tprs.std(axis=0)
        ax.fill_between(
            FPR_GRID, np.clip(mean_tpr - sd_tpr, 0, 1),
            np.clip(mean_tpr + sd_tpr, 0, 1),
            color=color, alpha=0.20, zorder=3,
        )
        ax.plot(
            FPR_GRID, mean_tpr, color=color, linewidth=2.2, zorder=4,
            label=f"{label}  (AUC = {rep_means.mean():.3f} ± {rep_means.std():.3f})",
        )

        annotations.append((label, rep_means))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(
        "Hadamard L1 fusion ROC under matched architecture,\n"
        "patient-grouped vs anonymous cross-validation",
        fontsize=12, fontweight="bold",
    )
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.94)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}")


if __name__ == "__main__":
    render()
