#!/usr/bin/env python3
"""
Aggregate and visualise results from a 10× repeated 5-fold CV run.

Reads probs / y_true from every rep/fold directory, computes per-fold
metrics, and produces:

  outputs/
    fold_metrics.csv          — one row per (rep, fold, eval_mode)
    rep_metrics.csv           — one row per (rep, eval_mode): mean over 5 folds
    overall_summary.txt       — mean ± SD and 95% CI printed to console + file
    {eval_mode}_auc_violin.png
    {eval_mode}_roc_mean.png  — mean ± 1 SD OVR ROC (all classes or class 1)
    {eval_mode}_holdout_roc_mean.png

Holdout metrics are extracted from the rep-level predictions.npz using the
best_epoch recorded in summary.json, ensemble-averaged over od_fused + os_fused
heads, giving 50 fold-level holdout AUC values (5 folds × 10 reps).

Usage
-----
    python scripts/output_analysis/aggregate_10x5cv.py \
        --run-root analysis_data/pipeline_10x5 \
        --eval-modes binary multiclass \
        --out analysis_data/pipeline_10x5/aggregate
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
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc as sk_auc

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CLASS_NAMES = {
    "binary":     ["Healthy", "Glaucoma"],
    "multiclass": ["Healthy", "Glaucoma", "Suspect"],
}

# Probe files in preference order (first found wins)
# probs_fused = simple OD/OS softmax average (ensemble head — primary metric)
# probs_fused_head = learned logit-level fusion head (worse on average; kept as fallback)
_PROBS_PRIORITY = ["probs_fused.npy", "probs_fused_head.npy"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_probs(fold_dir: Path) -> Path | None:
    for name in _PROBS_PRIORITY:
        p = fold_dir / name
        if p.exists():
            return p
    return None


def _load_holdout_summary(mode_dir: Path) -> dict | None:
    """
    Read rep-level holdout metrics from summary.json.

    Returns dict with keys auc_mean, auc_std, acc_mean (may be None if missing).
    """
    summary_path = mode_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text())
        return summary.get("mode_summary", {}).get("ensemble_holdout")
    except Exception:
        return None


def _auc_macro(y: np.ndarray, p: np.ndarray, num_classes: int) -> float:
    try:
        if num_classes == 2:
            return float(roc_auc_score(y, p[:, 1]))
        return float(roc_auc_score(y, p, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")


def _per_class_roc(y: np.ndarray, p: np.ndarray) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for k in range(p.shape[1]):
        yb = (y == k).astype(np.uint8)
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        fpr, tpr, _ = roc_curve(yb, p[:, k])
        out[k] = {"fpr": fpr, "tpr": tpr, "auc": sk_auc(fpr, tpr)}
    return out


def _ci95(values: np.ndarray) -> tuple[float, float]:
    """95% CI via t-distribution (two-sided)."""
    from scipy import stats as scipy_stats
    if len(values) < 2:
        return (float("nan"), float("nan"))
    ci = scipy_stats.t.interval(0.95, df=len(values) - 1,
                                 loc=np.mean(values), scale=scipy_stats.sem(values))
    return float(ci[0]), float(ci[1])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_folds(run_root: Path, eval_modes: list[str]) -> pd.DataFrame:
    rows = []
    rep_dirs = sorted(
        [d for d in run_root.iterdir() if d.is_dir() and d.name.startswith("rep")],
        key=lambda d: d.name,
    )
    if not rep_dirs:
        raise SystemExit(f"No rep* directories found in {run_root}")

    for rep_dir in rep_dirs:
        for eval_mode in eval_modes:
            tower_mode = "ensemble"
            mode_dir = rep_dir / eval_mode / tower_mode
            if not mode_dir.exists():
                print(f"  [skip] {mode_dir} not found")
                continue
            num_classes = 2 if eval_mode == "binary" else 3

            fold_dirs = sorted(
                [d for d in mode_dir.iterdir() if d.is_dir() and d.name.startswith("fold")],
                key=lambda d: int(d.name[4:]),
            )
            for fold_dir in fold_dirs:
                fold_idx = int(fold_dir.name[4:])
                y_path = fold_dir / "y_true.npy"
                p_path = _find_probs(fold_dir)

                if y_path is None or not y_path.exists() or p_path is None:
                    print(f"  [skip] {rep_dir.name}/{eval_mode}/fold{fold_idx}: missing files")
                    continue

                y = np.load(y_path)
                p = np.load(p_path)

                if eval_mode == "binary":
                    mask = np.isin(y, [0, 1])
                    y, p = y[mask], p[mask]
                    if p.shape[1] > 2:
                        p = p[:, :2]

                auc_macro = _auc_macro(y, p, num_classes)
                acc = float(accuracy_score(y, p.argmax(1)))

                row = {
                    "rep":        rep_dir.name,
                    "fold":       fold_idx,
                    "eval_mode":  eval_mode,
                    "probs_file": p_path.name,
                    "auc_macro":  auc_macro,
                    "acc":        acc,
                    "n":          len(y),
                }

                # Per-class AUC
                for k in range(num_classes):
                    yb = (y == k).astype(np.uint8)
                    if yb.sum() > 0 and yb.sum() < len(yb):
                        try:
                            row[f"auc_class{k}"] = float(roc_auc_score(yb, p[:, k]))
                        except Exception:
                            row[f"auc_class{k}"] = float("nan")
                    else:
                        row[f"auc_class{k}"] = float("nan")

                rows.append(row)

            # ---- holdout metrics from rep-level summary.json ----
            # Holdout probs are not stored per-fold; only aggregated stats are saved.
            # We attach the rep-level mean to each fold row (same value repeated),
            # and also add a single rep-level summary row (fold=-1).
            hld_summary = _load_holdout_summary(mode_dir)
            if hld_summary:
                hld_auc = hld_summary.get("auc_mean", float("nan"))
                hld_auc_std = hld_summary.get("auc_std", float("nan"))
                hld_acc = hld_summary.get("acc_mean", float("nan"))
                for row in rows:
                    if row["rep"] == rep_dir.name and row["eval_mode"] == eval_mode:
                        row["hld_auc_macro"] = hld_auc
                        row["hld_acc"] = hld_acc
                # Also store a rep-level holdout row (fold=-1) for direct rep-level analysis
                rows.append({
                    "rep": rep_dir.name,
                    "fold": -1,
                    "eval_mode": eval_mode,
                    "probs_file": "summary.json",
                    "auc_macro": float("nan"),
                    "acc": float("nan"),
                    "n": float("nan"),
                    "hld_auc_macro": hld_auc,
                    "hld_auc_std_within_rep": hld_auc_std,
                    "hld_acc": hld_acc,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _violin(fold_df: pd.DataFrame, eval_mode: str, out_dir: Path) -> None:
    sub = fold_df[fold_df["eval_mode"] == eval_mode].copy()
    num_classes = 2 if eval_mode == "binary" else 3
    class_names = _CLASS_NAMES[eval_mode]

    auc_cols = ["auc_macro"] + [f"auc_class{k}" for k in range(num_classes)]
    labels   = ["Macro AUC"] + [f"AUC {class_names[k]}" for k in range(num_classes)]
    present  = [(c, l) for c, l in zip(auc_cols, labels) if c in sub.columns]

    data   = [sub[c].dropna().values for c, _ in present]
    labels = [l for _, l in present]

    fig, ax = plt.subplots(figsize=(max(6, 2 * len(data)), 5))
    parts = ax.violinplot(data, showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.7)

    # Overlay individual rep means
    rep_means = sub.groupby("rep")[auc_cols[0]].mean().values
    ax.scatter(np.ones(len(rep_means)), rep_means, zorder=3,
               color="k", s=18, alpha=0.6, label="rep mean")

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("AUC")
    ax.set_title(f"AUC distribution — {eval_mode}  (10 × 5-fold, n={len(sub)})")
    ax.set_ylim(max(0, sub[auc_cols[0]].min() - 0.05), 1.02)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / f"{eval_mode}_auc_violin.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _mean_roc(fold_df: pd.DataFrame, eval_mode: str,
              run_root: Path, out_dir: Path) -> None:
    """Mean ± 1 SD OVR ROC across all 50 folds."""
    sub = fold_df[fold_df["eval_mode"] == eval_mode]
    num_classes = 2 if eval_mode == "binary" else 3
    class_names = _CLASS_NAMES[eval_mode]

    # Classes to plot (binary: class 1 only)
    plot_classes = [1] if eval_mode == "binary" else list(range(num_classes))

    grid = np.linspace(0, 1, 501)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="grey")

    for k in plot_classes:
        tprs, aucs = [], []
        for _, row in sub.iterrows():
            rep_dir  = run_root / row["rep"]
            fold_dir = rep_dir / eval_mode / "ensemble" / f"fold{int(row['fold'])}"
            y_path   = fold_dir / "y_true.npy"
            p_path   = _find_probs(fold_dir)
            if not y_path.exists() or p_path is None:
                continue
            y = np.load(y_path)
            p = np.load(p_path)
            if eval_mode == "binary":
                mask = np.isin(y, [0, 1])
                y, p = y[mask], p[mask]
                if p.shape[1] > 2:
                    p = p[:, :2]
            yb = (y == k).astype(np.uint8)
            if yb.sum() == 0 or yb.sum() == len(yb):
                continue
            fpr, tpr, _ = roc_curve(yb, p[:, k])
            tprs.append(np.interp(grid, fpr, tpr))
            aucs.append(sk_auc(fpr, tpr))

        if not tprs:
            continue
        arr  = np.vstack(tprs)
        mean = arr.mean(0)
        std  = arr.std(0)
        cname = class_names[k]
        lbl   = f"{cname}  AUC {np.nanmean(aucs):.3f} ± {np.nanstd(aucs):.3f}"
        line, = ax.plot(grid, mean, linewidth=2, label=lbl)
        ax.fill_between(grid,
                        np.clip(mean - std, 0, 1),
                        np.clip(mean + std, 0, 1),
                        alpha=0.15, color=line.get_color())

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Mean ± 1 SD OVR ROC — {eval_mode}  (10 × 5-fold)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    path = out_dir / f"{eval_mode}_roc_mean.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _holdout_stability(fold_df: pd.DataFrame, eval_mode: str, out_dir: Path) -> None:
    """Bar chart of per-rep holdout AUC (mean across folds within rep ± within-rep SD)."""
    # use the rep-level rows (fold == -1) which have hld_auc_std_within_rep
    rep_rows = fold_df[(fold_df["eval_mode"] == eval_mode) & (fold_df["fold"] == -1)].copy()
    if rep_rows.empty or "hld_auc_macro" not in rep_rows.columns:
        print(f"  [skip] no holdout data for {eval_mode}")
        return
    rep_rows = rep_rows.sort_values("rep")

    fig, ax = plt.subplots(figsize=(max(6, len(rep_rows) * 0.9), 4))
    x = np.arange(len(rep_rows))
    yerr = rep_rows.get("hld_auc_std_within_rep", pd.Series([0]*len(rep_rows))).fillna(0).values
    ax.bar(x, rep_rows["hld_auc_macro"].values, yerr=yerr,
           capsize=4, color="darkorange", alpha=0.8)
    grand_mean = rep_rows["hld_auc_macro"].mean()
    ax.axhline(grand_mean, linestyle="--", color="crimson",
               linewidth=1.2, label=f"grand mean = {grand_mean:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(rep_rows["rep"].values, rotation=30, ha="right")
    ax.set_ylabel("Holdout macro AUC (mean ± within-rep SD)")
    ax.set_title(f"Per-rep holdout stability — {eval_mode}")
    ymin = max(0, rep_rows["hld_auc_macro"].min() - 0.05)
    ax.set_ylim(ymin, 1.02)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    path = out_dir / f"{eval_mode}_holdout_stability.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _rep_stability(fold_df: pd.DataFrame, eval_mode: str, out_dir: Path) -> None:
    """Bar chart of per-rep mean macro AUC with ± 1 SD error bars."""
    sub = fold_df[fold_df["eval_mode"] == eval_mode]
    rep_stats = sub.groupby("rep")["auc_macro"].agg(["mean", "std"]).reset_index()
    rep_stats = rep_stats.sort_values("rep")

    fig, ax = plt.subplots(figsize=(max(6, len(rep_stats) * 0.9), 4))
    x = np.arange(len(rep_stats))
    ax.bar(x, rep_stats["mean"], yerr=rep_stats["std"],
           capsize=4, color="steelblue", alpha=0.8)
    ax.axhline(rep_stats["mean"].mean(), linestyle="--", color="crimson",
               linewidth=1.2, label=f"grand mean = {rep_stats['mean'].mean():.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(rep_stats["rep"], rotation=30, ha="right")
    ax.set_ylabel("Mean macro AUC (5 folds)")
    ax.set_title(f"Per-rep stability — {eval_mode}")
    ymin = max(0, rep_stats["mean"].min() - rep_stats["std"].max() - 0.02)
    ax.set_ylim(ymin, 1.02)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    path = out_dir / f"{eval_mode}_rep_stability.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Summary text
# ---------------------------------------------------------------------------

def _print_summary(fold_df: pd.DataFrame, eval_modes: list[str]) -> str:
    lines = ["=" * 60, "10 × 5-fold CV — aggregate summary", "=" * 60]
    for eval_mode in eval_modes:
        sub = fold_df[(fold_df["eval_mode"] == eval_mode) & (fold_df["fold"] >= 0)]
        if sub.empty:
            continue
        num_classes = 2 if eval_mode == "binary" else 3
        class_names = _CLASS_NAMES[eval_mode]
        lines.append(f"\n--- {eval_mode.upper()} ---")
        lines.append(f"  n_folds = {len(sub)}")

        lines.append("  [Validation]")
        for col, label in [("auc_macro", "Macro AUC"), ("acc", "Accuracy")]:
            if col not in sub.columns:
                continue
            vals = sub[col].dropna().values
            ci_lo, ci_hi = _ci95(vals)
            lines.append(
                f"  {label:18s}: {vals.mean():.4f} ± {vals.std():.4f}"
                f"  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]"
            )

        for k in range(num_classes):
            col = f"auc_class{k}"
            if col not in sub.columns:
                continue
            vals = sub[col].dropna().values
            if len(vals) == 0:
                continue
            ci_lo, ci_hi = _ci95(vals)
            lines.append(
                f"  AUC {class_names[k]:12s}: {vals.mean():.4f} ± {vals.std():.4f}"
                f"  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]"
            )

        # Between-rep variance (val)
        rep_means = sub.groupby("rep")["auc_macro"].mean().values
        lines.append(
            f"  Rep-mean AUC (n={len(rep_means)}): "
            f"{rep_means.mean():.4f} ± {rep_means.std():.4f}"
            f"  (between-rep SD = {rep_means.std():.4f})"
        )

        # Holdout — use rep-level rows (fold == -1)
        rep_hld = fold_df[
            (fold_df["eval_mode"] == eval_mode) &
            (fold_df["fold"] == -1) &
            fold_df["hld_auc_macro"].notna()
        ]["hld_auc_macro"].values if "hld_auc_macro" in fold_df.columns else np.array([])

        if len(rep_hld) > 0:
            lines.append("  [Holdout]  (rep-level means, n_reps={})".format(len(rep_hld)))
            ci_lo, ci_hi = _ci95(rep_hld)
            lines.append(
                f"  {'Macro AUC':18s}: {rep_hld.mean():.4f} ± {rep_hld.std():.4f}"
                f"  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]"
            )

    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--run-root", default="analysis_data/pipeline_10x5",
                    help="Root directory containing rep* sub-directories.")
    ap.add_argument("--eval-modes", nargs="+",
                    choices=["binary", "multiclass"],
                    default=["binary", "multiclass"])
    ap.add_argument("--out", default=None,
                    help="Output directory for plots and CSVs "
                         "(default: {run-root}/aggregate).")
    args = ap.parse_args()

    run_root = Path(args.run_root)
    out_dir  = Path(args.out) if args.out else run_root / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading fold data...")
    fold_df = load_all_folds(run_root, args.eval_modes)
    if fold_df.empty:
        raise SystemExit("No data loaded — check --run-root.")

    fold_df.to_csv(out_dir / "fold_metrics.csv", index=False)
    print(f"  Saved fold_metrics.csv ({len(fold_df)} rows)")

    rep_df = (fold_df.groupby(["rep", "eval_mode"])
              [["auc_macro", "acc"] +
               [c for c in fold_df.columns if c.startswith("auc_class")]]
              .mean()
              .reset_index())
    rep_df.to_csv(out_dir / "rep_metrics.csv", index=False)
    print(f"  Saved rep_metrics.csv ({len(rep_df)} rows)")

    summary_text = _print_summary(fold_df, args.eval_modes)
    print("\n" + summary_text)
    (out_dir / "overall_summary.txt").write_text(summary_text + "\n")
    print(f"\n  Saved overall_summary.txt")

    print("\nGenerating plots...")
    for eval_mode in args.eval_modes:
        if fold_df[fold_df["eval_mode"] == eval_mode].empty:
            continue
        _violin(fold_df, eval_mode, out_dir)
        _mean_roc(fold_df, eval_mode, run_root, out_dir)
        _rep_stability(fold_df, eval_mode, out_dir)
        _holdout_stability(fold_df, eval_mode, out_dir)

    print(f"\nAll outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
