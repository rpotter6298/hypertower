"""3-bin severity confusion matrix for vf_md regression heads.

Reads predictions.h5 files from a regression run, bins both actual and predicted
MD into 3 severity tiers (severe / moderate / no-problem), and reports:

  * confusion matrix (counts and per-row %)
  * exact-bin & adjacent-bin accuracy
  * per-bin recall
  * binary "no-problem vs disease" sensitivity/specificity at the −4.5 dB boundary
  * optional saved heatmap PNG

Bin boundaries (placed halfway between integer bins, matching plot_regression_predictions):
  severe       :  vf_md <= -9.5
  moderate     : -9.5 < vf_md <= -4.5
  no-problem   : vf_md > -4.5

Usage:
    python -m v4.scripts.analysis.severity_confusion \\
        v4/results/experiments/reg_head/baseline_reg_nt50 \\
        --save-fig analysis/figures/regression_severity_confusion.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

LABELS = ["severe (≤−10)", "moderate (−9..−5)", "no-problem (≥−4)"]


def severity_3bin(values: np.ndarray) -> np.ndarray:
    bins = np.full(values.shape, -1, dtype=int)
    bins[values <= -9.5] = 0
    bins[(values > -9.5) & (values <= -4.5)] = 1
    bins[values > -4.5] = 2
    return bins


def _decode(arr) -> np.ndarray:
    return np.array([s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in arr])


def collect_test_predictions(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    with h5py.File(path, "r") as f:
        if "hb" not in f:
            return None
        grp = f["hb"]
        logits = grp["logits"][:]
        y_true = grp["y_true"][:].astype(float)
        split  = grp["split"][:]
        n_folds, n_epochs, _, n_heads, _ = logits.shape
        ep, head, out = n_epochs - 1, n_heads - 1, 0
        actuals: list[np.ndarray] = []
        preds:   list[np.ndarray] = []
        for fold in range(n_folds):
            labels = _decode(split[fold])
            mask = (labels == "test") & np.isfinite(y_true) & np.isfinite(logits[fold, ep, :, head, out])
            if not mask.any():
                continue
            actuals.append(y_true[mask])
            preds.append(logits[fold, ep, mask, head, out].astype(float))
    if not actuals:
        return None
    return np.concatenate(actuals), np.concatenate(preds)


def report(actuals: np.ndarray, preds: np.ndarray) -> dict:
    ab = severity_3bin(actuals)
    pb = severity_3bin(preds)
    valid = (ab >= 0) & (pb >= 0)
    ab, pb = ab[valid], pb[valid]
    cm = np.zeros((3, 3), dtype=int)
    for x, y in zip(ab, pb):
        cm[x, y] += 1

    # Binary disease vs no-problem (bins 0+1 vs bin 2)
    actual_disease = ab <= 1
    pred_disease   = pb <= 1
    tp = int(np.sum(actual_disease & pred_disease))
    tn = int(np.sum(~actual_disease & ~pred_disease))
    fp = int(np.sum(~actual_disease & pred_disease))
    fn = int(np.sum(actual_disease & ~pred_disease))
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)

    return {
        "confusion":     cm,
        "n_test":        int(ab.size),
        "exact_acc":     float(np.mean(ab == pb)),
        "adjacent_acc":  float(np.mean(np.abs(ab - pb) <= 1)),
        "recall_per_bin": [float(np.mean(pb[ab == i] == i)) if (ab == i).any() else float("nan")
                            for i in range(3)],
        "n_per_bin":     [int((ab == i).sum()) for i in range(3)],
        "binary_sens":   sens,
        "binary_spec":   spec,
        "binary_tp":     tp,
        "binary_fp":     fp,
        "binary_fn":     fn,
        "binary_tn":     tn,
    }


def print_report(r: dict) -> None:
    cm = r["confusion"]
    print(f"\nn_test (pooled across reps × folds): {r['n_test']}")
    print(f"\nConfusion matrix (rows = actual, cols = predicted):")
    print(f"{'actual \\ pred':<22s}  {LABELS[0]:>16s}  {LABELS[1]:>20s}  {LABELS[2]:>18s}    n")
    for i in range(3):
        row = cm[i]
        print(f"{LABELS[i]:<22s}  {row[0]:>16d}  {row[1]:>20d}  {row[2]:>18d}    {row.sum()}")
    print(f"\nExact-bin accuracy:    {r['exact_acc']:.3f}")
    print(f"Adjacent-bin accuracy: {r['adjacent_acc']:.3f}")
    print("\nPer-bin recall:")
    for i in range(3):
        n = r["n_per_bin"][i]
        rec = r["recall_per_bin"][i]
        print(f"  {LABELS[i]:<22s}  n={n:>4d}  recall={rec:.3f}")
    print(f"\nBinary disease (severe+moderate) vs no-problem, threshold = −4.5 dB:")
    print(f"  sensitivity (correctly flag disease):   {r['binary_sens']:.3f}  ({r['binary_tp']}/{r['binary_tp']+r['binary_fn']})")
    print(f"  specificity (correctly clear healthy):  {r['binary_spec']:.3f}  ({r['binary_tn']}/{r['binary_tn']+r['binary_fp']})")


def save_heatmap(r: dict, path: Path) -> None:
    import matplotlib.pyplot as plt
    cm = r["confusion"]
    cm_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i,j]}\n({cm_pct[i,j]*100:.1f}%)",
                    ha="center", va="center",
                    color="white" if cm_pct[i,j] > 0.5 else "black",
                    fontsize=10)
    ax.set_xticks(range(3)); ax.set_xticklabels(LABELS, rotation=20, ha="right")
    ax.set_yticks(range(3)); ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"VF-MD severity confusion (n={r['n_test']})\n"
                 f"exact={r['exact_acc']:.3f}  adjacent={r['adjacent_acc']:.3f}  "
                 f"sens={r['binary_sens']:.3f}  spec={r['binary_spec']:.3f}")
    fig.colorbar(im, ax=ax, label="Row-normalised fraction")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved heatmap: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="A predictions.h5 file or a directory containing them")
    ap.add_argument("--save-fig", type=Path, default=None, help="Optional path to save the confusion-matrix heatmap PNG")
    args = ap.parse_args()

    if args.path.is_file():
        files = [args.path]
    else:
        files = sorted(args.path.rglob("predictions.h5"))
    if not files:
        raise SystemExit(f"No predictions.h5 under {args.path}")

    all_actual: list[np.ndarray] = []
    all_pred:   list[np.ndarray] = []
    for fp in files:
        res = collect_test_predictions(fp)
        if res is None:
            print(f"  skipped (no hb predictions): {fp}")
            continue
        a, p = res
        all_actual.append(a); all_pred.append(p)
    if not all_actual:
        raise SystemExit("No usable predictions found")

    actuals = np.concatenate(all_actual)
    preds   = np.concatenate(all_pred)
    print(f"Pooled across {len(all_actual)} predictions.h5 files")
    r = report(actuals, preds)
    print_report(r)
    if args.save_fig:
        save_heatmap(r, args.save_fig)


if __name__ == "__main__":
    main()
