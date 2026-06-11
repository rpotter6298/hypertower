"""Plot regression predictions against actual values from v4 predictions.h5.

Usage:
    python -m v4.scripts.analysis.plot_regression_predictions \
        v4/results/experiments/reg_head/baseline_reg

    python -m v4.scripts.analysis.plot_regression_predictions \
        v4/results/experiments/reg_head/baseline_reg/rep00/binary/predictions.h5 \
        --split val --phase hb --head hb_head --glaucoma-only
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _decode_array(arr: np.ndarray) -> np.ndarray:
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in arr])


def _head_names(grp: h5py.Group) -> list[str]:
    return [v.decode() if isinstance(v, bytes) else str(v) for v in grp["head_names"][:]]


def _find_prediction_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.name != "predictions.h5":
            raise SystemExit(f"Expected predictions.h5 file, got: {path}")
        return [path]
    if not path.is_dir():
        raise SystemExit(f"Not a file or directory: {path}")
    files = sorted(path.rglob("predictions.h5"))
    if not files:
        raise SystemExit(f"No predictions.h5 files found under: {path}")
    return files


def _choose_phase(f: h5py.File, phase: str | None) -> str:
    phases = sorted(k for k in f.keys() if isinstance(f[k], h5py.Group) and "logits" in f[k])
    if not phases:
        raise ValueError("No phase groups containing a logits dataset were found")
    if phase is None:
        return phases[0]
    if phase not in phases:
        raise ValueError(f"Phase {phase!r} not found. Available phases: {', '.join(phases)}")
    return phase


def _choose_head(grp: h5py.Group, head: str | None) -> tuple[str, int]:
    names = _head_names(grp)
    if not names:
        raise ValueError("No head names found")
    if head is None:
        return names[0], 0
    if head not in names:
        raise ValueError(f"Head {head!r} not found. Available heads: {', '.join(names)}")
    return head, names.index(head)


def _collect_points(
    path: Path,
    *,
    phase: str | None,
    head: str | None,
    split: str,
    epoch: int,
    output_index: int,
    glaucoma_only: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    with h5py.File(path, "r") as f:
        phase_name = _choose_phase(f, phase)
        grp = f[phase_name]
        head_name, head_idx = _choose_head(grp, head)

        logits = grp["logits"]
        n_folds, n_epochs, n_samples, _, n_outputs = logits.shape
        epoch_idx = epoch if epoch >= 0 else n_epochs + epoch
        if not 0 <= epoch_idx < n_epochs:
            raise ValueError(f"Epoch {epoch} is out of range for {n_epochs} epochs")
        if not 0 <= output_index < n_outputs:
            raise ValueError(f"Output index {output_index} is out of range for {n_outputs} outputs")

        y_true_all = grp["y_true"][:].astype(float)
        split_all = grp["split"][:]

        actuals: list[np.ndarray] = []
        preds: list[np.ndarray] = []
        folds: list[np.ndarray] = []
        for fold in range(n_folds):
            labels = _decode_array(split_all[fold])
            mask = np.ones(n_samples, dtype=bool) if split == "all" else labels == split
            if glaucoma_only:
                mask &= y_true_all != 0
            y_pred = logits[fold, epoch_idx, :, head_idx, output_index].astype(float)
            finite = mask & np.isfinite(y_true_all) & np.isfinite(y_pred)
            if not finite.any():
                continue
            actuals.append(y_true_all[finite])
            preds.append(y_pred[finite])
            folds.append(np.full(int(finite.sum()), fold, dtype=int))

    if not actuals:
        suffix = " and y_true != 0" if glaucoma_only else ""
        raise ValueError(f"No finite rows found for split={split!r}{suffix}")
    return (
        np.concatenate(actuals),
        np.concatenate(preds),
        np.concatenate(folds),
        phase_name,
        head_name,
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_pred - y_true
    mse = float(np.mean(residual ** 2))
    abs_residual = np.abs(residual)
    out = {
        "n": float(y_true.size),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(abs_residual)),
        "bias": float(np.mean(residual)),
        "within_1": float(np.mean(abs_residual <= 1.0)),
        "within_3": float(np.mean(abs_residual <= 3.0)),
        "within_5": float(np.mean(abs_residual <= 5.0)),
    }
    if y_true.size > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        out["r"] = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        out["r"] = float("nan")
    return out


BIN_LABELS = ["<=-10", "-9..-5", "-4..-1", ">=1"]


def _severity_bin(values: np.ndarray) -> np.ndarray:
    """Map continuous values onto glaucoma severity bins.

    Actual labels use integer bins: <=-10, -9..-5, -4..-1, >=1.
    Predictions are continuous, so boundaries are placed halfway between
    adjacent integer bins: -9.5, -4.5, and 0.0.
    """
    bins = np.full(values.shape, -1, dtype=int)
    bins[values <= -9.5] = 0
    bins[(values > -9.5) & (values <= -4.5)] = 1
    bins[(values > -4.5) & (values <= 0.0)] = 2
    bins[values > 0.0] = 3
    return bins


def _bin_summary_lines(y_true: np.ndarray, y_pred: np.ndarray) -> list[str]:
    actual_bins = _severity_bin(y_true)
    pred_bins = _severity_bin(y_pred)
    valid = (actual_bins >= 0) & (pred_bins >= 0)
    if not valid.any():
        return ["Bin summary: no rows matched the configured severity bins"]

    actual_bins = actual_bins[valid]
    pred_bins = pred_bins[valid]
    confusion = np.zeros((len(BIN_LABELS), len(BIN_LABELS)), dtype=int)
    for actual, pred in zip(actual_bins, pred_bins):
        confusion[actual, pred] += 1

    exact = float(np.mean(actual_bins == pred_bins))
    adjacent = float(np.mean(np.abs(actual_bins - pred_bins) <= 1))
    lines = [
        f"Bin summary: exact={exact:.3f}  within_adjacent={adjacent:.3f}",
        "  actual/pred      " + "  ".join(f"{label:>8s}" for label in BIN_LABELS),
    ]
    for idx, label in enumerate(BIN_LABELS):
        row = confusion[idx]
        n = int(row.sum())
        row_text = "  ".join(f"{v:8d}" for v in row)
        lines.append(f"  {label:>11s} n={n:3d}  {row_text}")

    lines.append("  per-actual-bin:")
    for idx, label in enumerate(BIN_LABELS):
        mask = actual_bins == idx
        if not mask.any():
            continue
        stats = _metrics(y_true[valid][mask], y_pred[valid][mask])
        lines.append(
            f"    {label:>7s} n={stats['n']:.0f} "
            f"mean_pred={float(np.mean(y_pred[valid][mask])):.2f} "
            f"bias={stats['bias']:.2f} mae={stats['mae']:.2f} rmse={stats['rmse']:.2f} "
            f"within_3={stats['within_3']:.3f} within_5={stats['within_5']:.3f}"
        )
    return lines


def _plot_one(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    folds: np.ndarray,
    *,
    title: str,
    output: Path,
    dpi: int,
) -> None:
    stats = _metrics(y_true, y_pred)
    residual = y_pred - y_true

    lo = float(np.nanmin([y_true.min(), y_pred.min()]))
    hi = float(np.nanmax([y_true.max(), y_pred.max()]))
    pad = max((hi - lo) * 0.05, 1.0)
    lo -= pad
    hi += pad

    fig, (ax_scatter, ax_resid) = plt.subplots(
        1,
        2,
        figsize=(11.5, 5.0),
        gridspec_kw={"width_ratios": [1.4, 1.0]},
        constrained_layout=True,
    )

    scatter = ax_scatter.scatter(
        y_true,
        y_pred,
        c=folds,
        cmap="tab10",
        s=34,
        alpha=0.72,
        linewidths=0,
    )
    ax_scatter.plot([lo, hi], [lo, hi], color="black", linewidth=1.2, linestyle="--", label="ideal")
    if y_true.size > 1:
        slope, intercept = np.polyfit(y_true, y_pred, deg=1)
        ax_scatter.plot(
            [lo, hi],
            [slope * lo + intercept, slope * hi + intercept],
            color="#b03a2e",
            linewidth=1.4,
            label=f"fit: y={slope:.2f}x{intercept:+.2f}",
        )
    ax_scatter.set_xlim(lo, hi)
    ax_scatter.set_ylim(lo, hi)
    ax_scatter.set_aspect("equal", adjustable="box")
    ax_scatter.set_xlabel("Actual")
    ax_scatter.set_ylabel("Predicted")
    ax_scatter.set_title(title)
    ax_scatter.grid(True, color="#e6e6e6", linewidth=0.8)
    ax_scatter.legend(loc="upper left", frameon=False)

    cbar = fig.colorbar(scatter, ax=ax_scatter, fraction=0.046, pad=0.04)
    cbar.set_label("Fold")

    text = (
        f"n={stats['n']:.0f}\n"
        f"MSE={stats['mse']:.3f}\n"
        f"RMSE={stats['rmse']:.3f}\n"
        f"MAE={stats['mae']:.3f}\n"
        f"bias={stats['bias']:.3f}\n"
        f"r={stats['r']:.3f}\n"
        f"±1={stats['within_1']:.3f}\n"
        f"±3={stats['within_3']:.3f}\n"
        f"±5={stats['within_5']:.3f}"
    )
    ax_scatter.text(
        0.98,
        0.02,
        text,
        transform=ax_scatter.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92},
    )

    bins = min(30, max(8, int(np.sqrt(residual.size))))
    ax_resid.hist(residual, bins=bins, color="#4c78a8", alpha=0.85, edgecolor="white")
    ax_resid.axvline(0, color="black", linewidth=1.1, linestyle="--")
    ax_resid.axvline(stats["bias"], color="#b03a2e", linewidth=1.4, label="mean residual")
    ax_resid.set_xlabel("Predicted - actual")
    ax_resid.set_ylabel("Count")
    ax_resid.set_title("Residuals")
    ax_resid.grid(True, axis="y", color="#e6e6e6", linewidth=0.8)
    ax_resid.legend(loc="upper right", frameon=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def default_output_path(path: Path, phase: str, head: str, split: str, label: str) -> Path:
    stem = f"regression_predictions_{phase}_{head}_{split}"
    if label:
        stem += f"_{label}"
    stem += ".png"
    return path.with_name(stem)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="A predictions.h5 file or a directory containing predictions.h5 files")
    ap.add_argument("--phase", default=None, help="Phase group to plot; defaults to the first logits phase")
    ap.add_argument("--head", default=None, help="Head name to plot; defaults to the first head")
    ap.add_argument("--split", default="test", help="Split label to plot: test, val, train, or all")
    ap.add_argument("--epoch", type=int, default=-1, help="Epoch index to plot; negative values count from the end")
    ap.add_argument("--output-index", type=int, default=0, help="Regression output index within logits' final axis")
    ap.add_argument("--glaucoma-only", action="store_true", help="Only plot rows where actual y_true is non-zero")
    ap.add_argument("--out-dir", type=Path, default=None, help="Optional directory for all output PNGs")
    ap.add_argument("--bin-summary", action="store_true", help="Print severity-bin accuracy and confusion matrix")
    ap.add_argument("--dpi", type=int, default=160, help="Output PNG resolution")
    args = ap.parse_args()

    filter_label = "glaucoma_only" if args.glaucoma_only else ""
    for pred_path in _find_prediction_files(args.path):
        y_true, y_pred, folds, phase, head = _collect_points(
            pred_path,
            phase=args.phase,
            head=args.head,
            split=args.split,
            epoch=args.epoch,
            output_index=args.output_index,
            glaucoma_only=args.glaucoma_only,
        )
        rel_title = pred_path.parent.as_posix()
        filter_text = ", glaucoma only" if args.glaucoma_only else ""
        title = f"{rel_title}\nphase={phase}, head={head}, split={args.split}, epoch={args.epoch}{filter_text}"
        if args.out_dir is None:
            out_path = default_output_path(pred_path, phase, head, args.split, filter_label)
        else:
            rep_name = "_".join(pred_path.parent.parts[-3:])
            suffix = f"_{filter_label}" if filter_label else ""
            out_path = args.out_dir / f"{rep_name}_{phase}_{head}_{args.split}{suffix}.png"
        _plot_one(y_true, y_pred, folds, title=title, output=out_path, dpi=args.dpi)
        stats = _metrics(y_true, y_pred)
        print(
            f"{out_path}  n={stats['n']:.0f}  "
            f"mse={stats['mse']:.4f}  rmse={stats['rmse']:.4f}  "
            f"mae={stats['mae']:.4f}  r={stats['r']:.4f}  "
            f"within_1={stats['within_1']:.3f}  "
            f"within_3={stats['within_3']:.3f}  "
            f"within_5={stats['within_5']:.3f}"
        )
        if args.bin_summary:
            print("\n".join(_bin_summary_lines(y_true, y_pred)))


if __name__ == "__main__":
    main()
