"""Post-hoc regression calibration on predictions.h5 files.

For each fold of a regression run, fit a linear calibration
   actual_md ≈ a · predicted_md + b
on the *val* split, then apply (a, b) to that fold's *test* predictions
and report metrics before vs after calibration. No retraining required.

Usage:
    python -m v4.scripts.analysis.calibrate_regression \\
        v4/results/experiments/reg_head/baseline_reg_nt50

Pass a single predictions.h5 file or a folder; the script finds every
predictions.h5 under it and produces a per-rep + aggregate report.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import h5py
import numpy as np

BIN_LABELS = ["<=-10", "-9..-5", "-4..-1", ">=1"]


def _severity_bin(values: np.ndarray) -> np.ndarray:
    bins = np.full(values.shape, -1, dtype=int)
    bins[values <= -9.5] = 0
    bins[(values > -9.5) & (values <= -4.5)] = 1
    bins[(values > -4.5) & (values <= 0.0)] = 2
    bins[values > 0.0] = 3
    return bins


def _decode(arr) -> np.ndarray:
    return np.array([s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in arr])


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    res = y_pred - y_true
    mse = float(np.mean(res ** 2))
    abs_res = np.abs(res)
    out = {
        "n":        int(y_true.size),
        "mse":      mse,
        "rmse":     math.sqrt(mse),
        "mae":      float(np.mean(abs_res)),
        "bias":     float(np.mean(res)),
        "within_1": float(np.mean(abs_res <= 1.0)),
        "within_3": float(np.mean(abs_res <= 3.0)),
        "within_5": float(np.mean(abs_res <= 5.0)),
        "r":        float(np.corrcoef(y_true, y_pred)[0, 1])
                    if y_true.size > 1 and np.std(y_pred) > 0 else float("nan"),
    }
    bins_true = _severity_bin(y_true)
    bins_pred = _severity_bin(y_pred)
    valid = (bins_true >= 0) & (bins_pred >= 0)
    out["bin_exact"]    = float(np.mean(bins_true[valid] == bins_pred[valid])) if valid.any() else float("nan")
    out["bin_adjacent"] = float(np.mean(np.abs(bins_true[valid] - bins_pred[valid]) <= 1)) if valid.any() else float("nan")
    return out


def _fit_linear(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Fit y_true = a * y_pred + b via OLS. Returns (a, b)."""
    if y_pred.size < 2 or np.std(y_pred) == 0:
        return 1.0, 0.0
    a, b = np.polyfit(y_pred, y_true, 1)
    return float(a), float(b)


def calibrate_one(path: Path) -> dict:
    """Run val→test linear calibration on one predictions.h5. Returns aggregate metrics."""
    with h5py.File(path, "r") as f:
        # Pick the last phase that has a head (heuristic: highest stage)
        phases = sorted(k for k in f.keys() if isinstance(f[k], h5py.Group) and "logits" in f[k])
        if not phases:
            return {"path": str(path), "skipped": "no logits"}
        # Prefer 'hb' if present
        phase = "hb" if "hb" in phases else phases[-1]
        grp = f[phase]
        logits   = grp["logits"][:]            # (folds, epochs, samples, heads, outputs)
        y_true   = grp["y_true"][:].astype(float)
        split    = grp["split"][:]
        n_folds, n_epochs, n_samples, n_heads, _ = logits.shape
        # Use the last epoch and head 0, output 0
        ep, head, out = n_epochs - 1, n_heads - 1, 0

        raw_test:   list[np.ndarray] = []
        cal_test:   list[np.ndarray] = []
        truth_test: list[np.ndarray] = []
        slopes: list[float] = []
        intercepts: list[float] = []

        for fold in range(n_folds):
            labels = _decode(split[fold])
            v_mask = (labels == "val")  & np.isfinite(y_true) & np.isfinite(logits[fold, ep, :, head, out])
            t_mask = (labels == "test") & np.isfinite(y_true) & np.isfinite(logits[fold, ep, :, head, out])
            if not v_mask.any() or not t_mask.any():
                continue
            v_pred = logits[fold, ep, v_mask, head, out].astype(float)
            v_true = y_true[v_mask]
            a, b = _fit_linear(v_true, v_pred)
            slopes.append(a); intercepts.append(b)
            t_pred = logits[fold, ep, t_mask, head, out].astype(float)
            t_true = y_true[t_mask]
            raw_test.append(t_pred)
            cal_test.append(a * t_pred + b)
            truth_test.append(t_true)

    if not raw_test:
        return {"path": str(path), "skipped": "no val/test rows"}

    truth = np.concatenate(truth_test)
    raw   = _metrics(truth, np.concatenate(raw_test))
    cal   = _metrics(truth, np.concatenate(cal_test))
    return {
        "path":   str(path),
        "slopes":     slopes,
        "intercepts": intercepts,
        "raw":    raw,
        "cal":    cal,
    }


def _fmt(row: dict, label: str) -> str:
    return (f"  {label}: mse={row['mse']:.3f} mae={row['mae']:.3f} bias={row['bias']:+.3f} "
            f"r={row['r']:.3f}  bin_exact={row['bin_exact']:.3f} bin_adj={row['bin_adjacent']:.3f}  "
            f"within_1={row['within_1']:.2f} within_3={row['within_3']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="predictions.h5 file or directory")
    args = ap.parse_args()

    if args.path.is_file():
        files = [args.path]
    else:
        files = sorted(args.path.rglob("predictions.h5"))
    if not files:
        raise SystemExit(f"No predictions.h5 found under {args.path}")

    # Aggregate raw vs cal across all reps
    agg_raw_truth, agg_raw_pred = [], []
    agg_cal_truth, agg_cal_pred = [], []
    all_slopes, all_intercepts = [], []

    for fp in files:
        res = calibrate_one(fp)
        if res.get("skipped"):
            print(f"\n{fp.parent.parent.name}: SKIPPED ({res['skipped']})")
            continue
        rel = fp.relative_to(args.path) if args.path.is_dir() else fp.name
        print(f"\n{rel}")
        print(f"  fold-wise slopes:      {', '.join(f'{a:.2f}' for a in res['slopes'])}")
        print(f"  fold-wise intercepts:  {', '.join(f'{b:+.2f}' for b in res['intercepts'])}")
        print(_fmt(res["raw"], "raw "))
        print(_fmt(res["cal"], "cal "))
        all_slopes.extend(res["slopes"])
        all_intercepts.extend(res["intercepts"])

    if all_slopes:
        print(
            f"\nCalibration parameters across all reps × folds (n={len(all_slopes)}):\n"
            f"  slope:      mean={np.mean(all_slopes):.3f} ± {np.std(all_slopes):.3f}  "
            f"min={np.min(all_slopes):.2f}  max={np.max(all_slopes):.2f}\n"
            f"  intercept:  mean={np.mean(all_intercepts):+.3f} ± {np.std(all_intercepts):.3f}"
        )


if __name__ == "__main__":
    main()
