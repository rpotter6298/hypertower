#!/usr/bin/env python3
"""
Quick utility to recover the best epoch metrics from HyperTower run folders.

Example:
    python scripts/extract_best_auc.py analysis_data/img_only_densenet_gt_bin/img_only_densenet_gt_bin_20251028_112733

By default it looks for columns named like `auc_fused` (set via --metric) inside each
`fold{n}_epoch_log.csv`, returning the epoch with the highest value plus the holdout
metrics, if present.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple


def to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if math.isnan(out):
        return None
    return out


def best_row(path: Path, metric: str) -> Optional[Dict[str, str]]:
    if not path.exists():
        return None
    best: Optional[Tuple[float, int, Dict[str, str]]] = None
    with path.open("r", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            val = to_float(row.get(metric))
            if val is None:
                continue
            epoch = int(row.get("epoch", reader.line_num))
            if best is None or val > best[0]:
                best = (val, epoch, row)
    return best[2] if best else None


def summarize_fold(row: Dict[str, str], metric: str) -> Dict[str, float]:
    data: Dict[str, float] = {}
    for key in (metric, f"holdout_{metric.split('_', 1)[-1]}", "holdout_auc_img", "holdout_auc_fused"):
        val = to_float(row.get(key))
        if val is not None:
            data[key] = val
    epoch_val = to_float(row.get("epoch"))
    if epoch_val is not None:
        data["epoch"] = int(epoch_val)
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract best-per-fold metric from HyperTower runs.")
    ap.add_argument("run_dir", type=Path, help="Run directory (contains fold*_epoch_log.csv)")
    ap.add_argument("--metric", default="auc_fused", help="Metric column to maximise (default: auc_fused)")
    ap.add_argument("--json", type=Path, default=None, help="Optional path to dump JSON summary")
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    metric: str = args.metric

    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    fold_summaries: Dict[str, Dict[str, float]] = {}
    metric_values = []

    for csv_path in sorted(run_dir.glob("fold*_epoch_log.csv")):
        best = best_row(csv_path, metric)
        fold_name = csv_path.stem.replace("_epoch_log", "")
        if best is None:
            print(f"{fold_name}: no valid '{metric}' values found")
            continue
        summary = summarize_fold(best, metric)
        fold_summaries[fold_name] = summary
        val = summary.get(metric)
        if val is not None:
            metric_values.append(val)
        holdout_val = summary.get(f"holdout_{metric.split('_', 1)[-1]}")
        print(f"{fold_name}: epoch={summary.get('epoch')} {metric}={val:.4f}" if val is not None else f"{fold_name}: epoch={summary.get('epoch')}")
        if holdout_val is not None:
            print(f"    holdout_{metric.split('_', 1)[-1]}={holdout_val:.4f}")

    if metric_values:
        mean_val = sum(metric_values) / len(metric_values)
        print(f"\nMean best {metric}: {mean_val:.4f}")

    if args.json:
        payload = {
            "run_dir": str(run_dir),
            "metric": metric,
            "folds": fold_summaries,
            "mean_metric": (sum(metric_values) / len(metric_values)) if metric_values else None,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"Summary written to {args.json}")


if __name__ == "__main__":
    main()
