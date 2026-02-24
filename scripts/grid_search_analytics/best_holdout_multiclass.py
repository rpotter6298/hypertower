#!/usr/bin/env python3
"""Rank multiclass runs by mean holdout AUC (fused) across folds."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


# ---------------------------
# Config (edit in IDE)
# ---------------------------
ANALYSIS_DIR = Path("analysis_data/grid_search")
TOP_N = 20
HEAD = "fused"  # fused | image | metadata
OUTPUT_CSV = Path("analysis_data/grid_search/plots/best_holdout_multiclass.csv")


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _infer_mode(summary: Optional[Dict[str, Any]]) -> Optional[str]:
    if not summary:
        return None
    eval_mode = summary.get("eval_mode")
    if isinstance(eval_mode, str):
        mode = eval_mode.strip().lower()
        if mode == "binary":
            return "binary"
        if mode in {"multiclass", "multi", "multi-class"}:
            return "multiclass"
    num_classes = summary.get("num_classes")
    if isinstance(num_classes, (int, float)):
        return "binary" if int(num_classes) <= 2 else "multiclass"
    class_names = summary.get("class_names")
    if isinstance(class_names, list) and class_names:
        return "binary" if len(class_names) <= 2 else "multiclass"
    return None


def _simple_fields(summary: Dict[str, Any]) -> Dict[str, Any]:
    keep: Dict[str, Any] = {}
    for key, val in summary.items():
        if key == "fold_metrics":
            continue
        if isinstance(val, (str, int, float, bool)) or val is None:
            keep[key] = val
    return keep


def _collect_fold_values(summary: Dict[str, Any], metric_key: str) -> List[float]:
    values: List[float] = []
    for entry in summary.get("fold_metrics") or []:
        if not isinstance(entry, dict):
            continue
        stats = entry.get("stats") if isinstance(entry.get("stats"), dict) else {}
        val = stats.get(metric_key)
        if isinstance(val, (int, float)):
            values.append(float(val))
    return values


def main() -> None:
    metric_key = f"holdout_auc_{HEAD}"
    rows: List[Dict[str, Any]] = []

    for run_dir in sorted(ANALYSIS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        summary = _read_json(run_dir / "summary.json")
        mode = _infer_mode(summary)
        if mode != "multiclass":
            continue

        values = _collect_fold_values(summary, metric_key)
        if not values:
            continue

        mean_val = float(np.mean(values))
        std_val = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")

        row = {
            "run_id": summary.get("run_id", run_dir.name),
            "run_dir": str(run_dir),
            "metric": metric_key,
            "mean": mean_val,
            "std": std_val,
            "n_folds": len(values),
            **_simple_fields(summary),
        }
        rows.append(row)

    if not rows:
        raise SystemExit("No multiclass runs with holdout AUC found.")

    df = pd.DataFrame(rows).sort_values(by="mean", ascending=False)
    top_df = df.head(TOP_N) if TOP_N else df

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(top_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved full ranking to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
