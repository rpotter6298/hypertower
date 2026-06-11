"""Shared loaders/aggregators for v4 figure scripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "v4" / "results" / "experiments"


def summarise_run(run_path: Path, primary_hint: Optional[str] = None) -> Optional[dict]:
    """Aggregate val/test primary-metric mean & std across reps for one run folder.

    Returns dict with n, val_mean, val_std, test_mean, test_std, metric_name, or None if no reps."""
    val, test = [], []
    metric_name = primary_hint
    for rep in sorted(run_path.glob("rep*")):
        s = next(iter(rep.rglob("summary.json")), None)
        if not s:
            continue
        d = json.loads(s.read_text())
        pm = d.get("primary_metric") or metric_name or "auc"
        metric_name = metric_name or pm
        v = d.get(f"mean_val_{pm}")
        t = d.get(f"mean_test_{pm}")
        if v is None or t is None or not np.isfinite(v) or not np.isfinite(t):
            continue
        val.append(float(v)); test.append(float(t))
    if not val:
        return None
    return {
        "n":         len(val),
        "metric":    metric_name or "auc",
        "val_mean":  float(np.mean(val)),
        "val_std":   float(np.std(val)),
        "test_mean": float(np.mean(test)),
        "test_std":  float(np.std(test)),
        "val_arr":   np.array(val),
        "test_arr":  np.array(test),
    }


def summarise_many(name_to_path: dict[str, Path], primary_hint: Optional[str] = None) -> dict[str, Optional[dict]]:
    """Apply summarise_run to a dict of labelled run folders."""
    return {label: summarise_run(p, primary_hint) for label, p in name_to_path.items()}


def fmt_status(s: Optional[dict]) -> str:
    if s is None:
        return "pending"
    return f"n={s['n']:>2d}  test={s['test_mean']:.4f}±{s['test_std']:.4f}"
