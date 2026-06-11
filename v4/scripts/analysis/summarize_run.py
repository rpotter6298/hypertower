"""summarize_run — print cross-rep stats for one v4 run folder.

A "run" is a folder like ``v4/results/experiments/tri_v1/grid/bcd75_cw1_nt25/``
containing ``rep00/``, ``rep01/``, ... — each with a per-rep ``summary.json``
under any ``out_dir_tags`` subdir (typically ``binary/`` or ``binary/ntower/``).

Usage:
    python -m v4.scripts.analysis.summarize_run <run_folder> [--per-rep] [--json]

Examples:
    python -m v4.scripts.analysis.summarize_run \
        v4/results/experiments/tri_v1/grid/bcd75_cw1_nt25
    python -m v4.scripts.analysis.summarize_run \
        v4/results/experiments/tri_v1/baseline_tri --per-rep
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def find_rep_summaries(run_dir: Path) -> list[tuple[int, Path]]:
    """Return [(rep_idx, summary_path), ...] sorted by rep_idx."""
    out: list[tuple[int, Path]] = []
    for rep_dir in sorted(run_dir.glob("rep*")):
        if not rep_dir.is_dir():
            continue
        m = re.match(r"rep(\d+)$", rep_dir.name)
        if not m:
            continue
        summary = next(iter(rep_dir.rglob("summary.json")), None)
        if summary is not None:
            out.append((int(m.group(1)), summary))
    return out


def load_rep(summary_path: Path) -> dict:
    """Extract the fields we summarise from one rep's summary.json.

    Uses `primary_metric` if present (for regression runs e.g. neg_mse),
    falling back to AUC for legacy classification runs.
    """
    d = json.loads(summary_path.read_text())
    pm = d.get("primary_metric", "auc")
    return {
        "primary":    pm,
        "val_mean":   float(d.get(f"mean_val_{pm}",  d.get("mean_val_auc",  float("nan")))),
        "val_std":    float(d.get(f"std_val_{pm}",   d.get("std_val_auc",   float("nan")))),
        "test_mean":  float(d.get(f"mean_test_{pm}", d.get("mean_test_auc", float("nan")))),
        "test_std":   float(d.get(f"std_test_{pm}",  d.get("std_test_auc",  float("nan")))),
        "elapsed_s":  float(d.get("elapsed_s",       float("nan"))),
        "eval_stage": d.get("eval_stage", "?"),
    }


def summarise(run_dir: Path) -> dict:
    reps = find_rep_summaries(run_dir)
    if not reps:
        return {"run": str(run_dir), "n_reps": 0, "reps": []}
    rows = [(idx, load_rep(p)) for idx, p in reps]
    val   = np.array([r[1]["val_mean"]  for r in rows])
    test  = np.array([r[1]["test_mean"] for r in rows])
    elaps = np.array([r[1]["elapsed_s"] for r in rows])
    out = {
        "run":         str(run_dir),
        "n_reps":      len(rows),
        "eval_stage":  rows[0][1]["eval_stage"],
        "primary":     rows[0][1]["primary"],
        "val_mean":    float(np.mean(val)),
        "val_std":     float(np.std(val)),
        "val_min":     float(np.min(val)),
        "val_max":     float(np.max(val)),
        "test_mean":   float(np.mean(test)),
        "test_std":    float(np.std(test)),
        "test_min":    float(np.min(test)),
        "test_max":    float(np.max(test)),
        "elapsed_total_s": float(np.sum(elaps)) if not np.isnan(elaps).any() else None,
        "reps": [
            {"rep": idx, **info} for idx, info in rows
        ],
    }
    return out


def render(s: dict, per_rep: bool = False) -> str:
    if s["n_reps"] == 0:
        return f"Run: {s['run']}\n  no reps with summary.json found."

    metric = s.get("primary", "auc")
    lines = [
        f"Run:        {s['run']}",
        f"Reps:       {s['n_reps']}  (eval_stage={s['eval_stage']}, metric={metric})",
        f"Val  {metric}:   {s['val_mean']:.4f} ± {s['val_std']:.4f}   "
        f"[min={s['val_min']:.4f}  max={s['val_max']:.4f}]",
        f"Test {metric}:   {s['test_mean']:.4f} ± {s['test_std']:.4f}   "
        f"[min={s['test_min']:.4f}  max={s['test_max']:.4f}]",
    ]
    if s.get("elapsed_total_s") is not None:
        h = s["elapsed_total_s"] / 3600
        lines.append(f"Compute:    {s['elapsed_total_s']:.0f} s total  ({h:.1f} h)")

    if per_rep:
        lines.append("")
        lines.append("Per-rep breakdown:")
        lines.append(f"  {'rep':>4s}   {'val':>7s}   {'test':>7s}   {'elapsed':>7s}")
        for r in s["reps"]:
            elapsed = (f"{r['elapsed_s']:.0f}s" if not np.isnan(r['elapsed_s']) else "-")
            lines.append(
                f"  {r['rep']:>4d}   "
                f"{r['val_mean']:>7.4f}   "
                f"{r['test_mean']:>7.4f}   "
                f"{elapsed:>7s}"
            )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="Path to a run folder (contains repNN/ subdirs)")
    ap.add_argument("--per-rep", action="store_true", help="Print one line per rep")
    ap.add_argument("--json",    action="store_true", help="Emit JSON instead of formatted text")
    args = ap.parse_args()

    if not args.run_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.run_dir}")

    s = summarise(args.run_dir)
    if args.json:
        print(json.dumps(s, indent=2))
    else:
        print(render(s, per_rep=args.per_rep))


if __name__ == "__main__":
    main()
