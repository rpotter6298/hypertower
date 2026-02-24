#!/usr/bin/env python3
"""
Scan an analysis directory for HyperTower run folders, extract the best per-fold
metric/accuracy from the epoch logs, and emit a combined summary.

Example:
    python scripts/batch_best_metrics.py \
        --analysis-dir analysis_data

    # Holdout ranking (faster, uses summary.json):
    python scripts/batch_best_metrics.py \
        --analysis-dir analysis_data/grid_search \
        --metric holdout_auc_fused \
        --acc-metric holdout_acc_fused \
        --source summary \
        --sort-by mean_auc --desc --top 10

The script assumes each run directory contains files named `fold{n}_epoch_log.csv`.
It reports runs that have all five folds (fold0..fold4) present by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REQUIRED_FOLDS = {f"fold{i}_epoch_log.csv" for i in range(5)}


def to_float(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        num = float(value)
        if math.isnan(num):
            return None
        return num
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    try:
        num = float(value)
    except ValueError:
        return None
    if math.isnan(num):
        return None
    return num


def best_value_from_csv(csv_path: Path, metric: str) -> Optional[Tuple[float, int]]:
    best: Optional[Tuple[float, int]] = None
    with csv_path.open("r", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            val = to_float(row.get(metric))
            if val is None:
                continue
            epoch = int(to_float(row.get("epoch")) or reader.line_num)
            if best is None or val > best[0]:
                best = (val, epoch)
    return best


def render_progress(current: int, total: Optional[int], matched: int) -> str:
    if total:
        width = 30
        filled = int(width * current / total)
        bar = "#" * filled + "-" * (width - filled)
        return f"[{bar}] {current}/{total} matched {matched}"
    return f"Scanned {current} dirs, matched {matched}"


def find_run_directories(root: Path,
                         shallow: bool,
                         required_files: Iterable[str],
                         show_progress: bool) -> Iterable[Path]:
    """
    Yield directories that look like HyperTower runs (contain at least the required fold logs).
    """
    required_set = set(required_files)
    if shallow:
        entries = [entry for entry in root.iterdir() if entry.is_dir()]
        entries.sort(key=lambda p: p.name)
        total = len(entries)
        matched = 0
        last_update = 0.0
        for idx, entry in enumerate(entries, start=1):
            if show_progress:
                now = time.monotonic()
                if now - last_update >= 0.1 or idx == total:
                    msg = render_progress(idx, total, matched)
                    print(f"\rScanning {msg}", end="", file=sys.stderr, flush=True)
                    last_update = now
            if not entry.is_dir():
                continue
            if all((entry / filename).is_file() for filename in required_set):
                matched += 1
                yield entry
        if show_progress:
            print(file=sys.stderr)
        return

    matched = 0
    scanned = 0
    last_update = 0.0
    for dirpath, dirnames, filenames in os_walk_sorted(root):
        scanned += 1
        if show_progress:
            now = time.monotonic()
            if now - last_update >= 0.2:
                msg = render_progress(scanned, None, matched)
                print(f"\rScanning {msg}", end="", file=sys.stderr, flush=True)
                last_update = now
        files = set(filenames)
        if required_set.issubset(files):
            matched += 1
            yield Path(dirpath)
    if show_progress:
        msg = render_progress(scanned, None, matched)
        print(f"\rScanning {msg}", end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)


def os_walk_sorted(root: Path):
    """
    Wrapper around os.walk that yields deterministic, sorted directory order.
    """
    import os

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        yield dirpath, dirnames, filenames


def read_summary(run_dir: Path) -> Optional[Dict[str, object]]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        data = json.loads(summary_path.read_text())
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def read_run_id(run_dir: Path, summary: Optional[Dict[str, object]] = None) -> str:
    data = summary if summary is not None else read_summary(run_dir)
    if data:
        rid = data.get("run_id")
        if isinstance(rid, str) and rid:
            return rid
    return run_dir.name


def mean(values: List[float]) -> Optional[float]:
    return (sum(values) / len(values)) if values else None


def metric_from_stats(stats: Dict[str, object], metric: str) -> Optional[float]:
    if stats.get("holdout_best_monitor") == metric:
        best_val = to_float(stats.get("holdout_best_so_far"))
        if best_val is not None:
            return best_val
    return to_float(stats.get(metric))

def task_from_summary(summary: Optional[Dict[str, object]]) -> Optional[str]:
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
    return None


def format_table(rows: List[Dict[str, Optional[object]]], columns: List[str]) -> str:
    col_widths = {
        col: max(len(col), max((len(fmt_value(row.get(col))) for row in rows), default=0))
        for col in columns
    }
    header = " | ".join(col.ljust(col_widths[col]) for col in columns)
    divider = "-+-".join("-" * col_widths[col] for col in columns)
    body_lines = [
        " | ".join(fmt_value(row.get(col)).ljust(col_widths[col]) for col in columns)
        for row in rows
    ]
    return "\n".join([header, divider, *body_lines])


def fmt_value(value: Optional[object]) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate best per-fold metrics from HyperTower runs.")
    ap.add_argument("--analysis-dir", type=Path, default=Path("analysis_data"),
                    help="Directory containing run subdirectories (default: analysis_data)")
    ap.add_argument("--metric", default="auc_fused",
                    help="Metric column to maximise (default: auc_fused)")
    ap.add_argument("--acc-metric", default="acc_fused",
                    help="Accuracy column to maximise (default: acc_fused)")
    ap.add_argument("--shallow", action="store_true",
                    help="Only scan directories directly under analysis-dir")
    ap.add_argument("--source", choices=["epoch_logs", "summary"], default="epoch_logs",
                    help="Where to read metrics from (default: epoch_logs)")
    ap.add_argument("--task", choices=["binary", "multiclass", "all"], default="all",
                    help="Filter runs by task type (default: all)")
    ap.add_argument("--no-progress", action="store_true",
                    help="Disable progress output")
    ap.add_argument("--match", default=None,
                    help="Only include run directories whose name contains this substring")
    ap.add_argument("--sort-by", choices=["mean_auc", "mean_acc"], default=None,
                    help="Optional column to sort by (default: none)")
    ap.add_argument("--desc", action="store_true",
                    help="Sort in descending order (default: ascending)")
    ap.add_argument("--top", type=int, default=None,
                    help="Limit output to the top N rows after sorting")
    ap.add_argument("--output-file", type=Path, default=None,
                    help="Optional path to write CSV summary")
    args = ap.parse_args()

    root = args.analysis_dir
    if not root.exists():
        raise SystemExit(f"Analysis directory not found: {root}")

    rows: List[Dict[str, Optional[object]]] = []
    missing_summary = 0
    unknown_task = 0

    required_files = REQUIRED_FOLDS if args.source == "epoch_logs" else ["summary.json"]
    for run_dir in find_run_directories(
            root,
            shallow=args.shallow,
            required_files=required_files,
            show_progress=not args.no_progress,
    ):
        if args.match and args.match not in run_dir.name:
            continue
        summary = None
        task_label = None
        if args.task != "all" or args.source == "summary":
            summary = read_summary(run_dir)
            if summary is None:
                missing_summary += 1
                continue
            task_label = task_from_summary(summary)
            if args.task != "all":
                if task_label is None:
                    unknown_task += 1
                    continue
                if task_label != args.task:
                    continue

        run_id = read_run_id(run_dir, summary)
        best_metrics: List[float] = []
        best_accs: List[float] = []
        if args.source == "summary":
            folds = summary.get("fold_metrics") if summary else None
            if not folds:
                continue
            for fold in folds:
                stats = fold.get("stats") or {}
                metric_val = metric_from_stats(stats, args.metric)
                acc_val = metric_from_stats(stats, args.acc_metric)
                if metric_val is None or acc_val is None:
                    best_metrics = []
                    best_accs = []
                    break
                best_metrics.append(metric_val)
                best_accs.append(acc_val)
        else:
            for fold_idx in range(5):
                csv_path = run_dir / f"fold{fold_idx}_epoch_log.csv"
                metric_entry = best_value_from_csv(csv_path, args.metric)
                acc_entry = best_value_from_csv(csv_path, args.acc_metric)
                if metric_entry is None or acc_entry is None:
                    # Skip this run if any fold is missing data
                    best_metrics = []
                    best_accs = []
                    break
                best_metrics.append(metric_entry[0])
                best_accs.append(acc_entry[0])

        if not best_metrics or not best_accs:
            continue

        rows.append({
            "run_id": run_id,
            "task": task_label,
            "relative_path": str(run_dir.relative_to(root)),
            "mean_auc": mean(best_metrics),
            "mean_acc": mean(best_accs),
        })

    if not rows:
        print("No matching runs found.")
        return

    if args.sort_by:
        def sort_key(row: Dict[str, Optional[float]]) -> float:
            value = row.get(args.sort_by)
            if value is None:
                return float("-inf") if args.desc else float("inf")
            return float(value)

        rows.sort(key=sort_key, reverse=args.desc)

    if args.top is not None:
        rows = rows[:args.top]

    columns = ["run_id", "task", "relative_path", "mean_auc", "mean_acc"]
    if args.task != "all":
        print(f"Task filter: {args.task}")
    if args.match:
        print(f"Name filter: {args.match}")
    print(f"Runs: {len(rows)}\n")
    print(format_table(rows, columns))

    if args.output_file:
        out_path = args.output_file
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"\nSummary written to {out_path}")
    if args.task != "all" and (missing_summary or unknown_task):
        print(f"\nSkipped {missing_summary} runs without summary.json and {unknown_task} with unknown task type.")


if __name__ == "__main__":
    main()
