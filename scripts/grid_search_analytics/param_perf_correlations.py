#!/usr/bin/env python3
"""
Analyze correlations between grid search parameters and performance metrics.

Example:
    python scripts/grid_search_analytics/param_perf_correlations.py \
        --analysis-dir analysis_data/grid_search \
        --task binary \
        --metric holdout_auc \
        --top 30
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

DEFAULT_EXCLUDE_KEYS = {
    "run_id",
    "fold_metrics",
    "best_metric",
    "best_metric_mode",
    "best_metric_mean",
    "best_metric_std",
    "eval_mode",
    "n_splits",
    "num_classes",
}

METRIC_MAP = {
    "auc": "auc_fused",
    "acc": "acc_fused",
    "holdout_auc": "holdout_auc_fused",
    "holdout_acc": "holdout_acc_fused",
}


def to_float(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
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


def mean(values: List[float]) -> Optional[float]:
    return (sum(values) / len(values)) if values else None


def read_json(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def read_summary(run_dir: Path) -> Optional[Dict[str, object]]:
    return read_json(run_dir / "summary.json")


def read_cli_args(run_dir: Path) -> Optional[Dict[str, object]]:
    return read_json(run_dir / "cli_args.json")


def read_run_id(run_dir: Path, summary: Optional[Dict[str, object]]) -> str:
    if summary:
        rid = summary.get("run_id")
        if isinstance(rid, str) and rid:
            return rid
    return run_dir.name


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


def metric_from_stats(stats: Dict[str, object], metric: str) -> Optional[float]:
    if metric.startswith("holdout_") and stats.get("holdout_best_monitor") == metric:
        best_val = to_float(stats.get("holdout_best_so_far"))
        if best_val is not None:
            return best_val
    return to_float(stats.get(metric))


def mean_metric(summary: Dict[str, object], metric: str) -> Optional[float]:
    folds = summary.get("fold_metrics") or []
    if not isinstance(folds, list) or not folds:
        return None
    values = []
    for fold in folds:
        stats = fold.get("stats") if isinstance(fold, dict) else None
        if not isinstance(stats, dict):
            return None
        val = metric_from_stats(stats, metric)
        if val is None:
            return None
        values.append(val)
    return mean(values)


def flatten_config(data: Dict[str, object],
                   prefix: str = "",
                   exclude_keys: Optional[Iterable[str]] = None) -> Dict[str, object]:
    out: Dict[str, object] = {}
    excludes = set(exclude_keys or [])
    for key, value in data.items():
        if key in excludes or key.startswith("best_"):
            continue
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            out.update(flatten_config(value, full_key, exclude_keys=excludes))
            continue
        if isinstance(value, list):
            continue
        out[full_key] = value
    return out


def rankdata(values: List[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def pearson(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 2:
        return None
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den_x = sum((xi - mean_x) ** 2 for xi in x)
    den_y = sum((yi - mean_y) ** 2 for yi in y)
    if den_x <= 0 or den_y <= 0:
        return None
    return num / math.sqrt(den_x * den_y)


def spearman(x: List[float], y: List[float]) -> Optional[float]:
    rx = rankdata(x)
    ry = rankdata(y)
    return pearson(rx, ry)


def correlation_ratio(categories: List[object], values: List[float]) -> Optional[float]:
    if len(categories) != len(values) or len(values) < 2:
        return None
    overall = mean(values)
    if overall is None:
        return None
    total = sum((v - overall) ** 2 for v in values)
    if total <= 0:
        return None
    sums: Dict[object, List[float]] = {}
    for cat, val in zip(categories, values):
        sums.setdefault(cat, []).append(val)
    between = 0.0
    for vals in sums.values():
        avg = mean(vals)
        if avg is None:
            continue
        between += len(vals) * (avg - overall) ** 2
    return math.sqrt(between / total)


def format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def format_metric(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def format_table(rows: List[Dict[str, object]], columns: List[str]) -> str:
    col_widths = {
        col: max(len(col), max((len(str(row.get(col, ""))) for row in rows), default=0))
        for col in columns
    }
    header = " | ".join(col.ljust(col_widths[col]) for col in columns)
    divider = "-+-".join("-" * col_widths[col] for col in columns)
    body = [
        " | ".join(str(row.get(col, "")).ljust(col_widths[col]) for col in columns)
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def render_progress(current: int, total: Optional[int], matched: int) -> str:
    if total:
        width = 30
        filled = int(width * current / total)
        bar = "#" * filled + "-" * (width - filled)
        return f"[{bar}] {current}/{total} matched {matched}"
    return f"Scanned {current} dirs, matched {matched}"


def iter_run_dirs(root: Path, shallow: bool, show_progress: bool) -> Iterable[Path]:
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
            if (entry / "summary.json").is_file():
                matched += 1
                yield entry
        if show_progress:
            print(file=sys.stderr)
        return

    matched = 0
    scanned = 0
    last_update = 0.0
    for dirpath, dirnames, filenames in os.walk(root):
        scanned += 1
        if show_progress:
            now = time.monotonic()
            if now - last_update >= 0.2:
                msg = render_progress(scanned, None, matched)
                print(f"\rScanning {msg}", end="", file=sys.stderr, flush=True)
                last_update = now
        if "summary.json" in filenames:
            matched += 1
            yield Path(dirpath)
    if show_progress:
        msg = render_progress(scanned, None, matched)
        print(f"\rScanning {msg}", end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description="Correlate grid search parameters with performance.")
    ap.add_argument("--analysis-dir", type=Path, default=Path("analysis_data/grid_search"),
                    help="Directory containing run subdirectories")
    ap.add_argument("--task", choices=["binary", "multiclass", "all"], default="all",
                    help="Filter runs by task type (default: all)")
    ap.add_argument("--metric", choices=sorted(METRIC_MAP.keys()), default="holdout_auc",
                    help="Performance metric to analyze (default: holdout_auc)")
    ap.add_argument("--sort-by", choices=["score", "abs_rho", "rho", "r", "eta"], default="score",
                    help="Sorting key for results (default: score)")
    ap.add_argument("--asc", action="store_true",
                    help="Sort ascending (default: descending)")
    ap.add_argument("--desc", action="store_true",
                    help="Sort descending (default: descending)")
    ap.add_argument("--top", type=int, default=30,
                    help="Limit output to top N parameters (default: 30)")
    ap.add_argument("--params", default=None,
                    help="Comma-separated list of parameter keys to include")
    ap.add_argument("--exclude", default=None,
                    help="Comma-separated list of parameter keys to exclude")
    ap.add_argument("--min-count", type=int, default=10,
                    help="Minimum runs required to analyze a parameter (default: 10)")
    ap.add_argument("--min-unique", type=int, default=2,
                    help="Minimum unique values required (default: 2)")
    ap.add_argument("--match", default=None,
                    help="Only include run directories whose name contains this substring")
    ap.add_argument("--shallow", action="store_true",
                    help="Only scan directories directly under analysis-dir")
    ap.add_argument("--no-progress", action="store_true",
                    help="Disable progress output")
    args = ap.parse_args()

    if args.asc and args.desc:
        raise SystemExit("Choose only one of --asc or --desc.")

    root = args.analysis_dir
    if not root.exists():
        raise SystemExit(f"Analysis directory not found: {root}")

    exclude_keys = set(DEFAULT_EXCLUDE_KEYS)
    if args.exclude:
        for item in args.exclude.split(","):
            item = item.strip()
            if item:
                exclude_keys.add(item)

    runs: List[Dict[str, object]] = []
    values_by_key: Dict[str, List[object]] = {}
    missing_summary = 0
    unknown_task = 0
    missing_cli = 0

    metric_key = METRIC_MAP[args.metric]

    for run_dir in iter_run_dirs(root, shallow=args.shallow, show_progress=not args.no_progress):
        if args.match and args.match not in run_dir.name:
            continue
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

        metric_value = mean_metric(summary, metric_key)
        if metric_value is None:
            continue

        cli_args = read_cli_args(run_dir)
        if cli_args is None:
            missing_cli += 1
        config_source = cli_args if cli_args is not None else summary
        config = flatten_config(config_source, exclude_keys=exclude_keys)

        run_id = read_run_id(run_dir, summary)
        runs.append({
            "run_id": run_id,
            "metric": metric_value,
            "config": config,
        })
        for key, value in config.items():
            values_by_key.setdefault(key, []).append(value)

    if not runs:
        print("No matching runs found.")
        return

    if args.params:
        param_keys = [p.strip() for p in args.params.split(",") if p.strip()]
    else:
        param_keys = []
        for key, values in values_by_key.items():
            unique_values = {format_value(v) for v in values}
            if len(unique_values) >= args.min_unique:
                param_keys.append(key)
        param_keys.sort()

    rows: List[Dict[str, object]] = []
    for key in param_keys:
        values = []
        metrics = []
        for run in runs:
            config = run.get("config", {})
            if key not in config:
                continue
            values.append(config[key])
            metrics.append(run["metric"])

        if len(values) < args.min_count:
            continue

        unique_values = {format_value(v) for v in values}
        if len(unique_values) < args.min_unique:
            continue

        numeric_values: List[float] = []
        numeric_ok = True
        for v in values:
            num = to_float(v)
            if num is None or isinstance(v, bool):
                numeric_ok = False
                break
            numeric_values.append(num)

        groups: Dict[object, List[float]] = {}
        for val, metric in zip(values, metrics):
            groups.setdefault(val, []).append(metric)
        group_means = {k: mean(v) for k, v in groups.items()}
        best_group = max(group_means.items(), key=lambda item: item[1] or float("-inf"))
        worst_group = min(group_means.items(), key=lambda item: item[1] or float("inf"))

        if numeric_ok and len(set(numeric_values)) >= 3:
            rho = spearman(numeric_values, metrics)
            r = pearson(numeric_values, metrics)
            score = abs(rho) if rho is not None else None
            row = {
                "param": key,
                "type": "numeric",
                "n": len(values),
                "distinct": len(unique_values),
                "score": format_metric(score) if score is not None else "",
                "rho": format_metric(rho),
                "r": format_metric(r),
                "best_value": format_value(best_group[0]),
                "best_mean": format_metric(best_group[1]),
                "worst_value": format_value(worst_group[0]),
                "worst_mean": format_metric(worst_group[1]),
            }
        else:
            eta = correlation_ratio(values, metrics)
            score = eta
            row = {
                "param": key,
                "type": "categorical",
                "n": len(values),
                "distinct": len(unique_values),
                "score": format_metric(score) if score is not None else "",
                "rho": "",
                "r": "",
                "best_value": format_value(best_group[0]),
                "best_mean": format_metric(best_group[1]),
                "worst_value": format_value(worst_group[0]),
                "worst_mean": format_metric(worst_group[1]),
            }

        rows.append(row)

    if not rows:
        print("No parameters met the minimum requirements.")
        return

    def sort_key(row: Dict[str, object]) -> float:
        raw = row.get(args.sort_by)
        if isinstance(raw, str):
            val = to_float(raw)
        else:
            val = to_float(raw)
        if val is None:
            return float("inf") if args.asc else float("-inf")
        return float(val)

    rows.sort(key=sort_key, reverse=not args.asc)
    if args.top is not None:
        rows = rows[:args.top]

    header_lines = []
    header_lines.append(f"Metric: {args.metric} (mean over folds)")
    if args.task != "all":
        header_lines.append(f"Task filter: {args.task}")
    if args.match:
        header_lines.append(f"Name filter: {args.match}")
    header_lines.append(f"Runs: {len(runs)}")
    if missing_cli:
        header_lines.append(f"Missing cli_args: {missing_cli}")
    if missing_summary or unknown_task:
        header_lines.append(f"Skipped: {missing_summary} missing summary, {unknown_task} unknown task")
    header_lines.append("")
    print("\n".join(header_lines))

    columns = [
        "param",
        "type",
        "n",
        "distinct",
        "score",
        "rho",
        "r",
        "best_value",
        "best_mean",
        "worst_value",
        "worst_mean",
    ]
    print(format_table(rows, columns))


if __name__ == "__main__":
    main()
