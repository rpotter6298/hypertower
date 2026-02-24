#!/usr/bin/env python3
"""
Build an HTML heatmap-style grid for grid search runs.

Each column is a run. The header shows mean metrics (auc, acc, holdout_auc,
holdout_acc). Rows encode hyperparameter options as red/green boxes.

Example:
    python scripts/grid_search_analytics/grid_search_heatmap.py \
        --analysis-dir analysis_data/grid_search \
        --task binary \
        --sort-by holdout_auc --desc \
        --top 40 \
        --format plot \
        --output analysis_data/grid_search_heatmap.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from html import escape
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


def mean(values: List[float]) -> Optional[float]:
    return (sum(values) / len(values)) if values else None


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


def read_cli_args(run_dir: Path) -> Optional[Dict[str, object]]:
    cli_path = run_dir / "cli_args.json"
    if not cli_path.exists():
        return None
    try:
        data = json.loads(cli_path.read_text())
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


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


def sort_value_key(value: object) -> Tuple[int, object]:
    if value is None:
        return (2, "")
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, (int, float)):
        return (0, value)
    return (1, str(value))


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


def build_html(runs: List[Dict[str, object]],
               row_specs: List[Tuple[str, object]],
               title: str,
               filters: List[str]) -> str:
    lines: List[str] = []
    lines.append("<!doctype html>")
    lines.append("<html lang=\"en\">")
    lines.append("<head>")
    lines.append("<meta charset=\"utf-8\">")
    lines.append(f"<title>{escape(title)}</title>")
    lines.append("<style>")
    lines.append(":root { --green: #4caf50; --red: #d9534f; --grid: #d0d0d0; --header: #f0f0f0; }")
    lines.append("body { margin: 0; padding: 16px; font-family: \"Courier New\", monospace; }")
    lines.append(".wrap { overflow-x: auto; }")
    lines.append("table { border-collapse: collapse; font-size: 12px; }")
    lines.append("th, td { border: 1px solid var(--grid); padding: 4px; text-align: center; }")
    lines.append("th.row-label { text-align: left; background: var(--header); position: sticky; left: 0; }")
    lines.append("thead th { background: var(--header); position: sticky; top: 0; z-index: 1; }")
    lines.append("th.run-id { writing-mode: vertical-rl; transform: rotate(180deg); white-space: nowrap; }")
    lines.append("td.cell { width: 14px; height: 14px; padding: 0; }")
    lines.append("td.on { background: var(--green); }")
    lines.append("td.off { background: var(--red); }")
    lines.append(".meta { margin-bottom: 12px; }")
    lines.append("</style>")
    lines.append("</head>")
    lines.append("<body>")
    lines.append(f"<h2>{escape(title)}</h2>")
    if filters:
        lines.append("<div class=\"meta\">")
        for item in filters:
            lines.append(f"<div>{escape(item)}</div>")
        lines.append("</div>")
    lines.append("<div class=\"wrap\">")
    lines.append("<table>")
    lines.append("<thead>")
    lines.append("<tr>")
    lines.append("<th>run_id</th>")
    for run in runs:
        run_id = escape(str(run.get("run_id", "")))
        rel_path = escape(str(run.get("relative_path", "")))
        title_attr = f" title=\"{rel_path}\"" if rel_path else ""
        lines.append(f"<th class=\"run-id\"{title_attr}>{run_id}</th>")
    lines.append("</tr>")
    for metric_key, label in [
        ("auc", "auc"),
        ("acc", "acc"),
        ("holdout_auc", "holdout_auc"),
        ("holdout_acc", "holdout_acc"),
    ]:
        lines.append("<tr>")
        lines.append(f"<th>{label}</th>")
        for run in runs:
            metrics = run.get("metrics", {})
            value = metrics.get(metric_key) if isinstance(metrics, dict) else None
            lines.append(f"<td>{escape(format_metric(value))}</td>")
        lines.append("</tr>")
    lines.append("</thead>")
    lines.append("<tbody>")
    for key, value in row_specs:
        label = f"{key}={format_value(value)}"
        lines.append("<tr>")
        lines.append(f"<th class=\"row-label\">{escape(label)}</th>")
        for run in runs:
            config = run.get("config", {})
            current = config.get(key) if isinstance(config, dict) else None
            cell_class = "on" if current == value else "off"
            lines.append(f"<td class=\"cell {cell_class}\"></td>")
        lines.append("</tr>")
    lines.append("</tbody>")
    lines.append("</table>")
    lines.append("</div>")
    lines.append("</body>")
    lines.append("</html>")
    return "\n".join(lines)


def truncate(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[:width - 3] + "..."


def build_text_grid(runs: List[Dict[str, object]],
                    row_specs: List[Tuple[str, object]],
                    filters: List[str],
                    col_width: int,
                    row_width: int,
                    color: bool) -> str:
    sep = " "
    lines: List[str] = []
    if filters:
        lines.extend(filters)
        lines.append("")

    def pad(text: str, width: int) -> str:
        return truncate(text, width).ljust(width)

    def colorize(text: str, enabled: bool) -> str:
        if not color:
            return text
        color_code = "\x1b[32m" if enabled else "\x1b[31m"
        return f"{color_code}{text}\x1b[0m"

    def row_line(label: str, values: List[str]) -> str:
        return pad(label, row_width) + sep + sep.join(pad(v, col_width) for v in values)

    run_ids = [str(run.get("run_id", "")) for run in runs]
    lines.append(row_line("run_id", run_ids))
    for metric_key, label in [
        ("auc", "auc"),
        ("acc", "acc"),
        ("holdout_auc", "holdout_auc"),
        ("holdout_acc", "holdout_acc"),
    ]:
        values = []
        for run in runs:
            metrics = run.get("metrics", {})
            value = metrics.get(metric_key) if isinstance(metrics, dict) else None
            values.append(format_metric(value))
        lines.append(row_line(label, values))

    divider = "-" * row_width + sep + sep.join("-" * col_width for _ in runs)
    lines.append(divider)

    for key, value in row_specs:
        label = f"{key}={format_value(value)}"
        cells: List[str] = []
        for run in runs:
            config = run.get("config", {})
            current = config.get(key) if isinstance(config, dict) else None
            enabled = current == value
            cell = colorize("##", enabled) if enabled else colorize("..", enabled)
            cells.append(cell)
        lines.append(row_line(label, cells))

    lines.append("")
    lines.append("Legend: ##=on ..=off")
    if color:
        lines.append("Colors: green=on red=off")
    return "\n".join(lines)


def parse_figsize(value: Optional[str], n_cols: int, n_rows: int) -> Tuple[float, float]:
    if value:
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if len(parts) == 2:
            try:
                return float(parts[0]), float(parts[1])
            except ValueError:
                pass
    width = min(40.0, max(8.0, n_cols * 0.3))
    height = min(40.0, max(6.0, (n_rows + 6) * 0.3))
    return width, height


def plot_heatmap(runs: List[Dict[str, object]],
                 row_specs: List[Tuple[str, object]],
                 filters: List[str],
                 output_path: Optional[Path],
                 figsize: Tuple[float, float],
                 dpi: int,
                 show: bool) -> None:
    if not show:
        import matplotlib
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    try:
        import seaborn as sns
    except ImportError:
        sns = None

    metric_labels = ["auc", "acc", "holdout_auc", "holdout_acc"]
    metric_matrix: List[List[float]] = []
    for label in metric_labels:
        row: List[float] = []
        for run in runs:
            metrics = run.get("metrics", {})
            value = metrics.get(label) if isinstance(metrics, dict) else None
            row.append(float(value) if value is not None else float("nan"))
        metric_matrix.append(row)

    param_labels = [f"{key}={format_value(value)}" for key, value in row_specs]
    param_matrix: List[List[int]] = []
    for key, value in row_specs:
        row = []
        for run in runs:
            config = run.get("config", {})
            current = config.get(key) if isinstance(config, dict) else None
            row.append(1 if current == value else 0)
        param_matrix.append(row)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    grid_rows = 2 if param_matrix else 1
    height_ratios = [2, max(2, len(param_matrix) * 0.5)] if param_matrix else [2]
    gs = fig.add_gridspec(grid_rows, 1, height_ratios=height_ratios, hspace=0.05)

    ax_metrics = fig.add_subplot(gs[0, 0])
    if sns:
        sns.heatmap(
            metric_matrix,
            ax=ax_metrics,
            cmap="viridis",
            annot=True,
            fmt=".3f",
            cbar=True,
            yticklabels=metric_labels,
            xticklabels=False,
        )
    else:
        im = ax_metrics.imshow(metric_matrix, aspect="auto", cmap="viridis")
        ax_metrics.set_yticks(range(len(metric_labels)))
        ax_metrics.set_yticklabels(metric_labels)
        fig.colorbar(im, ax=ax_metrics, fraction=0.02, pad=0.01)
        for i, row in enumerate(metric_matrix):
            for j, value in enumerate(row):
                if math.isnan(value):
                    continue
                ax_metrics.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=7, color="white")
    ax_metrics.set_ylabel("metrics")

    if param_matrix:
        ax_params = fig.add_subplot(gs[1, 0], sharex=ax_metrics)
        cmap = ListedColormap(["#d9534f", "#4caf50"])
        if sns:
            sns.heatmap(
                param_matrix,
                ax=ax_params,
                cmap=cmap,
                cbar=False,
                yticklabels=param_labels,
                xticklabels=[run.get("run_id", "") for run in runs],
                vmin=0,
                vmax=1,
            )
        else:
            ax_params.imshow(param_matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)
            ax_params.set_yticks(range(len(param_labels)))
            ax_params.set_yticklabels(param_labels)
            ax_params.set_xticks(range(len(runs)))
            ax_params.set_xticklabels([run.get("run_id", "") for run in runs], rotation=90)
        ax_params.set_xlabel("runs")
    else:
        ax_metrics.set_xticks(range(len(runs)))
        ax_metrics.set_xticklabels([run.get("run_id", "") for run in runs], rotation=90)
        ax_metrics.set_xlabel("runs")

    if filters:
        fig.suptitle("Grid Search Heatmap\n" + " | ".join(filters), fontsize=10)
    else:
        fig.suptitle("Grid Search Heatmap", fontsize=10)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build an HTML heatmap grid for grid search runs.")
    ap.add_argument("--analysis-dir", type=Path, default=Path("analysis_data/grid_search"),
                    help="Directory containing run subdirectories")
    ap.add_argument("--format", choices=["text", "html", "plot"], default="text",
                    help="Output format (default: text)")
    ap.add_argument("--task", choices=["binary", "multiclass", "all"], default="all",
                    help="Filter runs by task type (default: all)")
    ap.add_argument("--sort-by", choices=["auc", "acc", "holdout_auc", "holdout_acc"], default=None,
                    help="Metric to sort columns by (default: none)")
    ap.add_argument("--asc", action="store_true",
                    help="Sort in ascending order (default: descending)")
    ap.add_argument("--desc", action="store_true",
                    help="Sort in descending order (default: descending)")
    ap.add_argument("--top", type=int, default=None,
                    help="Limit to the top N runs after sorting")
    ap.add_argument("--cluster-rows", dest="cluster_rows", action="store_true",
                    help="Order parameter rows by prevalence in the selected runs (default)")
    ap.add_argument("--no-cluster-rows", dest="cluster_rows", action="store_false",
                    help="Keep parameter rows sorted alphabetically")
    ap.set_defaults(cluster_rows=True)
    ap.add_argument("--output", type=Path, default=None,
                    help="Optional path to write output")
    ap.add_argument("--params", default=None,
                    help="Comma-separated list of parameter keys to include")
    ap.add_argument("--exclude", default=None,
                    help="Comma-separated list of parameter keys to exclude")
    ap.add_argument("--match", default=None,
                    help="Only include run directories whose name contains this substring")
    ap.add_argument("--shallow", action="store_true",
                    help="Only scan directories directly under analysis-dir")
    ap.add_argument("--no-progress", action="store_true",
                    help="Disable progress output")
    ap.add_argument("--col-width", type=int, default=13,
                    help="Column width for text output (default: 13)")
    ap.add_argument("--row-width", type=int, default=36,
                    help="Row label width for text output (default: 36)")
    ap.add_argument("--color", action="store_true",
                    help="Use ANSI colors in text output")
    ap.add_argument("--figsize", default=None,
                    help="Figure size as 'width,height' (inches), for plot output")
    ap.add_argument("--dpi", type=int, default=140,
                    help="Figure DPI for plot output")
    ap.add_argument("--show", action="store_true",
                    help="Display plot window (only for format=plot)")
    args = ap.parse_args()

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

        metrics = {
            "auc": mean_metric(summary, "auc_fused"),
            "acc": mean_metric(summary, "acc_fused"),
            "holdout_auc": mean_metric(summary, "holdout_auc_fused"),
            "holdout_acc": mean_metric(summary, "holdout_acc_fused"),
        }
        if any(val is None for val in metrics.values()):
            continue

        cli_args = read_cli_args(run_dir)
        if cli_args is None:
            missing_cli += 1
        config_source = cli_args if cli_args is not None else summary
        config = flatten_config(config_source, exclude_keys=exclude_keys)
        run_id = read_run_id(run_dir, summary)
        runs.append({
            "run_id": run_id,
            "relative_path": str(run_dir.relative_to(root)),
            "task": task_label,
            "metrics": metrics,
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
            if len(unique_values) > 1:
                param_keys.append(key)
        param_keys.sort()

    row_specs: List[Tuple[str, object]] = []
    for key in param_keys:
        values = values_by_key.get(key, [])
        unique_values = []
        seen = set()
        for val in values:
            marker = (type(val), val)
            if marker in seen:
                continue
            seen.add(marker)
            unique_values.append(val)
        unique_values.sort(key=sort_value_key)
        for value in unique_values:
            row_specs.append((key, value))

    if args.asc and args.desc:
        raise SystemExit("Choose only one of --asc or --desc.")

    if args.sort_by:
        def sort_key(item: Dict[str, object]) -> float:
            metrics = item.get("metrics", {})
            val = metrics.get(args.sort_by) if isinstance(metrics, dict) else None
            if val is None:
                return float("inf") if args.asc else float("-inf")
            return float(val)

        runs.sort(key=sort_key, reverse=not args.asc)
    else:
        runs.sort(key=lambda r: str(r.get("run_id", "")))

    if args.top is not None:
        runs = runs[:args.top]

    if args.cluster_rows and runs:
        total = len(runs)
        counts_by_spec: Dict[Tuple[str, object], int] = {}
        for key, value in row_specs:
            counts_by_spec[(key, value)] = 0
        for run in runs:
            config = run.get("config", {})
            if not isinstance(config, dict):
                continue
            for key, value in row_specs:
                if config.get(key) == value:
                    counts_by_spec[(key, value)] += 1

        def row_sort(spec: Tuple[str, object]) -> Tuple[float, str, str]:
            count = counts_by_spec.get(spec, 0)
            score = count / total if total else 0.0
            key, value = spec
            return (-score, str(key), format_value(value))

        row_specs.sort(key=row_sort)

    filters = []
    if args.task != "all":
        filters.append(f"Task filter: {args.task}")
    if args.match:
        filters.append(f"Name filter: {args.match}")
    filters.append(f"Runs: {len(runs)}")
    filters.append(f"Params: {len(row_specs)}")
    filters.append(f"Row clustering: {'on' if args.cluster_rows else 'off'}")
    if missing_summary or unknown_task:
        filters.append(f"Skipped: {missing_summary} missing summary, {unknown_task} unknown task")
    if missing_cli:
        filters.append(f"Missing cli_args: {missing_cli}")

    title = "Grid Search Heatmap"
    if args.format == "html":
        html = build_html(runs, row_specs, title=title, filters=filters)
        output_path = args.output or Path("analysis_data/grid_search_heatmap.html")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html)
        print(f"Wrote {output_path}")
    elif args.format == "plot":
        output_path = args.output
        if output_path is None and not args.show:
            output_path = Path("analysis_data/grid_search_heatmap.png")
        figsize = parse_figsize(args.figsize, n_cols=len(runs), n_rows=len(row_specs))
        plot_heatmap(
            runs,
            row_specs,
            filters=filters,
            output_path=output_path,
            figsize=figsize,
            dpi=args.dpi,
            show=args.show,
        )
        if output_path is not None:
            print(f"Wrote {output_path}")
    else:
        text = build_text_grid(
            runs,
            row_specs,
            filters=filters,
            col_width=max(4, args.col_width),
            row_width=max(12, args.row_width),
            color=args.color,
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
            print(f"Wrote {args.output}")
        else:
            print(text)


if __name__ == "__main__":
    main()
