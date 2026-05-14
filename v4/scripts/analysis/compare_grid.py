"""compare_grid — print a ranked comparison table for a folder of v4 runs.

A "grid" folder is one whose immediate children are individual run folders, e.g.
``v4/results/experiments/tri_v1/grid/`` containing ``bcd35_cw0_nt15/``,
``bcd35_cw0_nt25/``, etc.  Each child must itself look like a run folder
(``repNN/.../summary.json``).

Usage:
    python -m v4.scripts.analysis.compare_grid <grid_folder>
        [--sort {test,val,name,reps}] [--reverse] [--csv]

Examples:
    python -m v4.scripts.analysis.compare_grid \
        v4/results/experiments/tri_v1/grid
    python -m v4.scripts.analysis.compare_grid \
        v4/results/experiments/tri_v1 --sort test
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from v4.scripts.analysis.summarize_run import summarise


def collect(grid_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for child in sorted(grid_dir.iterdir()):
        if not child.is_dir():
            continue
        s = summarise(child)
        if s["n_reps"] == 0:
            continue
        rows.append({
            "name":      child.name,
            "n":         s["n_reps"],
            "val_mean":  s["val_mean"],
            "val_std":   s["val_std"],
            "test_mean": s["test_mean"],
            "test_std":  s["test_std"],
        })
    return rows


def render_table(rows: list[dict]) -> str:
    if not rows:
        return "(no runs found)"
    name_w = max(len("name"), max(len(r["name"]) for r in rows))
    header = f"{'name':<{name_w}s}  {'reps':>4s}  {'val AUC':>17s}  {'test AUC':>17s}"
    sep    = "-" * len(header)
    lines  = [header, sep]
    for r in rows:
        lines.append(
            f"{r['name']:<{name_w}s}  {r['n']:>4d}  "
            f"{r['val_mean']:.4f} ± {r['val_std']:.4f}  "
            f"{r['test_mean']:.4f} ± {r['test_std']:.4f}"
        )
    return "\n".join(lines)


def render_csv(rows: list[dict]) -> str:
    buf = sys.stdout
    w   = csv.writer(buf)
    w.writerow(["name", "reps", "val_mean", "val_std", "test_mean", "test_std"])
    for r in rows:
        w.writerow([r["name"], r["n"],
                    f"{r['val_mean']:.6f}", f"{r['val_std']:.6f}",
                    f"{r['test_mean']:.6f}", f"{r['test_std']:.6f}"])
    return ""


_SORT_KEYS = {
    "test":  lambda r: r["test_mean"],
    "val":   lambda r: r["val_mean"],
    "name":  lambda r: r["name"],
    "reps":  lambda r: r["n"],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("grid_dir", type=Path,
                    help="Folder whose immediate children are run folders")
    ap.add_argument("--sort", choices=list(_SORT_KEYS),
                    default="test", help="Sort by which column (default: test)")
    ap.add_argument("--reverse", action="store_true",
                    help="Reverse the default ordering")
    ap.add_argument("--csv", action="store_true",
                    help="Emit CSV to stdout instead of a formatted table")
    args = ap.parse_args()

    if not args.grid_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.grid_dir}")

    rows = collect(args.grid_dir)
    descending = args.sort in {"test", "val", "reps"}
    if args.reverse:
        descending = not descending
    rows.sort(key=_SORT_KEYS[args.sort], reverse=descending)

    if args.csv:
        render_csv(rows)
    else:
        print(f"Grid: {args.grid_dir}  ({len(rows)} runs, sorted by {args.sort})\n")
        print(render_table(rows))


if __name__ == "__main__":
    main()
