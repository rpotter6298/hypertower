#!/usr/bin/env python3
"""
Grid-search runner for run_multifold experiments.

Features:
  * Enumerates the requested configuration grid and writes grid_plan.csv.
  * Picks the next incomplete run, marks it running, executes run_multifold.py.
  * Records AUC/accuracy metrics per fold into grid_report.csv.
  * Removes model checkpoints for runs dominated (80%+ metrics worse) by others.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import fcntl

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_SCRIPT = REPO_ROOT / "scripts" / "run_multifold.py"
MANIFEST = REPO_ROOT / "manifest.csv"
GRID_DIR = REPO_ROOT / "analysis_data" / "grid_search"
PLAN_PATH = GRID_DIR / "grid_plan.csv"
REPORT_PATH = GRID_DIR / "grid_report.csv"
LOCK_PATH = GRID_DIR / ".grid_lock"
MODELS_ROOT = REPO_ROOT / "models" / "grid_search"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Grid-search orchestrator for run_multifold.")
    ap.add_argument("--plan-date", default=datetime.now().strftime("%Y%m%d"),
                    help="Date prefix used when generating run IDs (default: today).")
    ap.add_argument("--regen-plan", action="store_true",
                    help="Rebuild the grid plan from scratch (overwrites existing plan).")
    ap.add_argument("--manifest", type=Path, default=MANIFEST,
                    help="UNet manifest CSV for cropper.")
    ap.add_argument("--weights-dir", type=Path, default=REPO_ROOT / "models" / "unet_segmenter",
                    help="Directory containing norm_* subfolders with best.pt.")
    ap.add_argument("--dry-run", action="store_true", help="Enumerate next run without executing.")
    ap.add_argument("--max-runs", type=int, default=1,
                    help="Maximum runs to execute in this invocation (default: 1).")
    ap.add_argument("--run-all", action="store_true",
                    help="Execute runs sequentially until plan is exhausted (overrides --max-runs).")
    return ap.parse_args()


@contextmanager
def file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def write_csv(path: Path, rows: List[Dict[str, str]], headers: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def grid_configs(base_date: str, weights_dir: Path) -> List[Dict[str, str]]:
    eval_modes = ["binary", "multiclass"]
    crop_variants = [
        ("norm_imagenet", "imagenet"),
        ("normalize_none", "none"),
        ("norm_per_image", "per_image"),
    ]
    tta_opts = [False, True]
    loss_modes = ["focal", "balanced", "none"]
    thaw_modes = ["none", "gradual"]

    se_configs = []
    # none
    se_configs.append(("none", {"se_enabled": False}))
    # bridge only
    for pre in (True, False):
        se_configs.append((
            "bridge",
            {"se_enabled": True, "se_where": "bridge", "bridge_pre_norm": pre, "tower_pre_norm": None},
        ))
    # tower only
    for pre in (True, False):
        se_configs.append((
            "tower",
            {"se_enabled": True, "se_where": "tower", "bridge_pre_norm": None, "tower_pre_norm": pre},
        ))
    # both (four combos)
    for b_pre in (True, False):
        for t_pre in (True, False):
            se_configs.append((
                "both",
                {
                    "se_enabled": True,
                    "se_where": "both",
                    "bridge_pre_norm": b_pre,
                    "tower_pre_norm": t_pre,
                },
            ))

    combos = []
    idx = 0
    for eval_mode in eval_modes:
        for variant, norm in crop_variants:
            weights_path = weights_dir / variant / "best.pt"
            for tta in tta_opts:
                for loss in loss_modes:
                    for thaw in thaw_modes:
                        for se_name, se_opts in se_configs:
                            run_id = f"{base_date}-{idx:04d}"
                            combos.append({
                                "run_id": run_id,
                                "status": "incomplete",
                                "eval_mode": eval_mode,
                                "crop_variant": variant,
                                "crop_normalize": norm,
                                "crop_weights": str(weights_path),
                                "crop_tta": str(tta),
                                "loss_mode": loss,
                                "thaw_mode": thaw,
                                "se_mode": se_name,
                                "se_bridge_pre_norm": str(se_opts.get("bridge_pre_norm")),
                                "se_tower_pre_norm": str(se_opts.get("tower_pre_norm")),
                            })
                            idx += 1
    return combos


PLAN_HEADERS = [
    "run_id",
    "status",
    "eval_mode",
    "crop_variant",
    "crop_normalize",
    "crop_weights",
    "crop_tta",
    "loss_mode",
    "thaw_mode",
    "se_mode",
    "se_bridge_pre_norm",
    "se_tower_pre_norm",
]


def ensure_plan(args: argparse.Namespace) -> None:
    if args.regen_plan or not PLAN_PATH.exists():
        combos = grid_configs(args.plan_date, args.weights_dir)
        write_csv(PLAN_PATH, combos, PLAN_HEADERS)
        print(f"[grid] Plan created with {len(combos)} runs at {PLAN_PATH}")


def select_next_run() -> Optional[Dict[str, str]]:
    rows = read_csv(PLAN_PATH)
    for row in rows:
        if row["status"] == "incomplete":
            row["status"] = "running"
            write_csv(PLAN_PATH, rows, PLAN_HEADERS)
            return row
    return None


def update_run_status(run_id: str, new_status: str) -> None:
    rows = read_csv(PLAN_PATH)
    for row in rows:
        if row["run_id"] == run_id:
            row["status"] = new_status
            break
    write_csv(PLAN_PATH, rows, PLAN_HEADERS)


def build_run_command(row: Dict[str, str], manifest: Path) -> List[str]:
    cmd = [
        sys.executable,
        str(RUN_SCRIPT),
        "--backbone",
        "resnet50",
        "--fusion-mode",
        "fused",
        "--epochs",
        "40",
        "--batch-size",
        "8",
        "--img-crop-manifest",
        str(manifest),
        "--img-crop-weights",
        row["crop_weights"],
        "--img-crop-normalize",
        row["crop_normalize"],
        "--eval_mode",
        row["eval_mode"],
        "--holdout-per-class",
        "12",
        "--run-id",
        row["run_id"],
        "--shortname",
        "grid_search",
    ]
    if row["crop_tta"] == "True":
        cmd.append("--img-crop-tta")

    # Loss/balancing modes
    if row["loss_mode"] == "focal":
        cmd.extend(["--focal-gamma", "2.0"])
    elif row["loss_mode"] == "balanced":
        cmd.append("--balanced-sampler")

    # Thaw schedule
    if row["thaw_mode"] == "gradual":
        cmd.append("--gradual-thaw")
        cmd.extend(["--thaw-ratio", "0.33"])
        cmd.extend(["--thaw-start-epoch", "10"])
        cmd.extend(["--thaw-target", "image"])

    # SE settings
    if row["se_mode"] == "none":
        cmd.append("--no-se")
    else:
        cmd.extend(["--se-reduction", "16"])
        cmd.extend(["--se-reduction-tower", "16"])
        cmd.extend(["--se-where", row["se_mode"]])
        bridge_pre = row["se_bridge_pre_norm"]
        tower_pre = row["se_tower_pre_norm"]
        if bridge_pre == "True":
            cmd.append("--se-pre-norm")
        elif bridge_pre == "False":
            cmd.append("--no-se-pre-norm")
        if tower_pre == "True":
            cmd.append("--se-pre-norm-tower")
        elif tower_pre == "False":
            cmd.append("--no-se-pre-norm-tower")
    return cmd


def run_command(cmd: List[str]) -> None:
    print("[grid] Launching:", " ".join(cmd))
    subprocess.run(cmd, check=True)


METRIC_KEYS = ["auc_fused", "auc_img", "auc_md", "acc_fused", "acc_img", "acc_md"]


def extract_metrics(run_id: str) -> Dict[str, str]:
    summary_path = REPO_ROOT / "analysis_data" / "grid_search" / run_id / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json for run {run_id}")
    with summary_path.open() as fh:
        summary = json.load(fh)

    rows = {}
    for fold in summary.get("fold_metrics", []):
        if not isinstance(fold, dict):
            continue
        f_idx = fold.get("fold")
        stats = fold.get("stats") or {}
        if not isinstance(stats, dict):
            continue
        for key in METRIC_KEYS:
            val = stats.get(key)
            if val is None:
                continue
            rows[f"metric_fold{f_idx}_{key}"] = str(val)
    best_mean = summary.get("best_metric_mean")
    if best_mean is not None:
        rows["metric_best_mean"] = str(best_mean)
    return rows


def update_report(row: Dict[str, str], metrics: Dict[str, str]) -> None:
    existing = read_csv(REPORT_PATH)
    # Remove existing entry for run_id
    existing = [r for r in existing if r.get("run_id") != row["run_id"]]
    record = {**row, **metrics}
    existing.append(record)
    headers = sorted({key for r in existing for key in r.keys()})
    write_csv(REPORT_PATH, existing, headers)


def load_report_rows() -> List[Dict[str, str]]:
    return read_csv(REPORT_PATH)


def metric_columns(rows: List[Dict[str, str]]) -> List[str]:
    keys = set()
    for row in rows:
        for key in row:
            if key.startswith("metric_"):
                keys.add(key)
    return sorted(keys)


def _to_float(val: str) -> Optional[float]:
    try:
        f = float(val)
        if math.isnan(f):
            return None
        return f
    except Exception:
        return None


def prune_dominated(rows: List[Dict[str, str]]) -> None:
    """
    Remove model directories for runs that are clearly dominated by another run.
    A run is dominated if:
      * Another run has a strictly higher metric_best_mean, OR
      * Another run is >= on >=80% of overlapping metrics and strictly better on at least one.
    """
    metrics = metric_columns(rows)
    if not metrics:
        return

    dominated = set()
    for row in rows:
        run_id = row["run_id"]
        row_vals = {m: row.get(m) for m in metrics}
        row_best = _to_float(row_vals.get("metric_best_mean"))

        for other in rows:
            if other["run_id"] == run_id:
                continue

            other_vals = {m: other.get(m) for m in metrics}
            other_best = _to_float(other_vals.get("metric_best_mean"))

            # Fast path: compare aggregate best mean if both have it
            if row_best is not None and other_best is not None and other_best > row_best:
                dominated.add(run_id)
                break

            # Fallback: overlap-wise dominance
            comparisons = []
            better = 0
            for key in metrics:
                v1 = _to_float(row_vals.get(key))
                v2 = _to_float(other_vals.get(key))
                if v1 is None or v2 is None:
                    continue
                comparisons.append(v2 >= v1)
                if v2 > v1:
                    better += 1
            if not comparisons:
                continue
            fraction = sum(comparisons) / len(comparisons)
            if fraction >= 0.8 and better > 0:
                dominated.add(run_id)
                break

    for run_id in dominated:
        model_dir = MODELS_ROOT / run_id
        if model_dir.exists():
            print(f"[grid] Removing dominated model artifacts for {run_id}")
            try:
                shutil.rmtree(model_dir)
            except OSError as exc:
                # Don't fail the grid run if cleanup isn't permitted (e.g., locked SMB dirs).
                print(f"[grid] Warning: could not remove {model_dir}: {exc}")


def main():
    args = parse_args()
    ensure_plan(args)
    if args.dry_run:
        with file_lock(LOCK_PATH):
            next_run = select_next_run()
            if next_run is None:
                print("[grid] No incomplete runs remaining.")
                return
            update_run_status(next_run["run_id"], "incomplete")
            print("[grid] Next run:", next_run)
            return

    max_runs = None if args.run_all else args.max_runs
    runs_done = 0

    while True:
        with file_lock(LOCK_PATH):
            next_run = select_next_run()
        if next_run is None:
            if runs_done == 0:
                print("[grid] All runs completed.")
            else:
                print(f"[grid] No more runs remaining after {runs_done} run(s).")
            return

        run_id = next_run["run_id"]
        try:
            cmd = build_run_command(next_run, args.manifest)
            run_command(cmd)
            metrics = extract_metrics(run_id)
            with file_lock(LOCK_PATH):
                update_run_status(run_id, "completed")
                update_report(next_run, metrics)
                report_rows = load_report_rows()
                prune_dominated(report_rows)
            print(f"[grid] Run {run_id} completed.")
        except Exception as exc:
            with file_lock(LOCK_PATH):
                update_run_status(run_id, "incomplete")
            raise SystemExit(f"[grid] Run {run_id} failed: {exc}") from exc

        runs_done += 1
        if max_runs is not None and runs_done >= max_runs:
            print(f"[grid] Reached run limit ({max_runs}); stopping.")
            return


if __name__ == "__main__":
    main()
