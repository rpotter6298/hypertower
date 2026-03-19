#!/usr/bin/env python3
"""
Run single-mode binary + multiclass for each IOP correction method and
collect all results under one output root for easy comparison.

Output layout:
  analysis_data/iop_corr_comparison/
    ratio/binary/single/   ratio/multiclass/single/
    ols/binary/single/     ols/multiclass/single/
    lad/binary/single/     lad/multiclass/single/
    multi/binary/single/   multi/multiclass/single/

Usage
-----
    python scripts/main/v2/run_iop_corr_comparison.py [V2HyperTower args...]

Any extra args (backbone, epochs, img-crop-*, etc.) are forwarded to every run.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classes.v2.v2_hypertower import V2HyperTower

IOP_METHODS  = ["ratio", "ols", "lad", "multi"]
EVAL_MODES   = ["binary", "multiclass"]
OUTPUT_ROOT  = "analysis_data/iop_corr_comparison"


def main() -> None:
    base_parser = V2HyperTower.build_parser()
    # Consume only the remaining (forwarded) args — iop-corr-method and
    # run-name are set by this script; eval-mode and tower-mode likewise.
    _, remaining = base_parser.parse_known_args()

    first_run = True
    for method in IOP_METHODS:
        for eval_mode in EVAL_MODES:
            run_name = method          # one sub-folder per method
            tm_dir   = (Path(OUTPUT_ROOT) / run_name / eval_mode / "single")
            if (tm_dir / "summary.json").exists():
                print(f"[iop_corr] {method}/{eval_mode}/single — already done, skipping.")
                first_run = False
                continue

            cli = list(remaining) + [
                "--eval-mode",       eval_mode,
                "--tower-mode",      "single",
                "--iop-corr-method", method,
                "--output-root",     OUTPUT_ROOT,
                "--run-name",        run_name,
            ]
            if not first_run:
                cli.append("--persist-img-crop-cache")

            print(f"\n[iop_corr] Starting {method}/{eval_mode}/single ...")
            args = base_parser.parse_args(cli)
            V2HyperTower(args).run()
            first_run = False

    # ── summary table ──────────────────────────────────────────────────────
    import json
    print("\n" + "=" * 60)
    print("IOP correction method comparison — single mode")
    print("=" * 60)
    header = f"{'Method':<8}  {'Mode':<12}  {'Val AUC':>10}  {'Hld AUC':>10}"
    print(header)
    print("-" * len(header))
    for method in IOP_METHODS:
        for eval_mode in EVAL_MODES:
            p = Path(OUTPUT_ROOT) / method / eval_mode / "single" / "summary.json"
            if not p.exists():
                print(f"{method:<8}  {eval_mode:<12}  {'missing':>10}  {'missing':>10}")
                continue
            ms = json.loads(p.read_text()).get("mode_summary", {})
            val = ms.get("classic_best_val", {})
            hld = ms.get("classic_holdout", {})
            val_s = f"{val['auc_mean']:.3f}±{val['auc_std']:.3f}" if val.get("auc_mean") else "—"
            hld_s = f"{hld['auc_mean']:.3f}±{hld['auc_std']:.3f}" if hld.get("auc_mean") else "—"
            print(f"{method:<8}  {eval_mode:<12}  {val_s:>10}  {hld_s:>10}")
    print("=" * 60)


if __name__ == "__main__":
    main()
