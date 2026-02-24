#!/usr/bin/env python3
"""Re-run a single grid-search configuration using the V2 loader pipeline."""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


# ---------------------------
# Config (edit in IDE)
# ---------------------------
RUN_ID = "20251129-0063"  # fallback if --run-number is not provided
OUTPUT_RUN_ID = RUN_ID  # fallback output run id
SHORTNAME = "re_runs_v2"  # output root under analysis_data/ and models/
GRID_PLAN = Path("analysis_data/grid_search/grid_plan.csv")
MANIFEST = Path("manifest.csv")
RUN_SCRIPT = Path("scripts/run_multifold_v2.py")
REBUILD_SCRIPT = Path("scripts/rebuild_run_best_plots.py")
PLOT_HEADS = ["fused", "image", "metadata"]
USE_HOLDOUT_BEST_FOR_PLOTS = True
OVERWRITE_HOLDOUT_PROBS = True
ALLOW_EXISTING_RUN_DIR = False
SAMPLE_MODE = "eye"  # eye-level for parity with v1 grid runs


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Re-run a single grid-search item with V2 loaders.")
    ap.add_argument(
        "--run-number",
        type=str,
        default=None,
        help="Last 4 digits of run_id (e.g., 0063).",
    )
    ap.add_argument(
        "--output-run-id",
        type=str,
        default=None,
        help="Optional output run id; defaults to matched run_id.",
    )
    return ap.parse_args()


def _read_plan(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Grid plan not found: {path}")
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def _find_row(rows: List[Dict[str, str]], run_id: str) -> Dict[str, str]:
    for row in rows:
        if row.get("run_id") == run_id:
            return row
    raise ValueError(f"run_id not found in grid plan: {run_id}")


def _resolve_run_id(rows: List[Dict[str, str]], run_number: str | None) -> str:
    if not run_number:
        return RUN_ID
    run_number = str(run_number).strip()
    if run_number.isdigit():
        run_number = run_number.zfill(4)
    matches = [
        r.get("run_id", "")
        for r in rows
        if str(r.get("run_id", "")).endswith(f"-{run_number}")
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Multiple run_ids matched run-number '{run_number}': {matches[:5]}{' ...' if len(matches) > 5 else ''}"
        )
    raise ValueError(f"No run_id found ending with '-{run_number}'")


def _build_run_command(row: Dict[str, str], output_run_id: str) -> List[str]:
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
        str(MANIFEST),
        "--img-crop-weights",
        row["crop_weights"],
        "--img-crop-normalize",
        row["crop_normalize"],
        "--eval_mode",
        row["eval_mode"],
        "--holdout-per-class",
        "12",
        "--run-id",
        output_run_id,
        "--shortname",
        SHORTNAME,
        "--sample-mode",
        SAMPLE_MODE,
    ]
    if row.get("crop_tta") == "True":
        cmd.append("--img-crop-tta")

    loss_mode = row.get("loss_mode")
    if loss_mode == "focal":
        cmd.extend(["--focal-gamma", "2.0"])
    elif loss_mode == "balanced":
        cmd.append("--balanced-sampler")

    thaw_mode = row.get("thaw_mode")
    if thaw_mode == "gradual":
        cmd.append("--gradual-thaw")
        cmd.extend(["--thaw-ratio", "0.33"])
        cmd.extend(["--thaw-start-epoch", "10"])
        cmd.extend(["--thaw-target", "image"])

    se_mode = row.get("se_mode")
    if se_mode == "none":
        cmd.append("--no-se")
    else:
        cmd.extend(["--se-reduction", "16"])
        cmd.extend(["--se-reduction-tower", "16"])
        cmd.extend(["--se-where", se_mode])
        bridge_pre = row.get("se_bridge_pre_norm")
        tower_pre = row.get("se_tower_pre_norm")
        if bridge_pre == "True":
            cmd.append("--se-pre-norm")
        elif bridge_pre == "False":
            cmd.append("--no-se-pre-norm")
        if tower_pre == "True":
            cmd.append("--se-pre-norm-tower")
        elif tower_pre == "False":
            cmd.append("--no-se-pre-norm-tower")

    return cmd


def _swap_in_holdout_best(models_dir: Path) -> None:
    for fold_dir in sorted(models_dir.glob("fold*")):
        if not fold_dir.is_dir():
            continue
        holdout_best = fold_dir / "model_holdout_best.pt"
        model_best = fold_dir / "model_best.pt"
        if not holdout_best.exists():
            print(f"[warn] {holdout_best} missing; skipping.")
            continue
        if model_best.exists():
            backup = fold_dir / "model_best_from_train.pt"
            if not backup.exists():
                try:
                    shutil.copy2(model_best, backup)
                except Exception:
                    pass
        try:
            shutil.copy2(holdout_best, model_best)
        except Exception as exc:
            print(f"[warn] failed to replace {model_best}: {exc}")


def _run_rebuild(run_dir: Path) -> None:
    for head in PLOT_HEADS:
        cmd = [
            sys.executable,
            str(REBUILD_SCRIPT),
            "--run-dir",
            str(run_dir),
            "--head",
            head,
            "--use-holdout",
        ]
        if OVERWRITE_HOLDOUT_PROBS:
            cmd.append("--overwrite")
        print("[rerun] Rebuilding holdout ROC plots:", " ".join(cmd))
        subprocess.run(cmd, check=True)


def main() -> None:
    args = _parse_args()
    rows = _read_plan(GRID_PLAN)
    run_id = _resolve_run_id(rows, args.run_number)
    row = _find_row(rows, run_id)
    output_run_id = args.output_run_id or run_id

    run_dir = Path("analysis_data") / SHORTNAME / output_run_id
    if run_dir.exists() and not ALLOW_EXISTING_RUN_DIR:
        raise SystemExit(
            f"Run directory already exists: {run_dir} (set ALLOW_EXISTING_RUN_DIR=True to reuse)"
        )

    cmd = _build_run_command(row, output_run_id=output_run_id)
    print("[rerun] Launching:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    models_dir = Path("models") / SHORTNAME / output_run_id
    if USE_HOLDOUT_BEST_FOR_PLOTS:
        print("[rerun] Swapping in holdout-best checkpoints for plotting.")
        _swap_in_holdout_best(models_dir)

    _run_rebuild(run_dir)
    print(f"[rerun] Done. Outputs in {run_dir}")


if __name__ == "__main__":
    main()
