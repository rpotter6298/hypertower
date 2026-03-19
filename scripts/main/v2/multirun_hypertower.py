#!/usr/bin/env python3
"""Thin CLI wrapper that runs V2 hypertower modes sequentially."""

from __future__ import annotations

from pathlib import Path
import sys
import argparse

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classes.v2.v2_hypertower import V2HyperTower


def parse_args():
    ap = argparse.ArgumentParser(
        description="Run selected eval/tower mode combinations sequentially."
    )
    ap.add_argument(
        "--eval-modes",
        nargs="+",
        choices=["binary", "multiclass"],
        default=["binary", "multiclass"],
    )
    ap.add_argument(
        "--tower-modes",
        nargs="+",
        choices=["single", "ensemble", "bilateral", "classic"],
        default=["single", "ensemble", "bilateral"],
    )
    return ap.parse_known_args()


def main():
    seq_args, remaining = parse_args()
    base_parser = V2HyperTower.build_parser()
    first_run = True
    for eval_mode in seq_args.eval_modes:
        for tower_mode in seq_args.tower_modes:
            tower_mode = "single" if tower_mode == "classic" else tower_mode
            cli = list(remaining) + ["--eval-mode", eval_mode, "--tower-mode", tower_mode]
            # Clear cache only on the first run; reuse it for all subsequent runs.
            if not first_run:
                cli.append("--persist-img-crop-cache")
            args = base_parser.parse_args(cli)
            # Skip if this mode is already fully complete.
            if args.run_name:
                tm_dir = Path(args.output_root) / args.run_name / eval_mode / tower_mode
                if (tm_dir / "summary.json").exists():
                    print(f"[compare] {eval_mode}:{tower_mode} already complete — skipping.")
                    first_run = False  # treat as done so cache is preserved for later runs
                    continue
            V2HyperTower(args).run()
            first_run = False


if __name__ == "__main__":
    main()
