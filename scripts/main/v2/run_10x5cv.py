#!/usr/bin/env python3
"""
10× repeated 5-fold CV runner for the best hypertower configuration
(nocrop, ensemble mode, both binary and multiclass).

Each repetition uses a different fold-seed so the 5 folds are split
differently, giving 50 folds per eval-mode total.  Holdout composition
is kept identical across repetitions (same --holdout-seed).

Results land under:
    {output-root}/rep{N:02d}/{eval_mode}/ensemble/fold{K}/

Usage
-----
    python scripts/main/v2/run_10x5cv.py \
        --n-reps 10 \
        --eval-modes binary multiclass \
        --output-root analysis_data/pipeline_10x5 \
        --epochs 40 --fused-head \
        --backbone refugelike

Any extra flags are forwarded directly to V2HyperTower.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classes.v2.v2_hypertower import V2HyperTower

# Base fold seed for rep 0; rep N uses BASE_SEED + N * SEED_STRIDE
_BASE_SEED   = 100
_SEED_STRIDE = 100


def _parse_own(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    ap.add_argument("--n-reps", type=int, default=10,
                    help="Number of repetitions (default: 10).")
    ap.add_argument("--eval-modes", nargs="+",
                    choices=["binary", "multiclass"],
                    default=["binary", "multiclass"])
    ap.add_argument("--output-root", default="analysis_data/pipeline_10x5",
                    help="Parent directory for all rep sub-runs.")
    ap.add_argument("-h", "--help", action="store_true")
    return ap.parse_known_args(argv)


def main(argv=None):
    own, remaining = _parse_own(argv)

    if own.help:
        print(__doc__)
        base_parser = V2HyperTower.build_parser()
        base_parser.print_help()
        return

    base_parser = V2HyperTower.build_parser()
    output_root = Path(own.output_root)
    first_run   = True

    for rep in range(own.n_reps):
        fold_seed = _BASE_SEED + rep * _SEED_STRIDE
        rep_label = f"rep{rep:02d}"

        for eval_mode in own.eval_modes:
            tower_mode = "ensemble"

            # Skip if already fully complete
            tm_dir = output_root / rep_label / eval_mode / tower_mode
            if (tm_dir / "summary.json").exists():
                print(f"[10x5cv] {rep_label} {eval_mode}:{tower_mode} — already done, skipping.")
                first_run = False
                continue

            cli = list(remaining) + [
                "--eval-mode",    eval_mode,
                "--tower-mode",   tower_mode,
                "--fold-seed",    str(fold_seed),
                "--run-name",     rep_label,
                "--output-root",  str(output_root),
            ]

            # Reuse crop cache across runs after the first
            if not first_run:
                cli.append("--persist-img-crop-cache")

            print(f"\n[10x5cv] Starting {rep_label} {eval_mode}:{tower_mode}  "
                  f"(fold_seed={fold_seed})")
            args = base_parser.parse_args(cli)
            V2HyperTower(args).run()
            first_run = False

    print(f"\n[10x5cv] All done.  Results in: {output_root}")


if __name__ == "__main__":
    main()
