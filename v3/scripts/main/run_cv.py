#!/usr/bin/env python
"""
V3 cross-validation runner.

Outer/inner k-fold: test = current fold, val = next fold, train = rest.
No holdout. No checkpoint saving.

Usage (single fold-seed, 5-fold, binary, ensemble):
  python -m v3.scripts.main.run_cv \
      --run-name my_run \
      --eval-mode binary \
      --tower-mode ensemble \
      --epochs 40 \
      --augment \
      --tune-binary-threshold \
      --in-memory-cache

Usage (10x5 rep-CV, seeds 100..1000):
  python -m v3.scripts.main.run_cv \
      --run-name my_run_10x5 \
      --reps 10 \
      --rep-seed-start 100 \
      --rep-seed-step 100 \
      --eval-mode binary \
      --tower-mode ensemble \
      --epochs 40 \
      --augment \
      --tune-binary-threshold \
      --in-memory-cache
"""
import argparse
import sys
from pathlib import Path

# Allow running as `python v3/scripts/main/run_cv.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from v3.classes.v3_hypertower import V3HyperTower


def build_parser() -> argparse.ArgumentParser:
    ap = V3HyperTower.build_parser()
    ap.description = __doc__
    ap.formatter_class = argparse.RawDescriptionHelpFormatter
    ap.add_argument(
        "--reps", type=int, default=1,
        help="Number of repetitions (each rep uses a different --fold-seed).",
    )
    ap.add_argument(
        "--rep-seed-start", type=int, default=100,
        help="fold-seed for rep 0 (default: 100).",
    )
    ap.add_argument(
        "--rep-seed-step", type=int, default=100,
        help="Increment between rep fold-seeds (default: 100; rep k uses seed start + k*step).",
    )
    ap.add_argument(
        "--rep-index", type=int, default=None,
        help="Override the rep directory index (e.g. 3 → rep03). "
             "Used by the distributed server to run a single rep of a multi-rep job.",
    )
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()

    reps           = int(args.reps)
    seed_start     = int(args.rep_seed_start)
    seed_step      = int(args.rep_seed_step)
    base_run_name  = args.run_name or "v3_cv"
    rep_index_override = getattr(args, "rep_index", None)

    for rep in range(reps):
        rep_seed = seed_start + rep * seed_step
        args.fold_seed = rep_seed

        dir_index = rep_index_override if (rep_index_override is not None and reps == 1) else rep
        if reps > 1 or rep_index_override is not None:
            args.run_name = f"{base_run_name}/rep{dir_index:02d}"
            print(f"\n{'='*60}", flush=True)
            print(f"Rep {dir_index+1}  fold_seed={rep_seed}", flush=True)
            print(f"{'='*60}", flush=True)
        else:
            args.run_name = base_run_name

        tower = V3HyperTower(args)
        out_dir = tower.run()
        print(f"\nRep {dir_index+1} output: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
