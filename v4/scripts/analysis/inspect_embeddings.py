"""inspect_embeddings — per-dim statistics on a run's saved feature embeddings.

Loads features.h5 from a run folder (must have been produced with
``save_features: true`` in the config), computes per-dimension variance,
sparsity (fraction of |z| < threshold), and useful summary stats.

Helpful for diagnosing whether a fusion stage is collapsing dimensions to
near-zero — which would silently null out information when the bridge uses
a Hadamard product.

Usage:
    python -m v4.scripts.analysis.inspect_embeddings <run_folder> [--phase NAME]
        [--threshold 0.05]

Examples:
    python -m v4.scripts.analysis.inspect_embeddings \
        v4/results/experiments/tri_v1/baseline_tri/rep00 --phase nt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def load_phase(features_path: Path, phase: str | None):
    with h5py.File(features_path, "r") as f:
        phases = list(f.keys())
        if phase is None:
            phase = phases[-1]
        if phase not in phases:
            raise SystemExit(
                f"Phase {phase!r} not in {features_path} (available: {phases})"
            )
        g = f[phase]
        z      = g["z"][:]            # (n_folds, n_samples, n_dim)
        split  = g["split"][:].astype(str)  # (n_folds, n_samples)
        y_true = g["y_true"][:]
    return phase, z, split, y_true, phases


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="Folder containing features.h5 (under repNN)")
    ap.add_argument("--phase", default=None,
                    help="Bridge phase name (e.g. nt, hb, cd_fuse). Default: last in file.")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="|z| threshold for the sparsity count (default: 0.05)")
    args = ap.parse_args()

    feat = next(iter(args.run_dir.rglob("features.h5")), None)
    if feat is None:
        raise SystemExit(f"No features.h5 found under {args.run_dir}")

    phase, z, split, _, all_phases = load_phase(feat, args.phase)

    val_mask  = (split == "val")
    z_val     = z[val_mask]   # (n_val_total, n_dim)
    z_train   = z[(split == "train")]
    n_dim     = z_val.shape[-1]

    print(f"Run:      {args.run_dir}")
    print(f"File:     {feat}")
    print(f"Phases:   {all_phases}")
    print(f"Phase:    {phase}  (z shape: {z.shape})")
    print(f"Split sizes: train={len(z_train)}  val={len(z_val)}")
    print()

    abs_z       = np.abs(z_val)
    per_dim_var = z_val.var(axis=0)
    per_dim_abs = abs_z.mean(axis=0)
    per_dim_max = abs_z.max(axis=0)
    sparsity    = (abs_z < args.threshold).mean(axis=0)

    print(f"Aggregate stats over val embeddings (|z| < {args.threshold} = 'near-zero'):")
    print(f"  global mean(|z|):            {abs_z.mean():.4f}")
    print(f"  global var(z):               {z_val.var():.4f}")
    print(f"  fraction near-zero (global): {(abs_z < args.threshold).mean():.4f}")
    print()
    print(f"Per-dimension summary ({n_dim} dims):")
    print(f"  variance:        min={per_dim_var.min():.4f}  med={np.median(per_dim_var):.4f}  max={per_dim_var.max():.4f}")
    print(f"  mean |z|:        min={per_dim_abs.min():.4f}  med={np.median(per_dim_abs):.4f}  max={per_dim_abs.max():.4f}")
    print(f"  near-zero rate:  min={sparsity.min():.4f}  med={np.median(sparsity):.4f}  max={sparsity.max():.4f}")

    # Dead dimensions: high near-zero rate
    dead = np.where(sparsity > 0.9)[0]
    print(f"  'dead' dims (>90% near-zero): {len(dead)}/{n_dim}  "
          f"{('idx: ' + str(dead.tolist())) if 0 < len(dead) <= 20 else ''}")
    weak = np.where(per_dim_var < 1e-4)[0]
    print(f"  'weak' dims (var < 1e-4):     {len(weak)}/{n_dim}  "
          f"{('idx: ' + str(weak.tolist())) if 0 < len(weak) <= 20 else ''}")


if __name__ == "__main__":
    main()
