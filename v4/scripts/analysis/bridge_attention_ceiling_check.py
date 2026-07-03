"""Variance decomposition: is the high-backbone end of the bridge_attention
sweep hitting a dataset ceiling?

For each (rep, fold) cell, we have 12 hb_test_auc measurements — one per
(bridge × backbone) condition. We decompose the variance two ways and compare
between the LOW-backbone and HIGH-backbone halves of the gradient:

    across-arch variance @ fixed (rep,fold)
        = var across the conditions in this subset, for the same fold split
        (small → architectures are interchangeable at this capacity)
    across-fold variance @ fixed architecture
        = var across the 50 fold-reps, for one condition
        (small → the fold split doesn't matter much)

If at the high-backbone end, across-fold dwarfs across-arch, the dataset's
fold-assignment noise dominates the architectural choice — i.e. all the
strong configurations are hitting the same ceiling.

Reads summary.json files directly, no inference needed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SWEEP_ROOT = Path("v4/results/experiments/bridge_attention")

LOW_BACKBONES  = ["mobilenet_v2", "resnet50", "efficientnet_b0"]
HIGH_BACKBONES = ["efficientnet_v2_m", "refugelike", "refuge_efficientnet_v2_m"]
BRIDGES = ["gated", "ortho"]


def collect_long() -> pd.DataFrame:
    rows = []
    for bridge in BRIDGES:
        for bb in LOW_BACKBONES + HIGH_BACKBONES:
            run_dir = SWEEP_ROOT / f"{bridge}_{bb}"
            if not run_dir.exists():
                continue
            for s in sorted(run_dir.glob("rep*/binary/summary.json")):
                rep = int(s.parents[1].name.replace("rep", ""))
                d = json.loads(s.read_text())
                for fr in d.get("fold_results", []):
                    v = fr.get("hb_test_auc")
                    if v is None or not np.isfinite(v):
                        continue
                    rows.append({
                        "bridge": bridge,
                        "backbone": bb,
                        "rep": rep,
                        "fold": fr["fold"],
                        "auc": float(v),
                    })
    return pd.DataFrame(rows)


def decompose(df: pd.DataFrame, label: str) -> None:
    # condition = bridge × backbone tuple
    df = df.copy()
    df["condition"] = df["bridge"] + "/" + df["backbone"]
    n_cond = df["condition"].nunique()
    n_cells = df.groupby(["rep", "fold"]).ngroups

    # Across-architecture variance @ fixed (rep, fold)
    grouped_cell = df.groupby(["rep", "fold"])["auc"]
    cell_var = grouped_cell.var(ddof=1)         # one var per (rep,fold) cell
    cell_std_mean = float(np.sqrt(cell_var.mean())) if not cell_var.empty else float("nan")

    # Across-fold-rep variance @ fixed architecture
    grouped_arch = df.groupby("condition")["auc"]
    arch_var = grouped_arch.var(ddof=1)
    arch_std_mean = float(np.sqrt(arch_var.mean())) if not arch_var.empty else float("nan")

    ratio = arch_std_mean / cell_std_mean if cell_std_mean > 0 else float("inf")
    mean_auc = float(df["auc"].mean())
    print(f"── {label} ──  (n_conditions={n_cond}, n_cells={n_cells}, n_obs={len(df)})")
    print(f"  mean AUC across all (cond, rep, fold) ........ {mean_auc:.4f}")
    print(f"  across-arch SD @ fixed (rep,fold) ............ {cell_std_mean:.4f}  "
          f"← architectural spread within the same fold")
    print(f"  across-foldrep SD @ fixed architecture ....... {arch_std_mean:.4f}  "
          f"← fold-assignment noise within a single architecture")
    print(f"  ratio  arch-SD / fold-SD ..................... {ratio:>6.2f}x  "
          f"({'fold noise dominates' if ratio > 2 else 'arch + fold comparable'})")
    print()


def main():
    df = collect_long()
    if df.empty:
        print("No data — has the full readout finished yet?")
        return

    n_cond = df["bridge"].nunique() * df["backbone"].nunique()
    print(f"Collected {len(df)} (cond, rep, fold) AUC observations from "
          f"{n_cond} (bridge × backbone) conditions\n")

    low  = df[df["backbone"].isin(LOW_BACKBONES)]
    high = df[df["backbone"].isin(HIGH_BACKBONES)]
    decompose(low,  "LOW backbones  (mobilenet_v2, resnet50, efficientnet_b0)")
    decompose(high, "HIGH backbones (efficientnet_v2_m, refugelike, refuge_v2m)")

    # Headline interpretation
    print("──────────────────────────────────────────────────────")
    print("Interpretation:")
    print("  - If both subsets show high arch-SD: architectures genuinely differ.")
    print("  - If LOW shows high arch-SD but HIGH shows low arch-SD: ceiling effect")
    print("    at the high-backbone end — all strong configurations hit the same wall.")
    print("  - Within each subset, ratio = fold-SD / arch-SD: when fold noise")
    print("    dominates by >2x, the cell-to-cell variation between architectures")
    print("    is smaller than the noise floor introduced by patient assignment.")


if __name__ == "__main__":
    main()
