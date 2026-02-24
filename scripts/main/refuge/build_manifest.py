"""Build manifest for U-Net segmenter combining REFUGE and Papila annotations."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional

import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classes.refuge_preprocessing import RefugePreprocessing

REFUGE_ROOT = Path("REFUGE")
PAPILA_IMAGES = Path("Papila/FundusImages")
PAPILA_CONTOURS = Path("Papila/ExpertsSegmentations/Contours")
DEFAULT_OUTPUT = Path("manifest.csv")


def pick_contour(base: str, kind: str) -> Optional[Path]:
    """Return contour path for Papila image (disc/cup)."""
    candidates = [
        PAPILA_CONTOURS / f"{base}_{kind}_exp2.txt",
        PAPILA_CONTOURS / f"{base}_{kind}_exp1.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def collect_refuge() -> pd.DataFrame:
    pre = RefugePreprocessing(REFUGE_ROOT)
    samples = []
    for sample in pre.build_manifest(refresh=True):
        if sample.mask_path is None:
            continue
        split = sample.split
        if split == "test":
            split = "holdout"
        samples.append(
            {
                "sample_id": sample.sample_id,
                "dataset": "refuge",
                "image_path": sample.image_path.resolve(),
                "annotation_disc": sample.mask_path.resolve(),
                "annotation_cup": sample.mask_path.resolve(),
                "annotation_type_disc": "mask",
                "annotation_type_cup": "mask",
                "split": split,
            }
        )
    return pd.DataFrame(samples)


def collect_papila() -> pd.DataFrame:
    samples = []
    if not PAPILA_IMAGES.exists():
        return pd.DataFrame(samples)
    for img_path in sorted(PAPILA_IMAGES.glob("RET*")):
        base = img_path.stem
        disc = pick_contour(base, "disc")
        cup = pick_contour(base, "cup")
        if disc is None or cup is None:
            continue
        samples.append(
            {
                "sample_id": f"papila_{base}",
                "dataset": "papila",
                "image_path": img_path.resolve(),
                "annotation_disc": disc.resolve(),
                "annotation_cup": cup.resolve(),
                "annotation_type_disc": "contour",
                "annotation_type_cup": "contour",
            }
        )
    return pd.DataFrame(samples)


def assign_splits(df: pd.DataFrame, holdout_ratio: float, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    df = df.copy()
    if "split" not in df.columns:
        df["split"] = None
    for dataset, group in df.groupby("dataset"):
        indices = list(group.index)

        # Preserve provided splits (e.g., REFUGE train/val/test); only populate
        # missing entries with "train" so downstream code has a default.
        split_series = df.loc[indices, "split"]
        missing = split_series.isna() | (split_series.astype(str).str.strip() == "")
        if missing.any():
            df.loc[missing[missing].index, "split"] = "train"
            split_series = df.loc[indices, "split"]

        if dataset != "papila":
            continue

        if holdout_ratio <= 0:
            continue

        desired_holdout = max(1, int(len(indices) * holdout_ratio))
        split_series = df.loc[indices, "split"]
        current_holdout_mask = split_series == "holdout"
        current_holdout = int(current_holdout_mask.sum())
        remaining = desired_holdout - current_holdout
        if remaining <= 0:
            continue

        candidate_indices = list(split_series[split_series == "train"].index)
        rng.shuffle(candidate_indices)
        selected = candidate_indices[:remaining]
        df.loc[selected, "split"] = "holdout"
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build U-Net manifest")
    parser.add_argument("--holdout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    refuge_df = collect_refuge()
    papila_df = collect_papila()
    combined = pd.concat([refuge_df, papila_df], ignore_index=True)
    combined = assign_splits(combined, holdout_ratio=args.holdout, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"Manifest saved to {args.output} with {len(combined)} entries")


if __name__ == "__main__":
    main()
