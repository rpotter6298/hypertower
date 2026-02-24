"""Filter segmentation metrics rows with near-zero Dice scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Drop samples where both disc and cup Dice are below a threshold "
            "(default 0.01) and report how many were removed."
        )
    )
    parser.add_argument("input", type=Path, help="Path to metrics CSV to filter")
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination CSV. Defaults to <input stem>_filtered.csv in the same directory.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Dice cutoff; rows with both dice_disc and dice_cup below this are removed.",
    )
    parser.add_argument(
        "--keep-summary",
        action="store_true",
        help="Always keep summary rows (sample_id == '__mean__').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)

    mask_low = (df["dice_disc"] < args.threshold) & (df["dice_cup"] < args.threshold)
    if args.keep_summary and "sample_id" in df.columns:
        mask_low &= df["sample_id"].ne("__mean__")

    removed = int(mask_low.sum())
    filtered = df.loc[~mask_low].copy()

    # Recompute summary if original file contained one
    if "sample_id" in filtered.columns:
        summary_mask = filtered["sample_id"].eq("__mean__")
        filtered = filtered.loc[~summary_mask].copy()
        if not filtered.empty:
            summary = filtered[["dice_disc", "dice_cup"]].mean()
            summary_row = {
                "sample_id": "__mean__",
                "dataset": "summary",
                "split": "summary",
                "dice_disc": summary["dice_disc"],
                "dice_cup": summary["dice_cup"],
            }
            filtered = pd.concat([filtered, pd.DataFrame([summary_row])], ignore_index=True)

    remaining = len(filtered)

    output_path = args.output
    if output_path is None:
        output_path = args.input.with_name(f"{args.input.stem}_filtered.csv")

    filtered.to_csv(output_path, index=False)

    print(f"Removed rows: {removed}")
    print(f"Remaining rows: {remaining}")
    print(f"Filtered metrics saved to: {output_path}")


if __name__ == "__main__":
    main()
