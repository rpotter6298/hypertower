#!/usr/bin/env python3
"""Describe PAPILA splits and V2 slot-based loaders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

# Ensure repo root is importable when running as: python3 scripts/test_v2_papila_loaders.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classes import build_papila_clinical
from classes.v2 import build_papila_profile, PatientFirstSplitManager, SlotLoaderFactory


def _describe_split(name: str, df, label_col: str) -> str:
    if df is None or df.empty:
        return f"{name}: empty"
    patient_ids = set(df["Patient ID"].tolist())
    class_counts = dict(df.groupby(label_col).size().to_dict())
    return (
        f"{name}: patients={len(patient_ids)} rows={len(df)} "
        f"class_rows={class_counts}"
    )


def _describe_batch(batch: dict) -> list[str]:
    lines = []
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            lines.append(f"{key}: tensor shape={tuple(val.shape)} dtype={val.dtype}")
        elif isinstance(val, list):
            non_none = next((v for v in val if v is not None), None)
            lines.append(
                f"{key}: list len={len(val)} sample_type={type(non_none).__name__ if non_none is not None else 'None'}"
            )
        else:
            lines.append(f"{key}: {type(val).__name__}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Describe PAPILA splits + V2 slot-based loaders.")
    ap.add_argument("--image-dir", default="Papila/FundusImages")
    ap.add_argument("--clinical-dir", default="Papila/ClinicalData")
    ap.add_argument("--label-col", default="Diagnosis")
    ap.add_argument(
        "--cat-cols",
        nargs="*",
        default=["Gender", "Phakic/Pseudophakic"],
        help="Categorical columns for PAPILA builder.",
    )
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--fold-seed", type=int, default=42)
    ap.add_argument("--holdout-per-class", type=int, default=1)
    ap.add_argument("--holdout-seed", type=int, default=123)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0, help="Which fold to inspect in detail.")
    ap.add_argument(
        "--sample-mode",
        choices=["patient", "eye"],
        default="patient",
        help="Build samples per patient (multi-slot) or per eye (row-level).",
    )
    args = ap.parse_args()

    clinical = build_papila_clinical(
        image_dir=args.image_dir,
        clinical_dir=args.clinical_dir,
        label_col=args.label_col,
        cat_cols=args.cat_cols,
        n_splits=args.n_splits,
        random_seed=args.fold_seed,
    )
    profile = build_papila_profile(
        patient_col="Patient ID",
        label_col=args.label_col,
        sample_mode=args.sample_mode,
    )

    print("=== PAPILA profile slots ===")
    for key, desc in profile.slot_descriptors().items():
        print(f"{key}: kind={desc.kind} required={desc.required} desc={desc.description}")
    print("aliases:", profile.semantic_aliases())

    split_args = SimpleNamespace(
        eval_mode="multiclass",
        holdout_per_class=args.holdout_per_class,
        holdout_seed=args.holdout_seed,
        n_splits=args.n_splits,
        fold_seed=args.fold_seed,
    )
    split_manager = PatientFirstSplitManager(patient_col="Patient ID", label_col=args.label_col)
    plans = split_manager.build_plans(clinical=clinical, args=split_args, profile=profile)

    print("\n=== Split summaries ===")
    for i, split in enumerate(plans):
        print(f"fold {i}:")
        print("  " + _describe_split("train", split.train, args.label_col))
        print("  " + _describe_split("val", split.val, args.label_col))
        print("  " + _describe_split("holdout", split.holdout, args.label_col))

    if args.fold < 0 or args.fold >= len(plans):
        raise SystemExit(f"Requested fold {args.fold} but only {len(plans)} folds are available")
    split = plans[args.fold]

    loader_factory = SlotLoaderFactory(num_workers=args.num_workers)
    loaders = loader_factory.build(
        clinical=clinical,
        split=split,
        args=SimpleNamespace(batch_size=args.batch_size),
        fold=args.fold,
        profile=profile,
    )

    print(f"\n=== Loader inspection (fold {args.fold}) ===")
    for name, loader in (("train", loaders.train), ("val", loaders.val), ("holdout", loaders.holdout)):
        if loader is None:
            print(f"{name}: None")
            continue
        print(f"{name}: batches={len(loader)} batch_size={loader.batch_size}")
        batch = next(iter(loader))
        for line in _describe_batch(batch):
            print(f"  {line}")


if __name__ == "__main__":
    main()
