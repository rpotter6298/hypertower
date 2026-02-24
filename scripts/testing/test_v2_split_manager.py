#!/usr/bin/env python3
"""Tiny smoke test for classes.v2.split_manager."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

# Ensure repo root is importable when running as: python3 scripts/test_v2_split_manager.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classes import build_papila_clinical
from classes.v2 import build_papila_profile
from classes.v2.split_manager import PatientFirstSplitManager, build_patient_split_plans


class _ClinicalStub:
    def __init__(self, df: pd.DataFrame, label_col: str = "Diagnosis") -> None:
        self.df = df
        self.label_col = label_col


def _make_fake_df(n_patients: int, n_classes: int, label_col: str) -> pd.DataFrame:
    rows = []
    for pid in range(1, n_patients + 1):
        label = (pid - 1) % n_classes
        for eye in ("OD", "OS"):
            rows.append(
                {
                    "Patient ID": pid,
                    "eyeID": eye,
                    label_col: label,
                    "dummy_feature": float(pid),
                }
            )
    return pd.DataFrame(rows)


def _summarize_fold(
    fold: int,
    split,
    label_col: str,
    expected_rows_per_patient: int | None = None,
) -> str:
    train_ids = set(split.train["Patient ID"].tolist())
    val_ids = set(split.val["Patient ID"].tolist())
    holdout_ids = set(split.holdout["Patient ID"].tolist()) if split.holdout is not None else set()

    if train_ids & val_ids:
        raise RuntimeError(f"Fold {fold}: train/val overlap detected")
    if train_ids & holdout_ids:
        raise RuntimeError(f"Fold {fold}: train/holdout overlap detected")
    if val_ids & holdout_ids:
        raise RuntimeError(f"Fold {fold}: val/holdout overlap detected")

    if expected_rows_per_patient is not None:
        # Used only for synthetic data where we know OD+OS are both present.
        for name, df in (("train", split.train), ("val", split.val), ("holdout", split.holdout)):
            if df is None or df.empty:
                continue
            counts = df.groupby("Patient ID").size().unique().tolist()
            if counts != [expected_rows_per_patient]:
                raise RuntimeError(f"Fold {fold}: {name} has broken per-patient row grouping: {counts}")

    train_cls = dict(split.train.groupby(label_col).size().to_dict())
    val_cls = dict(split.val.groupby(label_col).size().to_dict())
    hold_cls = dict(split.holdout.groupby(label_col).size().to_dict()) if split.holdout is not None else {}
    return (
        f"fold={fold} "
        f"train_patients={len(train_ids)} val_patients={len(val_ids)} holdout_patients={len(holdout_ids)} "
        f"train_rows={len(split.train)} val_rows={len(split.val)} holdout_rows={0 if split.holdout is None else len(split.holdout)} "
        f"train_class_rows={train_cls} val_class_rows={val_cls} holdout_class_rows={hold_cls}"
    )


def _confirm_holdout_consistency_and_exclusion(splits) -> None:
    holdout_sets: list[set] = []
    val_union: set = set()
    for split in splits:
        holdout_ids = set(split.holdout["Patient ID"].tolist()) if split.holdout is not None else set()
        holdout_sets.append(holdout_ids)
        val_union.update(split.val["Patient ID"].tolist())

    # A) Holdout should be the same patients across all folds.
    baseline = holdout_sets[0] if holdout_sets else set()
    for i, holdout_ids in enumerate(holdout_sets):
        if holdout_ids != baseline:
            raise RuntimeError(
                f"Holdout mismatch: fold 0 has {sorted(baseline)}, fold {i} has {sorted(holdout_ids)}"
            )

    # B) Holdout patients should never appear in any validation/test fold.
    overlap = baseline & val_union
    if overlap:
        raise RuntimeError(f"Holdout patients found in val/test sets: {sorted(overlap)}")

    print(
        "Holdout checks: OK "
        f"(constant across folds, holdout_patients={len(baseline)}, overlap_with_any_val=0)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke test PatientFirstSplitManager with synthetic data.")
    ap.add_argument("--dataset", choices=["papila", "synthetic"], default="papila")
    ap.add_argument(
        "--patients",
        type=int,
        default=30,
        help="Synthetic mode only: number of fake patients to generate.",
    )
    ap.add_argument(
        "--synthetic-classes",
        type=int,
        default=3,
        help="Synthetic mode only: number of classes to generate.",
    )
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--holdout-per-class", type=int, default=1)
    ap.add_argument("--fold-seed", type=int, default=42)
    ap.add_argument("--holdout-seed", type=int, default=123)
    ap.add_argument("--image-dir", default="Papila/FundusImages")
    ap.add_argument("--clinical-dir", default="Papila/ClinicalData")
    ap.add_argument("--label-col", default="Diagnosis")
    ap.add_argument(
        "--sample-mode",
        choices=["patient", "eye"],
        default="patient",
        help="Build samples per patient (multi-slot) or per eye (row-level).",
    )
    ap.add_argument(
        "--cat-cols",
        nargs="*",
        default=["Gender", "Phakic/Pseudophakic"],
        help="Categorical columns for PAPILA builder.",
    )
    args = ap.parse_args()

    if args.synthetic_classes < 2:
        raise SystemExit("--synthetic-classes must be >= 2")

    expected_rows_per_patient: int | None = None
    if args.dataset == "papila":
        clinical = build_papila_clinical(
            image_dir=args.image_dir,
            clinical_dir=args.clinical_dir,
            label_col=args.label_col,
            cat_cols=args.cat_cols,
            n_splits=args.n_splits,
            random_seed=args.fold_seed,
        )
        df = clinical.df.copy()
        print(
            f"Loaded PAPILA dataframe: rows={len(df)} patients={df['Patient ID'].nunique()} "
            f"labels={dict(df.groupby(args.label_col).size().to_dict())}"
        )
    else:
        df = _make_fake_df(args.patients, args.synthetic_classes, args.label_col)
        clinical = _ClinicalStub(df=df, label_col=args.label_col)
        expected_rows_per_patient = 2
        print(
            f"Loaded synthetic dataframe: rows={len(df)} patients={df['Patient ID'].nunique()} "
            f"labels={dict(df.groupby(args.label_col).size().to_dict())}"
        )
    n_classes = int(df[args.label_col].nunique())
    print(f"Detected classes from dataframe: n_classes={n_classes}")

    split_args = SimpleNamespace(
        eval_mode="multiclass",
        holdout_per_class=args.holdout_per_class,
        holdout_seed=args.holdout_seed,
        n_splits=args.n_splits,
        fold_seed=args.fold_seed,
    )

    manager = PatientFirstSplitManager(patient_col="Patient ID", label_col=args.label_col)
    profile = build_papila_profile(
        patient_col="Patient ID",
        label_col=args.label_col,
        sample_mode=args.sample_mode,
    )
    splits = manager.build_plans(clinical=clinical, args=split_args, profile=profile)

    print("=== Adapter split manager output ===")
    for fold, split in enumerate(splits):
        print(
            _summarize_fold(
                fold,
                split,
                label_col=args.label_col,
                expected_rows_per_patient=expected_rows_per_patient,
            )
        )
    _confirm_holdout_consistency_and_exclusion(splits)

    # Also smoke-test the pure vector API directly.
    patient_labels = df.groupby("Patient ID")[args.label_col].first()
    plans = build_patient_split_plans(
        patient_ids=patient_labels.index.to_numpy(),
        patient_labels=patient_labels.to_numpy(),
        n_splits=args.n_splits,
        seed=args.fold_seed,
        holdout_per_class=args.holdout_per_class,
        holdout_seed=args.holdout_seed,
    )
    print(f"\nVector API produced {len(plans)} fold plans.")
    print("OK: split manager smoke test passed.")


if __name__ == "__main__":
    main()
