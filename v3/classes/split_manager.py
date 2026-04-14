"""V3 split manager — proper outer/inner k-fold CV.

Outer fold k = test set.
Val = outer fold (k+1) % n_splits (rotated).
Train = remaining n_splits-2 folds.

Every patient appears in test exactly once and in val exactly once.
No pre-carved holdout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from .network_manager import PatientSplit


@dataclass(frozen=True)
class SplitPlan:
    train_patient_ids: set[Any]
    val_patient_ids: set[Any]
    test_patient_ids: set[Any]


def build_patient_split_plans(
    patient_ids: Iterable[Any],
    patient_labels: Iterable[Any],
    *,
    n_splits: int,
    seed: int,
) -> list[SplitPlan]:
    """
    Outer/inner k-fold splitter.

    For each outer fold k:
      - test  = patients in fold k
      - val   = patients in fold (k+1) % n_splits
      - train = patients in remaining n_splits-2 folds
    """
    ids = np.asarray(list(patient_ids))
    labels = np.asarray(list(patient_labels))
    if ids.ndim != 1 or labels.ndim != 1:
        raise ValueError("patient_ids and patient_labels must be 1D arrays")
    if ids.size != labels.size:
        raise ValueError(f"Length mismatch: ids={ids.size}, labels={labels.size}")
    if ids.size == 0:
        raise ValueError("No patients available for splitting")
    if len(set(ids.tolist())) != ids.size:
        raise ValueError("patient_ids must be unique")
    if n_splits < 3:
        raise ValueError("n_splits must be >= 3 for outer/inner k-fold")

    use_stratified = _can_stratify(labels, n_splits)
    if use_stratified:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        outer_folds = list(splitter.split(ids, labels))
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        outer_folds = list(splitter.split(ids))

    # Build index sets for each outer fold
    fold_index_sets: list[set] = []
    for _, test_idx in outer_folds:
        fold_index_sets.append(set(ids[test_idx].tolist()))

    plans: list[SplitPlan] = []
    for k in range(n_splits):
        test_ids  = fold_index_sets[k]
        val_ids   = fold_index_sets[(k + 1) % n_splits]
        train_ids: set = set()
        for j in range(n_splits):
            if j != k and j != (k + 1) % n_splits:
                train_ids |= fold_index_sets[j]
        plans.append(SplitPlan(
            train_patient_ids=train_ids,
            val_patient_ids=val_ids,
            test_patient_ids=test_ids,
        ))
    return plans


class EyeLevelSplitManager:
    """
    Eye-level (leaky) splitter — splits on individual eye rows, ignoring
    patient grouping.  Same patient's eyes can appear in different folds.
    Used to demonstrate the effect of data leakage.
    """

    def __init__(
        self,
        *,
        patient_col: str = "Patient ID",
        label_col: Optional[str] = None,
    ) -> None:
        self.patient_col = patient_col
        self.label_col = label_col

    def build_plans(
        self,
        *,
        clinical: Any,
        args: Any,
        profile: Optional[Any] = None,
    ) -> list[PatientSplit]:
        profile_label_col = getattr(profile, "label_col", None) if profile is not None else None
        label_col = self.label_col or profile_label_col or getattr(clinical, "label_col", None)
        if label_col is None:
            raise ValueError("Could not resolve label column")

        if not hasattr(clinical, "df"):
            raise ValueError("Clinical object must expose a dataframe at .df")
        df_full = clinical.df.copy().reset_index(drop=True)

        eval_mode = str(getattr(args, "eval_mode", "multiclass")).lower()
        if eval_mode == "binary":
            df_full = df_full[df_full[label_col].isin([0, 1])].reset_index(drop=True)

        n_splits  = int(getattr(args, "n_splits", 5))
        fold_seed = int(getattr(args, "fold_seed", 42))

        labels = df_full[label_col].to_numpy()
        eye_ids = df_full.index.to_numpy()

        # Reuse build_patient_split_plans with eye-row IDs as the "patients"
        plans = build_patient_split_plans(
            patient_ids=eye_ids,
            patient_labels=labels,
            n_splits=n_splits,
            seed=fold_seed,
        )

        out: list[PatientSplit] = []
        for plan in plans:
            train_df = df_full[df_full.index.isin(plan.train_patient_ids)].reset_index(drop=True)
            val_df   = df_full[df_full.index.isin(plan.val_patient_ids)].reset_index(drop=True)
            test_df  = df_full[df_full.index.isin(plan.test_patient_ids)].reset_index(drop=True)
            out.append(PatientSplit(train=train_df, val=val_df, test=test_df))
        return out


class PatientFirstSplitManager:
    """Patient-level splitter for V3. Outer/inner k-fold, no holdout."""

    def __init__(
        self,
        *,
        patient_col: str = "Patient ID",
        label_col: Optional[str] = None,
    ) -> None:
        self.patient_col = patient_col
        self.label_col = label_col

    def build_plans(
        self,
        *,
        clinical: Any,
        args: Any,
        profile: Optional[Any] = None,
    ) -> list[PatientSplit]:
        profile_label_col   = getattr(profile, "label_col",   None) if profile is not None else None
        profile_patient_col = getattr(profile, "patient_col", None) if profile is not None else None
        patient_col = profile_patient_col or self.patient_col
        label_col = self.label_col or profile_label_col or getattr(clinical, "label_col", None)
        if label_col is None:
            raise ValueError("Could not resolve label column")

        if not hasattr(clinical, "df"):
            raise ValueError("Clinical object must expose a dataframe at .df")
        df_full = clinical.df.copy()
        self._validate_columns(df_full, label_col, patient_col=patient_col)

        eval_mode = str(getattr(args, "eval_mode", "multiclass")).lower()
        if eval_mode == "binary":
            df_full = df_full[df_full[label_col].isin([0, 1])].reset_index(drop=True)

        n_splits  = int(getattr(args, "n_splits", 5))
        fold_seed = int(getattr(args, "fold_seed", 42))

        patient_table = self._patient_label_table(df_full, label_col, patient_col=patient_col)
        plans = build_patient_split_plans(
            patient_ids=patient_table[patient_col].to_numpy(),
            patient_labels=patient_table["_label"].to_numpy(),
            n_splits=n_splits,
            seed=fold_seed,
        )

        out: list[PatientSplit] = []
        for plan in plans:
            train_df = df_full[df_full[patient_col].isin(plan.train_patient_ids)].reset_index(drop=True)
            val_df   = df_full[df_full[patient_col].isin(plan.val_patient_ids)].reset_index(drop=True)
            test_df  = df_full[df_full[patient_col].isin(plan.test_patient_ids)].reset_index(drop=True)
            out.append(PatientSplit(train=train_df, val=val_df, test=test_df))
        return out

    def _validate_columns(self, df: pd.DataFrame, label_col: str, patient_col: Optional[str] = None) -> None:
        pcol = patient_col or self.patient_col
        if pcol not in df.columns:
            raise ValueError(f"Missing required patient column: {pcol!r}")
        if label_col not in df.columns:
            raise ValueError(f"Missing required label column: {label_col!r}")

    def _patient_label_table(
        self,
        df: pd.DataFrame,
        label_col: str,
        patient_col: Optional[str] = None,
    ) -> pd.DataFrame:
        pcol = patient_col or self.patient_col
        grouped = (
            df.groupby(pcol, as_index=False)[label_col]
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
            .rename(columns={label_col: "_label"})
            .sort_values(pcol)
            .reset_index(drop=True)
        )
        if grouped.empty:
            raise ValueError("No patients available for splitting")
        return grouped


def _can_stratify(labels: np.ndarray, n_splits: int) -> bool:
    if labels.size == 0:
        return False
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2:
        return False
    return bool(np.all(counts >= n_splits))
