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
    holdout_patient_ids: set[Any]


def build_patient_split_plans(
    patient_ids: Iterable[Any],
    patient_labels: Iterable[Any],
    *,
    n_splits: int,
    seed: int,
    holdout_per_class: int = 0,
    holdout_seed: int = 123,
) -> list[SplitPlan]:
    """
    Core vector-based splitter.

    Inputs are one row per patient:
    - patient_ids: unique patient IDs
    - patient_labels: one label per patient
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
        raise ValueError("patient_ids must be unique (one label per patient)")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    holdout_ids: set[Any] = set()
    if holdout_per_class > 0:
        rng = np.random.default_rng(holdout_seed)
        for label in np.unique(labels):
            idx = np.where(labels == label)[0]
            if idx.size == 0:
                continue
            n = min(holdout_per_class, idx.size)
            chosen = rng.choice(idx, size=n, replace=False)
            holdout_ids.update(ids[chosen].tolist())

    keep_mask = ~np.isin(ids, list(holdout_ids))
    cv_ids = ids[keep_mask]
    cv_labels = labels[keep_mask]
    if cv_ids.size < n_splits:
        raise ValueError(
            f"Not enough patients ({cv_ids.size}) for n_splits={n_splits} after holdout removal"
        )

    use_stratified = _can_stratify(cv_labels, n_splits)
    if use_stratified:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(cv_ids, cv_labels))
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(cv_ids))

    plans: list[SplitPlan] = []
    for train_idx, val_idx in splits:
        plans.append(
            SplitPlan(
                train_patient_ids=set(cv_ids[train_idx].tolist()),
                val_patient_ids=set(cv_ids[val_idx].tolist()),
                holdout_patient_ids=set(holdout_ids),
            )
        )
    return plans


class PatientFirstSplitManager:
    """
    Patient-level splitter for V2.

    Behavior:
    - Optional binary filtering happens first (labels in {0,1} only).
    - Optional holdout is sampled at the patient level (never per-eye rows).
    - K-fold split is built on remaining patients.
    - Returned dataframes contain all rows for each selected patient.
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
        profile_patient_col = getattr(profile, "patient_col", None) if profile is not None else None
        patient_col = profile_patient_col or self.patient_col
        label_col = self.label_col or profile_label_col or getattr(clinical, "label_col", None)
        if label_col is None:
            raise ValueError("Could not resolve label column from SplitManager or clinical.label_col")

        if not hasattr(clinical, "df"):
            raise ValueError("Clinical object must expose a dataframe at .df")
        df_full = clinical.df.copy()
        self._validate_columns(df_full, label_col, patient_col=patient_col)

        eval_mode = str(getattr(args, "eval_mode", "multiclass")).lower()
        if eval_mode == "binary":
            df_full = df_full[df_full[label_col].isin([0, 1])].reset_index(drop=True)

        holdout_per_class = int(getattr(args, "holdout_per_class", 0) or 0)
        holdout_seed = int(getattr(args, "holdout_seed", 123))
        n_splits = int(getattr(args, "n_splits", 5))
        fold_seed = int(getattr(args, "fold_seed", 42))

        patient_table = self._patient_label_table(df_full, label_col, patient_col=patient_col)
        plans = build_patient_split_plans(
            patient_ids=patient_table[patient_col].to_numpy(),
            patient_labels=patient_table["_label"].to_numpy(),
            n_splits=n_splits,
            seed=fold_seed,
            holdout_per_class=holdout_per_class,
            holdout_seed=holdout_seed,
        )

        out: list[PatientSplit] = []
        for plan in plans:
            train_df = (
                df_full[df_full[patient_col].isin(plan.train_patient_ids)]
                .reset_index(drop=True)
            )
            val_df = (
                df_full[df_full[patient_col].isin(plan.val_patient_ids)]
                .reset_index(drop=True)
            )
            holdout_df = None
            if plan.holdout_patient_ids:
                holdout_df = (
                    df_full[df_full[patient_col].isin(plan.holdout_patient_ids)]
                    .reset_index(drop=True)
                )
            out.append(PatientSplit(train=train_df, val=val_df, holdout=holdout_df))
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
