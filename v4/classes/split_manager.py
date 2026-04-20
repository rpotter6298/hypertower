"""split_manager — generic stratified k-fold splitter.

SplitManager splits a DataFrame into train/val/test folds using an
outer/inner k-fold scheme.  The grouping identity is controlled by
``group_col``:

    group_col=None       — row-level splits (each row is its own identity)
    group_col="Patient ID" — group-level splits (all rows sharing a group
                             key land in the same fold)

The translation from a conceptual "identity level" to a concrete column
name belongs in the caller (typically the orchestrator), which has access
to the data bundle's column schema.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SplitPlan:
    train_ids: set[Any]
    val_ids:   set[Any]
    test_ids:  set[Any]


@dataclass
class Split:
    """One fold's train/val/test DataFrames."""
    train: pd.DataFrame
    val:   pd.DataFrame
    test:  Optional[pd.DataFrame] = None


# ---------------------------------------------------------------------------
# Core splitter
# ---------------------------------------------------------------------------

def _can_stratify(labels: np.ndarray, n_splits: int) -> bool:
    if labels.size == 0:
        return False
    unique, counts = np.unique(labels, return_counts=True)
    return len(unique) >= 2 and bool(np.all(counts >= n_splits))


def _build_split_plans(
    ids:      np.ndarray,
    labels:   np.ndarray,
    n_splits: int,
    seed:     int,
) -> list[SplitPlan]:
    """Outer/inner k-fold: test=fold k, val=fold (k+1)%n, train=remaining."""
    if ids.size == 0:
        raise ValueError("No samples available for splitting")
    if len(set(ids.tolist())) != ids.size:
        raise ValueError("ids must be unique")
    if n_splits < 3:
        raise ValueError("n_splits must be >= 3 for outer/inner k-fold")

    if _can_stratify(labels, n_splits):
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(splitter.split(ids, labels))
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(splitter.split(ids))

    fold_sets = [set(ids[test_idx].tolist()) for _, test_idx in folds]

    plans = []
    for k in range(n_splits):
        test_ids  = fold_sets[k]
        val_ids   = fold_sets[(k + 1) % n_splits]
        train_ids = set().union(*(fold_sets[j] for j in range(n_splits)
                                  if j != k and j != (k + 1) % n_splits))
        plans.append(SplitPlan(train_ids=train_ids, val_ids=val_ids, test_ids=test_ids))
    return plans


# ---------------------------------------------------------------------------
# SplitManager
# ---------------------------------------------------------------------------

class SplitManager:
    """Generic stratified k-fold split manager.

    Parameters
    ----------
    group_col : str | None
        Column whose values define the grouping identity for fold assignment.
        ``None`` splits on individual rows (no grouping).
    label_col : str | None
        Column used for stratification.  Resolved from the DataFrame at
        ``build_plans`` time if not provided here.
    """

    def __init__(
        self,
        group_col:  Optional[str] = None,
        label_col:  Optional[str] = None,
    ) -> None:
        self.group_col = group_col
        self.label_col = label_col

    def build_plans(
        self,
        df:       pd.DataFrame,
        *,
        n_splits: int  = 5,
        seed:     int  = 42,
        label_col: Optional[str] = None,
    ) -> list[Split]:
        """Build n_splits fold plans from df.

        Parameters
        ----------
        df        : full DataFrame (pre-filtered to the desired eval mode)
        n_splits  : number of folds (must be >= 3)
        seed      : random seed for reproducibility
        label_col : override for stratification column (falls back to
                    ``self.label_col``, then raises)
        """
        lc = label_col or self.label_col
        if lc is None:
            raise ValueError("label_col must be provided to build_plans or SplitManager")
        if lc not in df.columns:
            raise ValueError(f"label_col {lc!r} not found in DataFrame")

        df = df.copy().reset_index(drop=True)

        if self.group_col is None:
            # Row-level: each row is its own identity
            ids    = df.index.to_numpy()
            labels = df[lc].to_numpy()
            plans  = _build_split_plans(ids, labels, n_splits, seed)
            return [
                Split(
                    train=df[df.index.isin(p.train_ids)].reset_index(drop=True),
                    val  =df[df.index.isin(p.val_ids)  ].reset_index(drop=True),
                    test =df[df.index.isin(p.test_ids)  ].reset_index(drop=True),
                )
                for p in plans
            ]
        else:
            gc = self.group_col
            if gc not in df.columns:
                raise ValueError(f"group_col {gc!r} not found in DataFrame")
            # Group-level: collapse to one row per group, then split
            group_table = (
                df.groupby(gc, as_index=False)[lc]
                .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
                .rename(columns={lc: "_label"})
                .sort_values(gc)
                .reset_index(drop=True)
            )
            plans = _build_split_plans(
                group_table[gc].to_numpy(),
                group_table["_label"].to_numpy(),
                n_splits,
                seed,
            )
            return [
                Split(
                    train=df[df[gc].isin(p.train_ids)].reset_index(drop=True),
                    val  =df[df[gc].isin(p.val_ids)  ].reset_index(drop=True),
                    test =df[df[gc].isin(p.test_ids)  ].reset_index(drop=True),
                )
                for p in plans
            ]
