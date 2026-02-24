from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


class DataBundle:
    """
    Generic, torch-free container for metadata and file/label bookkeeping.

    Keeps feature typing, vectorization, and patient-level splits generic.
    Dataset-specific preprocessing (e.g., eye canonicalization) should live
    in the dataset builder (e.g., papila_builders in v2).
    """

    def __init__(
        self,
        *,
        image_dir: str,
        clinical_dir: Optional[str] = None,
        label_col: str,
        patient_col: str = "Patient ID",
        cat_cols: Optional[Iterable[str]] = None,
        max_unique_for_cat: int = 4,
        n_splits: int = 5,
        random_seed: int = 42,
        filename_template: str = "RET{pid:03d}{eye}.jpg",
        image_path_fn: Optional[Callable[[pd.Series], Path]] = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.label_col = label_col
        self.patient_col = patient_col
        self.max_unique_for_cat = max_unique_for_cat
        self.n_splits = n_splits
        self.filename_template = filename_template
        self.image_path_fn = image_path_fn
        self.clinical_dir = Path(clinical_dir) if clinical_dir else None

        # Internal state
        self.frames: List[pd.DataFrame] = []
        self.df: pd.DataFrame = pd.DataFrame()
        self.scalar_cols: List[str] = []
        self.cat_cols: List[str] = list(cat_cols) if cat_cols is not None else []
        self.scalar_stats: Dict[str, Dict[str, float]] = {}
        self.cat_maps: Dict[str, Dict[object, int]] = {}
        self.feature_dim: int = 0
        self.folds: Dict[int, Dict[str, List[object]]] = {}
        self.random_seed = int(random_seed)

    # ------------------- Public API -------------------
    def add_df(
        self,
        df: pd.DataFrame,
        *,
        id_column: Optional[str] = None,
        exclude_cols: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Add a dataframe and re-run typing, stats, and K-fold indices.
        QC rules:
          - Must have patient ID column; if not provided under that name, specify id_column.
        """
        df = df.copy()
        self._ensure_patient_id(df, id_column)
        if self.label_col not in df.columns:
            raise ValueError(f"label_col '{self.label_col}' not found in added dataframe")

        self.frames.append(df)
        self._refresh_master_df(exclude_cols=exclude_cols)
        self._infer_or_validate_feature_types(exclude_cols=exclude_cols)
        self._compute_numeric_stats()
        self._build_cat_maps()
        self._compute_feature_dim()
        self._build_kfold_indices()

    def get_split_ids(self, fold: int) -> Tuple[List[object], List[object]]:
        rec = self.folds.get(fold)
        if not rec:
            raise KeyError(f"Fold {fold} not available. Built folds: {sorted(self.folds.keys())}")
        return rec["train_ids"], rec["test_ids"]

    def get_split_dfs(self, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_ids, test_ids = self.get_split_ids(fold)
        train_df = self.df[self.df[self.patient_col].isin(train_ids)].reset_index(drop=True)
        test_df = self.df[self.df[self.patient_col].isin(test_ids)].reset_index(drop=True)
        return train_df, test_df

    def vectorize_row(self, row: pd.Series) -> np.ndarray:
        """Return a numpy feature vector (torch-free)."""
        feats: List[float] = []
        miss: List[float] = []
        # numeric
        for col in self.scalar_cols:
            v = pd.to_numeric(row.get(col), errors="coerce")
            if pd.isna(v):
                miss.append(1.0)
                v = self.scalar_stats[col]["median"]
            else:
                miss.append(0.0)
            lo = self.scalar_stats[col]["min"]
            hi = self.scalar_stats[col]["max"]
            feats.append((float(v) - lo) / (hi - lo) if hi > lo else 0.0)
        # categorical
        for col in self.cat_cols:
            mapping = self.cat_maps[col]
            one = [0.0] * len(mapping)
            key = row.get(col)
            one[mapping.get(key, 0)] = 1.0  # 0 is <UNK>
            feats.extend(one)
        # numeric missing flags
        feats.extend(miss)
        return np.asarray(feats, dtype=np.float32)

    def get_image_path(self, row: pd.Series) -> Path:
        if self.image_path_fn is not None:
            return Path(self.image_path_fn(row))
        pid = int(row[self.patient_col])
        eye = row.get("eyeID", "")
        if eye in ("OS", "OD"):
            eye_str = eye
        else:
            eye_str = str(eye)
        return self.image_dir / self.filename_template.format(pid=pid, eye=eye_str)

    def encode_metadata(self, row: pd.Series) -> np.ndarray:
        return self.vectorize_row(row)

    def get_label(self, row: pd.Series) -> int:
        return int(row[self.label_col])

    # ------------------- Internal helpers -------------------
    def _ensure_patient_id(self, df: pd.DataFrame, id_column: Optional[str]) -> None:
        if self.patient_col in df.columns:
            return
        if id_column and id_column in df.columns:
            df.rename(columns={id_column: self.patient_col}, inplace=True)
            return
        candidates = [
            c
            for c in df.columns
            if c.lower().replace(" ", "") in {"patientid", "patient", "pid"}
        ]
        if len(candidates) == 1:
            df.rename(columns={candidates[0]: self.patient_col}, inplace=True)
            return
        raise ValueError(
            f"A '{self.patient_col}' column is required; provide id_column=... if it has a different name."
        )

    def _refresh_master_df(self, exclude_cols: Optional[Iterable[str]] = None) -> None:
        self.df = pd.concat(self.frames, axis=0, ignore_index=True)
        if exclude_cols:
            self.df = self.df.drop(columns=[c for c in exclude_cols if c in self.df.columns])

    def _infer_or_validate_feature_types(self, exclude_cols: Optional[Iterable[str]] = None) -> None:
        excluded = set(exclude_cols or []) | {self.label_col, self.patient_col}
        feature_candidates = [c for c in self.df.columns if c not in excluded]
        cats = set(self.cat_cols) if self.cat_cols else set()
        scalars = set()
        for c in feature_candidates:
            if c in cats:
                continue
            s = self.df[c]
            as_num = pd.to_numeric(s, errors="coerce")
            num_missing = as_num.isna().mean()
            num_unique = s.dropna().nunique()
            if as_num.notna().any() and num_missing < 1.0 and num_unique > self.max_unique_for_cat:
                scalars.add(c)
            else:
                if num_unique <= self.max_unique_for_cat or as_num.isna().mean() > 0.0:
                    cats.add(c)
                else:
                    scalars.add(c)
        self.cat_cols = sorted(cats)
        self.scalar_cols = sorted(scalars)

    def _compute_numeric_stats(self) -> None:
        self.scalar_stats.clear()
        for col in self.scalar_cols:
            s = pd.to_numeric(self.df[col], errors="coerce")
            vals = s.dropna().astype(float).values
            if vals.size == 0:
                lo, hi, med = 0.0, 1.0, 0.0
            else:
                lo, hi = float(np.min(vals)), float(np.max(vals))
                med = float(np.median(vals))
                if hi <= lo:
                    hi = lo + 1.0
            self.scalar_stats[col] = {"min": lo, "max": hi, "median": med}

    def _build_cat_maps(self) -> None:
        self.cat_maps.clear()
        for col in self.cat_cols:
            cats = [v for v in self.df[col].dropna().unique().tolist()]
            try:
                cats = sorted(cats)
            except Exception:
                pass
            mapping = {"<UNK>": 0}
            for i, v in enumerate(cats, start=1):
                mapping[v] = i
            self.cat_maps[col] = mapping

    def _compute_feature_dim(self) -> None:
        self.feature_dim = len(self.scalar_cols) + sum(len(m) for m in self.cat_maps.values()) + len(self.scalar_cols)

    # ------------------- K-fold on unique patients -------------------
    def _build_kfold_indices(self) -> None:
        pats = self.df[self.patient_col].unique().tolist()
        labels_by_pat: Dict[object, object] = {}
        for pid, grp in self.df.groupby(self.patient_col):
            lab = grp[self.label_col].dropna()
            if len(lab) == 0:
                labels_by_pat[pid] = 0
            else:
                labels_by_pat[pid] = lab.mode().iloc[0]
        y_pat = np.array([labels_by_pat[p] for p in pats])

        try:
            from sklearn.model_selection import StratifiedGroupKFold

            sgkf = StratifiedGroupKFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_seed
            )
            split_iter = sgkf.split(X=pats, y=y_pat, groups=pats)
        except Exception:
            from sklearn.model_selection import StratifiedKFold

            skf = StratifiedKFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_seed
            )
            split_iter = skf.split(X=np.zeros(len(pats)), y=y_pat)

        self.folds.clear()
        for i, (train_idx, test_idx) in enumerate(split_iter):
            train_ids = [pats[j] for j in train_idx]
            test_ids = [pats[j] for j in test_idx]
            self.folds[i] = {"train_ids": train_ids, "test_ids": test_ids}
