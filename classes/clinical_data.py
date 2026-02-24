# clinical_data.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, Dict, List, Tuple
import numpy as np
import pandas as pd

class ClinicalData:
    """
    Torch-free container for clinical metadata and file/label bookkeeping.
    - Holds one or more dataframes (via add_df) and harmonizes columns
    - Canonical IDs: 'Patient ID' must exist (or be specified and will be renamed)
    - Canonical eye column: 'eyeID' recoded to 'OS'/'OD' if present; if absent, set to 0
    - Feature typing (if cat_cols not provided):
        * Categorical if (a) <= max_unique categorical threshold (default 4), or
          (b) values cannot be coerced to float; otherwise numeric (scalar)
    - Scaling/imputation:
        * Numeric: min–max to [0,1], median imputation; + one missing flag per numeric feature
        * Categorical: one-hot with '<UNK>' bucket at index 0
    - Patient-level K-fold indices stored as dict: folds[k] -> {'train_ids': [...], 'test_ids': [...]}
    """

    def __init__(
        self,
        image_dir: str,
        clinical_dir: Optional[str],
        label_col: str,
        # typing / detection
        cat_cols: Optional[Iterable[str]] = None,
        max_unique_for_cat: int = 4,
        # splitting
        n_splits: int = 5,
        random_seed: int = 42,
    ):
        self.image_dir = Path(image_dir)
        self.clinical_dir = Path(clinical_dir) if clinical_dir else None
        self.label_col = label_col
        self.max_unique_for_cat = max_unique_for_cat
        self.n_splits = n_splits

        # Internal state
        self.frames: List[pd.DataFrame] = []         # raw frames as added
        self.df: pd.DataFrame = pd.DataFrame()        # concatenated
        self.scalar_cols: List[str] = []
        self.cat_cols: List[str] = list(cat_cols) if cat_cols is not None else []
        self.scalar_stats: Dict[str, Dict[str, float]] = {}
        self.cat_maps: Dict[str, Dict[object, int]] = {}
        self.feature_dim: int = 0
        self.folds: Dict[int, Dict[str, List[object]]] = {}  # fold -> {'train_ids': [], 'test_ids': []}
        self.random_seed = int(random_seed)

    # ------------------- Public API -------------------
    def add_df(
        self,
        df: pd.DataFrame,
        id_column: Optional[str] = None,
        eye_column: Optional[str] = None,
        exclude_cols: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Add a dataframe and re-run harmonization, typing, stats, and K-fold indices.
        QC rules:
          - Must have patient ID column; if not provided under that name, specify id_column.
          - eyeID, if present, must be binary; recoded to 'OS'/'OD'. If absent, create and set to 0.
        """
        df = df.copy()
        # --- QC: Patient ID ---
        pid_col = self._ensure_patient_id(df, id_column)
        # --- QC: eyeID ---
        self._canonicalize_eye_column(df, eye_column)
        # --- Normalize label presence ---
        if self.label_col not in df.columns:
            raise ValueError(f"label_col '{self.label_col}' not found in added dataframe")

        # append & refresh
        self.frames.append(df)
        self._refresh_master_df(exclude_cols=exclude_cols)
        self._infer_or_validate_feature_types(exclude_cols=exclude_cols)
        self._compute_numeric_stats()
        self._build_cat_maps()
        self._compute_feature_dim()
        self._build_kfold_indices()

    def get_split_ids(self, fold: int) -> Tuple[List[object], List[object]]:
        rec = self.folds.get(fold)
        if not rec: raise KeyError(f"Fold {fold} not available. Built folds: {sorted(self.folds.keys())}")
        return rec['train_ids'], rec['test_ids']

    def get_split_dfs(self, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_ids, test_ids = self.get_split_ids(fold)
        train_df = self.df[self.df['Patient ID'].isin(train_ids)].reset_index(drop=True)
        test_df  = self.df[self.df['Patient ID'].isin(test_ids)].reset_index(drop=True)
        return train_df, test_df

    def vectorize_row(self, row: pd.Series) -> np.ndarray:
        """Return a numpy feature vector (torch-free)."""
        feats: List[float] = []
        miss: List[float] = []
        # numeric
        for col in self.scalar_cols:
            v = pd.to_numeric(row.get(col), errors='coerce')
            if pd.isna(v):
                miss.append(1.0)
                v = self.scalar_stats[col]['median']
            else:
                miss.append(0.0)
            lo = self.scalar_stats[col]['min']; hi = self.scalar_stats[col]['max']
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

    def get_image_path(self, row: pd.Series, filename_template: str = "RET{pid:03d}{eye}.jpg") -> Path:
        pid = int(row['Patient ID']); eye = row.get('eyeID', 0)
        if eye in ("OS", "OD"):
            eye_str = eye
        else:
            eye_str = str(eye)
        return self.image_dir / filename_template.format(pid=pid, eye=eye_str)

    # ------------------- Internal helpers -------------------
    def _ensure_patient_id(self, df: pd.DataFrame, id_column: Optional[str]) -> str:
        if 'Patient ID' in df.columns:
            return 'Patient ID'
        if id_column and id_column in df.columns:
            df.rename(columns={id_column: 'Patient ID'}, inplace=True)
            return 'Patient ID'
        # try auto-detect common variants
        candidates = [c for c in df.columns if c.lower().replace(" ", "") in {"patientid","patient","pid"}]
        if len(candidates) == 1:
            df.rename(columns={candidates[0]: 'Patient ID'}, inplace=True)
            return 'Patient ID'
        raise ValueError("A 'Patient ID' column is required; provide id_column=... if it has a different name.")

    def _canonicalize_eye_column(self, df: pd.DataFrame, eye_column: Optional[str]) -> None:
        # Find source
        src = None
        if 'eyeID' in df.columns: src = 'eyeID'
        elif eye_column and eye_column in df.columns: src = eye_column
        else:
            # try auto detect
            for c in df.columns:
                if 'eye' in c.lower():
                    src = c; break
        if src is None:
            df['eyeID'] = 0
            return
        # Map to OS/OD
        s = df[src]
        def norm(v):
            if pd.isna(v): return None
            x = str(v).strip().upper()
            if x in {"OS","L","LEFT","0"}: return "OS"
            if x in {"OD","R","RIGHT","1"}: return "OD"
            # numbers like 2? fall back by parity
            try:
                num = int(float(x))
                return "OD" if num % 2 == 1 else "OS"
            except Exception:
                return None
        mapped = s.map(norm)
        uniq = {u for u in mapped.dropna().unique().tolist()}
        if not uniq.issubset({"OS","OD"}):
            raise ValueError(f"eyeID must be binary; found values {sorted(uniq)}")
        df['eyeID'] = mapped.fillna("OS")
        if src != 'eyeID':
            # keep original too if you want, but we standardize on 'eyeID'
            pass

    def _refresh_master_df(self, exclude_cols: Optional[Iterable[str]] = None) -> None:
        self.df = pd.concat(self.frames, axis=0, ignore_index=True)
        # drop columns explicitly excluded
        if exclude_cols:
            self.df = self.df.drop(columns=[c for c in exclude_cols if c in self.df.columns])

    def _infer_or_validate_feature_types(self, exclude_cols: Optional[Iterable[str]] = None) -> None:
        excluded = set(exclude_cols or []) | {self.label_col, 'Patient ID'}
        # we keep canonical 'eyeID' as categorical if present
        feature_candidates = [c for c in self.df.columns if c not in excluded]
        # If user pre-specified cat_cols in __init__, respect them and infer the rest
        cats = set(self.cat_cols) if self.cat_cols else set()
        scalars = set()
        for c in feature_candidates:
            if c == 'eyeID':
                cats.add('eyeID'); continue
            if c in cats: continue
            s = self.df[c]
            # try numeric coercion
            as_num = pd.to_numeric(s, errors='coerce')
            num_missing = as_num.isna().mean()
            num_unique = s.dropna().nunique()
            if as_num.notna().any() and num_missing < 1.0 and num_unique > self.max_unique_for_cat:
                scalars.add(c)
            else:
                # categorical if few uniques OR non-numeric
                if num_unique <= self.max_unique_for_cat or as_num.isna().mean() > 0.0:
                    cats.add(c)
                else:
                    scalars.add(c)
        self.cat_cols = sorted(cats)
        self.scalar_cols = sorted(scalars)

    def _compute_numeric_stats(self) -> None:
        self.scalar_stats.clear()
        for col in self.scalar_cols:
            s = pd.to_numeric(self.df[col], errors='coerce')
            vals = s.dropna().astype(float).values
            if vals.size == 0:
                lo, hi, med = 0.0, 1.0, 0.0
            else:
                lo, hi = float(np.min(vals)), float(np.max(vals))
                med = float(np.median(vals))
                if hi <= lo: hi = lo + 1.0
            self.scalar_stats[col] = {"min": lo, "max": hi, "median": med}

    def _build_cat_maps(self) -> None:
        self.cat_maps.clear()
        for col in self.cat_cols:
            cats = [v for v in self.df[col].dropna().unique().tolist()]
            try: cats = sorted(cats)
            except Exception: pass
            mapping = {"<UNK>": 0}
            for i, v in enumerate(cats, start=1): mapping[v] = i
            self.cat_maps[col] = mapping

    def _compute_feature_dim(self) -> None:
        self.feature_dim = len(self.scalar_cols) + sum(len(m) for m in self.cat_maps.values()) + len(self.scalar_cols)

    # ------------------- K-fold on unique patients -------------------
    def _build_kfold_indices(self) -> None:
        # unique patients and a per-patient label for stratification if possible
        pats = self.df['Patient ID'].unique().tolist()
        # Derive a patient label as the mode of their rows (fallback to first valid)
        labels_by_pat = {}
        for pid, grp in self.df.groupby('Patient ID'):
            lab = grp[self.label_col].dropna()
            if len(lab) == 0:
                labels_by_pat[pid] = 0
            else:
                labels_by_pat[pid] = lab.mode().iloc[0]
        y_pat = np.array([labels_by_pat[p] for p in pats])

        # Try to use StratifiedGroupKFold if available, else fall back to StratifiedKFold on patient labels
        try:
            from sklearn.model_selection import StratifiedGroupKFold
            sgkf = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_seed)
            split_iter = sgkf.split(X=pats, y=y_pat, groups=pats)
        except Exception:
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_seed)
            split_iter = skf.split(X=np.zeros(len(pats)), y=y_pat)

        self.folds.clear()
        for i, (train_idx, test_idx) in enumerate(split_iter):
            train_ids = [pats[j] for j in train_idx]
            test_ids  = [pats[j] for j in test_idx]
            self.folds[i] = {"train_ids": train_ids, "test_ids": test_ids}
