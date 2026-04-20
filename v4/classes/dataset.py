"""dataset — data packaging for v4: shells, DataBundle, HTDataset.

ShellEntry / LoaderShell
    Minimal, data-free structures representing *who* to sample and in what
    order.  entity_id is opaque to the orchestrator; towers interpret it.

DataBundle
    Accumulates per-eye DataFrames and derives scalar/categorical stats used
    by ClinicalDataView.  No kfold, no vectorization — those live in the
    profile and orchestrator respectively.

HTDataset / ht_collate
    PyTorch Dataset that delegates sample retrieval to towers via get_sample.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------

@dataclass
class ShellEntry:
    """One sample slot in a LoaderShell.

    entity_id : opaque — defined by the profile, interpreted by towers.
    label     : integer class label.
    meta      : per-entry context a profile wants to pass through.
    """
    entity_id: Any
    label:     int
    meta:      dict = field(default_factory=dict)


@dataclass
class LoaderShell:
    """An ordered sequence of ShellEntry objects for one split/fold."""
    entries: list[ShellEntry]

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)


# ---------------------------------------------------------------------------
# DataBundle
# ---------------------------------------------------------------------------

class DataBundle:
    """Accumulates per-eye DataFrames and derives feature metadata.

    Responsibilities: column type inference, scalar stats (min/max/median),
    categorical index maps, and feature dim.  Everything else (splits,
    vectorization, image paths) lives in the profile that uses this bundle.
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
    ) -> None:
        self.image_dir         = Path(image_dir)
        self.label_col         = label_col
        self.patient_col       = patient_col
        self.max_unique_for_cat = max_unique_for_cat
        self.filename_template = filename_template
        self.clinical_dir      = Path(clinical_dir) if clinical_dir else None

        self.frames:       List[pd.DataFrame]           = []
        self.df:           pd.DataFrame                 = pd.DataFrame()
        self.scalar_cols:  List[str]                    = []
        self.cat_cols:     List[str]                    = list(cat_cols) if cat_cols else []
        self.scalar_stats: Dict[str, Dict[str, float]]  = {}
        self.cat_maps:     Dict[str, Dict[object, int]] = {}
        self.feature_dim:  int                          = 0

    def add_df(
        self,
        df: pd.DataFrame,
        *,
        id_column: Optional[str] = None,
        exclude_cols: Optional[Iterable[str]] = None,
    ) -> None:
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

    # ── Internal ─────────────────────────────────────────────────────────────

    def _ensure_patient_id(self, df: pd.DataFrame, id_column: Optional[str]) -> None:
        if self.patient_col in df.columns:
            return
        if id_column and id_column in df.columns:
            df.rename(columns={id_column: self.patient_col}, inplace=True)
            return
        candidates = [
            c for c in df.columns
            if c.lower().replace(" ", "") in {"patientid", "patient", "pid"}
        ]
        if len(candidates) == 1:
            df.rename(columns={candidates[0]: self.patient_col}, inplace=True)
            return
        raise ValueError(
            f"A '{self.patient_col}' column is required; "
            f"provide id_column=... if it has a different name."
        )

    def _refresh_master_df(self, exclude_cols: Optional[Iterable[str]] = None) -> None:
        self.df = pd.concat(self.frames, axis=0, ignore_index=True)
        if exclude_cols:
            self.df = self.df.drop(columns=[c for c in exclude_cols if c in self.df.columns])

    def _infer_or_validate_feature_types(self, exclude_cols: Optional[Iterable[str]] = None) -> None:
        excluded = set(exclude_cols or []) | {self.label_col, self.patient_col}
        candidates = [c for c in self.df.columns if c not in excluded]
        cats = set(self.cat_cols)
        scalars: set[str] = set()
        for c in candidates:
            if c in cats:
                continue
            s       = self.df[c]
            as_num  = pd.to_numeric(s, errors="coerce")
            n_uniq  = s.dropna().nunique()
            if as_num.notna().any() and as_num.isna().mean() < 1.0 and n_uniq > self.max_unique_for_cat:
                scalars.add(c)
            else:
                cats.add(c)
        self.cat_cols    = sorted(cats)
        self.scalar_cols = sorted(scalars)

    def _compute_numeric_stats(self) -> None:
        self.scalar_stats.clear()
        for col in self.scalar_cols:
            vals = pd.to_numeric(self.df[col], errors="coerce").dropna().astype(float).values
            if vals.size == 0:
                lo, hi, med = 0.0, 1.0, 0.0
            else:
                lo, hi = float(np.min(vals)), float(np.max(vals))
                med    = float(np.median(vals))
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
            mapping: Dict[object, int] = {"<UNK>": 0}
            for i, v in enumerate(cats, start=1):
                mapping[v] = i
            self.cat_maps[col] = mapping

    def _compute_feature_dim(self) -> None:
        self.feature_dim = (
            len(self.scalar_cols)
            + sum(len(m) for m in self.cat_maps.values())
            + len(self.scalar_cols)
        )


# ---------------------------------------------------------------------------
# HTDataset / ht_collate
# ---------------------------------------------------------------------------

class HTDataset(Dataset):
    """PyTorch Dataset backed by a LoaderShell.

    Delegates sample retrieval to each tower's get_sample(entry).
    Batch keys are tower names plus "label".
    """

    def __init__(self, shell: LoaderShell, towers: dict) -> None:
        self.entries = shell.entries
        self.towers  = towers

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry  = self.entries[idx]
        sample = {
            "label":     torch.tensor(entry.label, dtype=torch.long),
            "entity_id": entry.entity_id,
        }
        for name, tower in self.towers.items():
            sample[name] = tower.get_sample(entry)
        return sample


def to_label_tensor(labels, device: torch.device) -> torch.Tensor:
    """Normalise a batch of labels (tensor or list) to a long tensor on device."""
    if torch.is_tensor(labels):
        return labels.to(device=device, dtype=torch.long)
    return torch.as_tensor(labels, dtype=torch.long, device=device)


def ht_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate HTDataset samples.

    Tensor values are stacked; dict values (side dicts from patient-level
    shells) are stacked per inner key; everything else becomes a list.
    """
    if not batch:
        return {}
    result: dict[str, Any] = {}
    for key in batch[0]:
        vals  = [s[key] for s in batch]
        first = vals[0]
        if isinstance(first, torch.Tensor):
            result[key] = torch.stack(vals, dim=0)
        elif isinstance(first, dict):
            result[key] = {
                side: torch.stack([v[side] for v in vals], dim=0)
                for side in first
            }
        else:
            result[key] = vals
    return result
