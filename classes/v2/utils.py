"""General-purpose utilities for the V2 hypertower pipeline."""
from __future__ import annotations

import random as pyrandom
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def seed_everything(seed: int) -> None:
    pyrandom.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str | None) -> torch.device:
    if device_arg and device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _drop_mixed_label_patients(df: pd.DataFrame, *, patient_col: str, label_col: str):
    """Remove patients whose rows carry conflicting labels. Returns (clean_df, mixed_pids)."""
    per_patient = (
        df.groupby(patient_col)[label_col]
        .agg(lambda s: set(pd.to_numeric(s, errors="coerce").dropna().astype(int).tolist()))
    )
    mixed = [pid for pid, labels in per_patient.items() if len(labels) > 1]
    if not mixed:
        return df, []
    return df[~df[patient_col].isin(mixed)].reset_index(drop=True), mixed


def _relabel_mixed_patients_to_max(df: pd.DataFrame, *, patient_col: str, label_col: str):
    """Set all rows for each patient to that patient's max observed label."""
    out = df.copy()
    labels = pd.to_numeric(out[label_col], errors="coerce")
    patient_max = labels.groupby(out[patient_col]).transform("max")
    changed_rows = int((labels != patient_max).fillna(False).sum())
    out[label_col] = patient_max.astype(int)
    per_patient_unique = out.groupby(patient_col)[label_col].nunique(dropna=True)
    still_mixed = per_patient_unique[per_patient_unique > 1].index.tolist()
    return out.reset_index(drop=True), changed_rows, still_mixed
