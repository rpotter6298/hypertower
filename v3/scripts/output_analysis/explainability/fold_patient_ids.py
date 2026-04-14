"""
Derives test-set patient IDs for any (rep, fold) without re-running training.

The split is fully deterministic: fold_seed = rep_seed_start + rep * rep_seed_step.
build_samples() groups by patient_id with sort=True (pandas default), and the
test DataLoader uses shuffle=False — so rows in predictions_test.csv are always
in ascending Patient ID order within each test fold.

Usage:
    from v3.scripts.output_analysis.explainability.fold_patient_ids import get_test_patient_ids
    pids = get_test_patient_ids(rep=0, fold=2)   # list of int patient IDs, sorted

    # Attach to a pooled predictions DataFrame:
    df = attach_patient_ids(df, clinical_dir=...)
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT    = Path(__file__).resolve().parents[4]
CLINICAL_DIR = REPO_ROOT / "Papila" / "ClinicalData"

# These match the defaults in run_cv.py
_REP_SEED_START = 100
_REP_SEED_STEP  = 100
_N_SPLITS       = 5
_EVAL_MODE      = "binary"
_LABEL_COL      = "Diagnosis"
_PATIENT_COL    = "Patient ID"


@lru_cache(maxsize=4)
def _load_clinical(clinical_dir: Path) -> pd.DataFrame:
    """
    Load OD/OS Excel sheets, extract Patient ID + Diagnosis, binary-filter.
    Returns a DataFrame with one row per eye (OD+OS stacked), columns:
    [Patient ID, Diagnosis, eyeID, VF_MD].
    """
    od = pd.read_excel(clinical_dir / "patient_data_od.xlsx", header=1)
    os_ = pd.read_excel(clinical_dir / "patient_data_os.xlsx", header=1)
    od["eyeID"] = "OD"
    os_["eyeID"] = "OS"
    df = pd.concat([od, os_], ignore_index=True)
    # Raw column is "ID" (e.g. "#002"); canonicalize to "Patient ID"
    if "Patient ID" not in df.columns and "ID" in df.columns:
        df.rename(columns={"ID": "Patient ID"}, inplace=True)
    df["Patient ID"] = df["Patient ID"].astype(str).str.extract(r"(\d+)")[0].astype(int)
    df["Diagnosis"]  = pd.to_numeric(df["Diagnosis"], errors="coerce")
    df["VF_MD"]      = pd.to_numeric(df["VF_MD"],     errors="coerce")
    # PAPILA encoding: 0=Normal, 1=Glaucoma, 2=Suspect
    # Binary mode keeps 0 and 1, excludes Suspect (2)
    df = df[df["Diagnosis"].isin([0, 1])].copy()
    return df.reset_index(drop=True)


def _build_splits(clinical_dir: Path, fold_seed: int) -> list[Any]:
    """Return list of PatientSplit for a given fold seed."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from v3.classes.split_manager import PatientFirstSplitManager, build_patient_split_plans

    df = _load_clinical(clinical_dir)

    # Patient-level label table (mode label per patient)
    patient_table = (
        df.groupby(_PATIENT_COL)[_LABEL_COL]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
    )
    plans_raw = build_patient_split_plans(
        patient_ids=patient_table[_PATIENT_COL].to_numpy(),
        patient_labels=patient_table[_LABEL_COL].to_numpy(),
        n_splits=_N_SPLITS,
        seed=fold_seed,
    )

    # Wrap into PatientSplit-like objects with .test DataFrame
    class _Split:
        def __init__(self, test_ids):
            self.test = df[df[_PATIENT_COL].isin(test_ids)].reset_index(drop=True)

    return [_Split(p.test_patient_ids) for p in plans_raw]


def get_test_patient_ids(rep: int, fold: int,
                         clinical_dir: Path = CLINICAL_DIR,
                         rep_seed_start: int = _REP_SEED_START,
                         rep_seed_step:  int = _REP_SEED_STEP) -> list[int]:
    """
    Return sorted list of Patient IDs in the test set for (rep, fold).
    Matches the row order of predictions_test.csv for that fold.
    """
    fold_seed = rep_seed_start + rep * rep_seed_step
    plans = _build_splits(clinical_dir, fold_seed)
    test_df = plans[fold].test
    # groupby sorts by default → same order as build_samples / test loader
    return sorted(test_df[_PATIENT_COL].unique().tolist())


def _row_to_patient_pos(row_idx: int, n_patients: int, batch_size: int) -> int:
    """
    Map a single-mode row index to its patient position in the sorted patient list.

    collect_probs_single_components (aggregate_patient=False) emits predictions
    in batch-interleaved order: for each batch of B patients, OD rows come first
    then OS rows.  The last batch may be smaller than batch_size.

        Batch i (B patients): rows [i*2B .. i*2B+B-1] = OD
                                    [i*2B+B .. i*2B+2B-1] = OS
        Patient position = i*B + (row_in_batch % B)
    """
    full = n_patients // batch_size
    last_b = n_patients % batch_size
    for bi in range(full):
        s = bi * 2 * batch_size
        if s <= row_idx < s + 2 * batch_size:
            return bi * batch_size + (row_idx - s) % batch_size
    if last_b > 0:
        s = full * 2 * batch_size
        return full * batch_size + (row_idx - s) % last_b
    raise IndexError(f"row_idx {row_idx} out of range for n_patients={n_patients}")


def attach_patient_ids(df: pd.DataFrame,
                       clinical_dir: Path = CLINICAL_DIR,
                       rep_seed_start: int = _REP_SEED_START,
                       rep_seed_step:  int = _REP_SEED_STEP) -> pd.DataFrame:
    """
    Add a 'patient_id' column to a pooled predictions DataFrame.
    Requires 'rep' and 'fold' columns (added by load_all_predictions).
    The 'idx' column is the row index within each fold's test set.
    For patient-level modes (ensemble): idx == patient position directly.
    """
    df = df.copy()
    pid_col = []

    for _, row in df.iterrows():
        rep_idx  = int(row["rep"].replace("rep", ""))
        fold_idx = int(row["fold"].replace("fold", ""))
        idx      = int(row["idx"])
        pids = get_test_patient_ids(rep_idx, fold_idx,
                                    clinical_dir=clinical_dir,
                                    rep_seed_start=rep_seed_start,
                                    rep_seed_step=rep_seed_step)
        pid_col.append(pids[idx] if idx < len(pids) else None)

    df["patient_id"] = pid_col
    return df


def attach_patient_ids_single(df: pd.DataFrame,
                               clinical_dir: Path = CLINICAL_DIR,
                               batch_size: int = 8,
                               rep_seed_start: int = _REP_SEED_START,
                               rep_seed_step:  int = _REP_SEED_STEP) -> pd.DataFrame:
    """
    Like attach_patient_ids but for single (eye-level) mode.
    Single mode emits predictions in batch-interleaved order (see _row_to_patient_pos).
    batch_size must match the --batch-size used during training (default 8).
    """
    df = df.copy()
    pid_col = []

    for _, row in df.iterrows():
        rep_idx  = int(row["rep"].replace("rep", ""))
        fold_idx = int(row["fold"].replace("fold", ""))
        idx      = int(row["idx"])
        pids = get_test_patient_ids(rep_idx, fold_idx,
                                    clinical_dir=clinical_dir,
                                    rep_seed_start=rep_seed_start,
                                    rep_seed_step=rep_seed_step)
        patient_pos = _row_to_patient_pos(idx, len(pids), batch_size)
        pid_col.append(pids[patient_pos])

    df["patient_id"] = pid_col
    return df
