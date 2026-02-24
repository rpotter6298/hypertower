# papila_builders.py
from typing import List, Dict
import numpy as np
import pandas as pd

from classes import ClinicalData  # adjust import path if needed

# ---- Pachymetry → IOP correction (per PAPILA Table 3) ----
_PACHY_TABLE: Dict[int, int] = {
    475:+5, 485:+4, 495:+4, 505:+3, 515:+2, 525:+1, 535:+1,
    545: 0, 555:-1, 565:-1, 575:-2, 585:-3, 595:-4, 605:-4, 615:-5,
}
_PACHY_KEYS = np.array(sorted(_PACHY_TABLE.keys()))

def _nearest_pachy_key(x: float) -> int:
    idx = int(np.argmin(np.abs(_PACHY_KEYS - float(x))))
    return int(_PACHY_KEYS[idx])

def _pick_iop(row: pd.Series) -> float:
    """Prefer Pneumatic, else Perkins; may return NaN."""
    raw = row["Pneumatic"] if not pd.isna(row.get("Pneumatic", np.nan)) else row.get("Perkins", np.nan)
    return float(raw) if not pd.isna(raw) else np.nan

def _correct_iop(raw_iop: float, pachy: float) -> float:
    """Return corrected IOP using nearest pachymetry bin; if pachy missing, return raw."""
    if pd.isna(raw_iop):
        return np.nan
    if pd.isna(pachy):
        return float(raw_iop)
    key = _nearest_pachy_key(float(pachy))
    return float(raw_iop) + float(_PACHY_TABLE[key])

def _apply_iop_and_drop_md(df: pd.DataFrame) -> pd.DataFrame:
    """Add IOP_raw/IOP_corr and drop VF_MD if present (in-place safe)."""
    # IOP_raw
    df["IOP_raw"] = df.apply(_pick_iop, axis=1)

    # IOP_corr
    pachy = df.get("Pachymetry", pd.Series(np.nan, index=df.index))
    df["IOP_corr"] = [
        _correct_iop(r, p) for r, p in zip(df["IOP_raw"].values, pachy.values)
    ]

    # Drop VF_MD if present
    if "VF_MD" in df.columns:
        df.drop(columns=["VF_MD"], inplace=True)
    return df


def build_papila_clinical(
    image_dir: str,
    clinical_dir: str,
    label_col: str,
    cat_cols: List[str],
    n_splits: int = 5,
    random_seed: int = 42,
) -> ClinicalData:
    """
    Build ClinicalData exactly like the user's original build_clinical:
      - add_df(OD), set eyeID='OD'
      - add_df(OS), set eyeID='OS'
      - normalize 'Patient ID' on frames
    THEN:
      - compute IOP_raw / IOP_corr on each frame
      - drop VF_MD
      - refresh master df + kfold indices
    """
    clinical = ClinicalData(
        image_dir=image_dir,
        clinical_dir=clinical_dir,
        label_col=label_col,
        cat_cols=cat_cols,
        n_splits=n_splits,
        random_seed=random_seed,
    )

    # --- Load exactly like original build_clinical ---
    clinical.add_df(pd.read_excel(f"{clinical_dir}/patient_data_od.xlsx", header=1), id_column="ID")
    clinical.frames[0]["eyeID"] = "OD"

    clinical.add_df(pd.read_excel(f"{clinical_dir}/patient_data_os.xlsx", header=1), id_column="ID")
    clinical.frames[1]["eyeID"] = "OS"

    # Normalize 'Patient ID' on the per-eye frames (string → int)
    for frame in clinical.frames:
        frame["Patient ID"] = frame["Patient ID"].astype(str).str.extract(r"(\d+)")[0].astype(int)

    # Build initial master as in original
    clinical._refresh_master_df()

    # --- Post-processing ON THE FRAMES (so everything stays consistent) ---
    for i in range(len(clinical.frames)):
        clinical.frames[i] = _apply_iop_and_drop_md(clinical.frames[i])

    # Refresh master again so IOP_raw/IOP_corr & MD removal propagate
    clinical._refresh_master_df()
    clinical._build_kfold_indices()

    return clinical
