from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from classes.v2.data_bundle import DataBundle

# ---- Pachymetry → IOP correction (per PAPILA Table 3) ----
_PACHY_TABLE: Dict[int, int] = {
    475: +5,
    485: +4,
    495: +4,
    505: +3,
    515: +2,
    525: +1,
    535: +1,
    545: 0,
    555: -1,
    565: -1,
    575: -2,
    585: -3,
    595: -4,
    605: -4,
    615: -5,
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
    """Add IOP_raw/IOP_corr and drop source IOP columns + VF_MD if present (in-place safe)."""
    df["IOP_raw"] = df.apply(_pick_iop, axis=1)
    pachy = df.get("Pachymetry", pd.Series(np.nan, index=df.index))
    df["IOP_corr"] = [
        _correct_iop(r, p) for r, p in zip(df["IOP_raw"].values, pachy.values)
    ]
    drop_cols = [c for c in ("Pneumatic", "Perkins", "VF_MD") if c in df.columns]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
    return df


def _canonicalize_eye_column(df: pd.DataFrame) -> None:
    if "eyeID" in df.columns:
        src = "eyeID"
    else:
        src = None
        for c in df.columns:
            if "eye" in c.lower():
                src = c
                break
        if src is None:
            df["eyeID"] = "OS"
            return

    s = df[src]

    def norm(v):
        if pd.isna(v):
            return None
        x = str(v).strip().upper()
        if x in {"OS", "L", "LEFT", "0"}:
            return "OS"
        if x in {"OD", "R", "RIGHT", "1"}:
            return "OD"
        try:
            num = int(float(x))
            return "OD" if num % 2 == 1 else "OS"
        except Exception:
            return None

    mapped = s.map(norm)
    uniq = {u for u in mapped.dropna().unique().tolist()}
    if not uniq.issubset({"OS", "OD"}):
        raise ValueError(f"eyeID must be binary; found values {sorted(uniq)}")
    df["eyeID"] = mapped.fillna("OS")


def build_papila_data(
    *,
    image_dir: str,
    clinical_dir: str,
    label_col: str,
    cat_cols: List[str],
    n_splits: int = 5,
    random_seed: int = 42,
) -> DataBundle:
    """
    Build a DataBundle for PAPILA with dataset-specific preprocessing:
      - load OD/OS Excel sheets
      - normalize Patient ID
      - canonicalize eyeID
      - compute IOP_raw / IOP_corr, drop VF_MD
      - build feature typing & folds
    """
    bundle = DataBundle(
        image_dir=image_dir,
        clinical_dir=clinical_dir,
        label_col=label_col,
        patient_col="Patient ID",
        cat_cols=cat_cols,
        n_splits=n_splits,
        random_seed=random_seed,
        filename_template="RET{pid:03d}{eye}.jpg",
    )

    od = pd.read_excel(f"{clinical_dir}/patient_data_od.xlsx", header=1)
    od["eyeID"] = "OD"
    os = pd.read_excel(f"{clinical_dir}/patient_data_os.xlsx", header=1)
    os["eyeID"] = "OS"

    for frame in (od, os):
        if "Patient ID" not in frame.columns and "ID" in frame.columns:
            frame.rename(columns={"ID": "Patient ID"}, inplace=True)
        frame["Patient ID"] = frame["Patient ID"].astype(str).str.extract(r"(\d+)")[0].astype(int)
        _canonicalize_eye_column(frame)

    bundle.add_df(od, id_column="ID")
    bundle.add_df(os, id_column="ID")

    for i in range(len(bundle.frames)):
        bundle.frames[i] = _apply_iop_and_drop_md(bundle.frames[i])

    bundle._refresh_master_df()
    bundle._infer_or_validate_feature_types()
    bundle._compute_numeric_stats()
    bundle._build_cat_maps()
    bundle._compute_feature_dim()
    bundle._build_kfold_indices()

    return bundle
