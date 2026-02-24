#!/usr/bin/env python3
"""Univariate ROC curves for PAPILA clinical variables."""
import re
from pathlib import Path
from typing import Iterable, List, Tuple
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classes import build_papila_clinical


# ---------------------------
# Config (edit in IDE)
# ---------------------------
IMAGE_DIR = "Papila/FundusImages"
CLINICAL_DIR = "Papila/ClinicalData"
LABEL_COL = "Diagnosis"
CAT_COLS = ["Gender", "Phakic/Pseudophakic"]
EXCLUDE_COLS = {"Pneumatic", "Perkins"}
INCLUDE_CATEGORICAL = False
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0
DROP_LABELS = [2]
OUTPUT_DIR = Path("analysis_data/basic_analysis/papila_univariate_roc")
DEBUG_PRINTS = False
PLOT_PER_FEATURE = False
DI_OPTRE_COL_PREFIXES = ("dioptre",)
ADD_DIOPTRE_ABS = True
ADD_DIOPTRE_SQUARED = True


def _sanitize(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")
    return safe or "var"


def _select_binary_labels(labels: pd.Series,
                          positive_label: int,
                          negative_label: int,
                          drop_labels: Iterable[int]) -> Tuple[np.ndarray, np.ndarray]:
    labels_num = pd.to_numeric(labels, errors="coerce")
    use_num = labels_num.notna().any()
    lab = labels_num if use_num else labels.astype(str)

    drop_set = set(drop_labels or [])
    keep = lab.isin([positive_label, negative_label])
    if drop_set:
        keep &= ~lab.isin(drop_set)

    y = (lab == positive_label).astype(int)
    return y.values, keep.values


def _compute_roc(y: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    fpr, tpr, thresholds = roc_curve(y, scores, pos_label=1)
    auc_val = float(auc(fpr, tpr))
    return fpr, tpr, thresholds, auc_val


def _best_threshold(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> Tuple[float, float, float]:
    youden = tpr - fpr
    idx = int(np.nanargmax(youden))
    return float(thresholds[idx]), float(tpr[idx]), float(fpr[idx])


def _plot_roc(fpr: np.ndarray, tpr: np.ndarray, auc_val: float, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, lw=1.8, label=f"AUC={auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_overlay(curves, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    cmap = plt.get_cmap("tab20")
    for i, (name, fpr, tpr, auc_val) in enumerate(curves):
        color = cmap(i % cmap.N)
        ax.plot(fpr, tpr, lw=1.6, color=color, label=f"{name} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize="small")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _iter_numeric(df: pd.DataFrame, cols: List[str]):
    for col in cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        yield col, s


def _is_dioptre_col(col: str) -> bool:
    name = str(col).strip().lower()
    return any(name.startswith(prefix) for prefix in DI_OPTRE_COL_PREFIXES)


def _iter_numeric_with_transforms(df: pd.DataFrame, cols: List[str]):
    for col, s in _iter_numeric(df, cols):
        yield col, s
        if _is_dioptre_col(col):
            if ADD_DIOPTRE_ABS:
                yield f"{col}_abs", s.abs()
            if ADD_DIOPTRE_SQUARED:
                yield f"{col}_sq", s.pow(2)


def _include_in_overlay(feature_name: str) -> bool:
    name = str(feature_name).strip().lower()
    if _is_dioptre_col(name) and not name.endswith("_abs"):
        return False
    return True


def _iter_categorical(df: pd.DataFrame, cols: List[str]):
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col]
        vals = s.dropna().unique().tolist()
        try:
            vals = sorted(vals)
        except Exception:
            pass
        for v in vals:
            name = f"{col}=={v}"
            ind = (s == v).astype(int)
            yield name, ind


def main() -> None:
    clinical = build_papila_clinical(
        image_dir=IMAGE_DIR,
        clinical_dir=CLINICAL_DIR,
        label_col=LABEL_COL,
        cat_cols=CAT_COLS,
    )

    df = clinical.df.copy()
    labels = df[LABEL_COL]
    y_all, keep_mask = _select_binary_labels(labels, POSITIVE_LABEL, NEGATIVE_LABEL, DROP_LABELS)

    out_dir = OUTPUT_DIR
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    curves = []

    base_exclude = {LABEL_COL, "Patient ID"} | EXCLUDE_COLS | set(CAT_COLS)
    if "eyeID" not in CAT_COLS:
        base_exclude.add("eyeID")
    candidate_cols = [c for c in df.columns if c not in base_exclude]
    numeric_cols = []
    for col in candidate_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().any():
            numeric_cols.append(col)

    if DEBUG_PRINTS:
        for col in ("IOP_raw", "IOP_corr"):
            if col not in df.columns:
                print(f"[debug] {col} missing from df")
                continue
            s = pd.to_numeric(df[col], errors="coerce")
            print(f"[debug] {col}: non-null={int(s.notna().sum())}, unique={int(s.nunique(dropna=True))}")
    for col, series in _iter_numeric_with_transforms(df, numeric_cols):
        mask = keep_mask & series.notna().values
        y = y_all[mask]
        scores = series.values[mask].astype(float)
        if y.size < 2 or np.unique(y).size < 2:
            continue
        if np.nanmin(scores) == np.nanmax(scores):
            continue
        fpr, tpr, thresholds, auc_val = _compute_roc(y, scores)
        thr, best_tpr, best_fpr = _best_threshold(fpr, tpr, thresholds)
        direction = "high" if auc_val >= 0.5 else "low"
        title = f"{col} (n={y.size}, direction={direction})"
        if PLOT_PER_FEATURE:
            out_path = plot_dir / f"roc_{_sanitize(col)}.png"
            _plot_roc(fpr, tpr, auc_val, title, out_path)
        if _include_in_overlay(col):
            curves.append((col, fpr, tpr, auc_val))
        rows.append({
            "feature": col,
            "kind": "numeric",
            "n": int(y.size),
            "auc": auc_val,
            "direction": direction,
            "best_threshold": thr,
            "best_tpr": best_tpr,
            "best_fpr": best_fpr,
            "best_specificity": 1.0 - best_fpr,
        })

    if INCLUDE_CATEGORICAL:
        cat_cols_use = [c for c in clinical.cat_cols if c not in EXCLUDE_COLS]
        for name, ind in _iter_categorical(df, cat_cols_use):
            mask = keep_mask & ind.notna().values
            y = y_all[mask]
            scores = ind.values[mask].astype(float)
            if y.size < 2 or np.unique(y).size < 2:
                continue
            if np.nanmin(scores) == np.nanmax(scores):
                continue
            fpr, tpr, thresholds, auc_val = _compute_roc(y, scores)
            thr, best_tpr, best_fpr = _best_threshold(fpr, tpr, thresholds)
            direction = "high" if auc_val >= 0.5 else "low"
            title = f"{name} (n={y.size}, direction={direction})"
            if PLOT_PER_FEATURE:
                out_path = plot_dir / f"roc_{_sanitize(name)}.png"
                _plot_roc(fpr, tpr, auc_val, title, out_path)
            if _include_in_overlay(name):
                curves.append((name, fpr, tpr, auc_val))
            rows.append({
                "feature": name,
                "kind": "categorical",
                "n": int(y.size),
                "auc": auc_val,
                "direction": direction,
                "best_threshold": thr,
                "best_tpr": best_tpr,
                "best_fpr": best_fpr,
                "best_specificity": 1.0 - best_fpr,
            })

    if not rows:
        raise SystemExit("No valid features produced ROC curves. Check labels and feature columns.")

    overlay_path = plot_dir / "roc_overlay.png"
    _plot_overlay(curves, "Univariate ROC curves", overlay_path)

    out_df = pd.DataFrame(rows).sort_values(by="auc", ascending=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_dir / "summary.csv", index=False)
    print(out_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved overlay plot to: {overlay_path}")
    if PLOT_PER_FEATURE:
        print(f"Saved per-feature plots to: {plot_dir}")
    print(f"Saved summary to: {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
