#!/usr/bin/env python3
"""Run classical ML models and plot *combined* ROC curves (multimodel overlays).

Keeps your original workflow for folds/tests exactly the same.
Only changes: collects predictions per test and makes:
  • One ROC plot per class (OvR), overlaying all models
  • One binary ROC plot (Healthy vs Glaucoma), overlaying all models
"""
import os
import re
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc

from classes import build_papila_clinical

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SPLIT_ROOT = Path("HelpCode/kfold")
TRUST_INDEX_COL = False

# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------
def build_feature_matrix(clinical):
    df = clinical.df.copy()
    scalars = ["Age", "dioptre_1", "dioptre_2", "astigmatism", "Pachymetry", "Axial_Length", "IOP_corr"]
    cats = ["Gender", "Phakic/Pseudophakic"]
    X = pd.concat([df[scalars], pd.get_dummies(df[cats].astype("category"), drop_first=False, prefix=cats)], axis=1)
    y = df[clinical.label_col].astype(int).values
    return X, y, scalars, df  # X keeps NaNs; we impute per-fold

# ---------------------------------------------------------------------------
# Models with tuned hyper-parameters (unchanged)
# ---------------------------------------------------------------------------
def make_models() -> Dict[str, Pipeline]:
    return {
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1, 
                class_weight="balanced",
                max_iter=200, 
                solver="lbfgs", 
                multi_class="auto")),
        ]),
        "kNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=11, weights="distance")),
        ]),
        "RF": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=8,
                                           min_samples_split=4, random_state=42)),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(C=10, kernel="rbf", gamma=0.1, probability=True)),
        ]),
    }

# ---------------------------------------------------------------------------
# Split helpers copied from paper_clinical_baselines_official.py (unchanged)
# ---------------------------------------------------------------------------
_FNAME_RE = re.compile(r"RET\s*(\d+)\s*([Oo][DSs])\.jpg$", re.IGNORECASE)

def _read_sheet_any(p: Path) -> pd.DataFrame:
    if p.suffix.lower() == ".xlsx":
        return pd.read_excel(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() == ".txt":
        lines = [ln.strip() for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
        return pd.DataFrame({"filename": lines})
    raise ValueError(f"Unsupported split file type: {p.suffix}")

def _normcols(cols: List[str]) -> Dict[str, str]:
    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())
    return {norm(c): c for c in cols}

def _parse_fname_to_pid_eye(fname: str) -> Optional[Tuple[int, str]]:
    base = os.path.basename(str(fname))
    m = _FNAME_RE.search(base.replace(" ", ""))
    if not m:
        return None
    return int(m.group(1)), m.group(2).upper()

def _rows_from_sheet(sheet: pd.DataFrame, df_master: pd.DataFrame) -> List[int]:
    cols = _normcols(list(sheet.columns))
    if "filename" in cols:
        fn_col = cols["filename"]
        lookup: Dict[str, List[int]] = {}
        for i, (pid, eye) in enumerate(zip(df_master["Patient ID"].astype(int), df_master["eyeID"].astype(str))):
            lookup.setdefault(f"{pid}|{eye.upper()}", []).append(i)
        rows: List[int] = []
        for fn in sheet[fn_col].astype(str).tolist():
            pe = _parse_fname_to_pid_eye(fn)
            if pe is None:
                continue
            pid, eye = pe
            rows.extend(lookup.get(f"{pid}|{eye}", []))
        return rows
    if "patientid" in cols and "eyeid" in cols:
        pid_col, eye_col = cols["patientid"], cols["eyeid"]
        lookup = {}
        for i, (pid, eye) in enumerate(zip(df_master["Patient ID"].astype(int), df_master["eyeID"].astype(str))):
            lookup.setdefault(f"{pid}|{eye.upper()}", []).append(i)
        rows = []
        for pid, eye in zip(sheet[pid_col], sheet[eye_col]):
            rows.extend(lookup.get(f"{int(pid)}|{str(eye).upper()}", []))
        return rows
    if TRUST_INDEX_COL and "index" in cols:
        idx = sheet[cols["index"]].astype(int).tolist()
        n = len(df_master)
        return [i for i in idx if 0 <= i < n]
    raise RuntimeError("Split sheet missing usable columns")

def _pair_train_test_files(dir_train: Path, dir_test: Path) -> List[Tuple[Path, Path]]:
    def fold_key(p: Path) -> str:
        m = re.search(r"(\d+)", p.stem)
        return m.group(1) if m else p.stem.lower()
    trains = sorted([p for p in dir_train.iterdir() if p.is_file() and p.suffix.lower() in (".xlsx", ".csv", ".txt")], key=fold_key)
    tests = sorted([p for p in dir_test.iterdir() if p.is_file() and p.suffix.lower() in (".xlsx", ".csv", ".txt")], key=fold_key)
    return [(trains[i], tests[i]) for i in range(min(len(trains), len(tests)))]

def iter_official_folds_xlsx(clinical, split_root: Path, test_name: str) -> Iterable[Tuple[pd.DataFrame, pd.DataFrame]]:
    df_master = clinical.df.copy()
    test_dir = split_root / test_name
    dir_train = test_dir / "Train"
    dir_test = test_dir / "Test"
    if not dir_train.exists() or not dir_test.exists():
        raise FileNotFoundError(f"Expected: {dir_train} and {dir_test}")
    for train_file, test_file in _pair_train_test_files(dir_train, dir_test):
        sh_tr, sh_te = _read_sheet_any(train_file), _read_sheet_any(test_file)
        tr_rows, te_rows = _rows_from_sheet(sh_tr, df_master), _rows_from_sheet(sh_te, df_master)
        tr_df, te_df = df_master.iloc[tr_rows].copy(), df_master.iloc[te_rows].copy()
        yield tr_df, te_df

# ---------------------------------------------------------------------------
# Utilities (unchanged)
# ---------------------------------------------------------------------------
def _prepare_fold_X(X: pd.DataFrame, scalars: List[str], tr_idx: np.ndarray, te_idx: np.ndarray):
    Xtr, Xte = X.iloc[tr_idx].copy(), X.iloc[te_idx].copy()
    med = Xtr[scalars].median(numeric_only=True)
    Xtr[scalars] = Xtr[scalars].fillna(med)
    Xte[scalars] = Xte[scalars].fillna(med)
    return Xtr.values.astype(np.float32), Xte.values.astype(np.float32)

# ---------------------------------------------------------------------------
# NEW: combined plotting helpers (multimodel overlays)
# ---------------------------------------------------------------------------
def _plot_multiclass_overlay(y_true: np.ndarray, prob_dict: Dict[str, np.ndarray], out_dir: Path, test_tag: str):
    """One figure per class (OvR), overlaying all models."""
    n_classes = next(iter(prob_dict.values())).shape[1]
    class_names = [f"Class{k}" for k in range(n_classes)]
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    for k in range(n_classes):
        fig, ax = plt.subplots(figsize=(6, 5))
        for model_name, proba in prob_dict.items():
            fpr, tpr, _ = roc_curve(y_bin[:, k], proba[:, k])
            auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=1.8, label=f"{model_name} (AUC={auc_val:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{class_names[k]} vs Rest — {test_tag}")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()
        fig.savefig(out_dir / f"{test_tag}_{class_names[k]}.png", dpi=170)
        plt.close(fig)

def _plot_binary_overlay(y_true: np.ndarray, prob1d_dict: Dict[str, np.ndarray], out_dir: Path, test_tag: str):
    """One figure (Healthy vs Glaucoma), overlaying all models. Assumes y_true ∈ {0,1}."""
    fig, ax = plt.subplots(figsize=(6, 5))
    any_curve = False
    for model_name, scores in prob1d_dict.items():
        if scores.size == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1.8, label=f"{model_name} (AUC={auc_val:.3f})")
        any_curve = True
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Binary Healthy vs Glaucoma — {test_tag}")
    if any_curve:
        ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_dir / f"{test_tag}_binary.png", dpi=170)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Main (same folds/tests flow; only result collation & plotting changed)
# ---------------------------------------------------------------------------
def main():
    clinical = build_papila_clinical(
        image_dir="Papila/FundusImages",
        clinical_dir="Papila/ClinicalData",
        label_col="Diagnosis",
        cat_cols=["Gender", "Phakic/Pseudophakic"],
    )
    X, y, scalars, _ = build_feature_matrix(clinical)
    models = make_models()
    out_dir = Path("analysis_data/roc_baselines")
    out_dir.mkdir(parents=True, exist_ok=True)

    tests = [
        ("Test 3", False), ("Test 4", True)
    ] if (SPLIT_ROOT / "Test 3").exists() else [
        ("Test 1", False), ("Test 2", True)
    ]

    for test_name, is_binary in tests:
        # Collect per-model probabilities following your original per-model loop.
        # For multiclass: dict[model] -> (N, C)
        # For binary:     dict[model] -> (N,) (probability of class 1)
        prob_dict_multi: Dict[str, np.ndarray] = {}
        prob_dict_bin: Dict[str, np.ndarray] = {}
        y_ref_multi: Optional[np.ndarray] = None
        y_ref_bin: Optional[np.ndarray] = None

        for model_name, model in models.items():
            y_all: List[np.ndarray] = []
            p_all: List[np.ndarray] = []

            for fold_idx, (train_df, test_df) in enumerate(iter_official_folds_xlsx(clinical, SPLIT_ROOT, test_name), 1):
                # Keep your exact masking/handling
                dup_rows = set(train_df.index).intersection(set(test_df.index))
                shared_pids = set(train_df["Patient ID"]).intersection(set(test_df["Patient ID"]))
                if test_name in ("Test 1", "Test 2") and shared_pids:
                    train_df = train_df[~train_df["Patient ID"].isin(shared_pids)].copy()
                    dup_rows = set(train_df.index).intersection(set(test_df.index))
                    shared_pids = set(train_df["Patient ID"]).intersection(set(test_df["Patient ID"]))

                tr_idx, te_idx = train_df.index.values, test_df.index.values

                if is_binary:
                    # original binary handling: drop Suspects on both sets
                    mask_tr = np.isin(y[tr_idx], [0, 1])
                    mask_te = np.isin(y[te_idx], [0, 1])
                    if not mask_tr.any() or not mask_te.any():
                        # skip empty fold (keeps behavior safe without changing fold logic)
                        continue
                    Xtr, Xte = _prepare_fold_X(X, scalars, tr_idx[mask_tr], te_idx[mask_te])
                    ytr, yte = y[tr_idx][mask_tr], y[te_idx][mask_te]
                else:
                    Xtr, Xte = _prepare_fold_X(X, scalars, tr_idx, te_idx)
                    ytr, yte = y[tr_idx], y[te_idx]

                # Fit and score (unchanged approach)
                model.fit(Xtr, ytr)
                if is_binary:
                    if hasattr(model[-1], "predict_proba"):
                        prob = model.predict_proba(Xte)[:, 1]
                    else:
                        dec = model.decision_function(Xte)
                        prob = 1.0 / (1.0 + np.exp(-dec)) if np.ptp(dec) > 0 else np.full_like(dec, 0.5)
                    y_all.append(yte)
                    p_all.append(prob)
                else:
                    if hasattr(model[-1], "predict_proba"):
                        prob = model.predict_proba(Xte)
                    else:
                        dec = model.decision_function(Xte)
                        if dec.ndim == 1:
                            dec = np.stack([-dec, dec], axis=1)
                        e = np.exp(dec - dec.max(axis=1, keepdims=True))
                        prob = e / e.sum(axis=1, keepdims=True)
                    y_all.append(yte)
                    p_all.append(prob)

            if not y_all:
                # No valid folds for this model under this test (e.g., all-bad after mask); skip
                continue

            y_cat = np.concatenate(y_all)
            p_cat = np.concatenate(p_all)

            if is_binary:
                # Store 1D scores per model
                prob_dict_bin[model_name] = p_cat
                if y_ref_bin is None:
                    y_ref_bin = y_cat
                else:
                    # Align lengths defensively (should match in normal use)
                    n = min(len(y_ref_bin), len(y_cat))
                    y_ref_bin = y_ref_bin[:n]
                    prob_dict_bin[model_name] = prob_dict_bin[model_name][:n]
            else:
                # Store (N, C) per model
                prob_dict_multi[model_name] = p_cat
                if y_ref_multi is None:
                    y_ref_multi = y_cat
                else:
                    # Align lengths defensively (should match in normal use)
                    n = min(len(y_ref_multi), len(y_cat))
                    y_ref_multi = y_ref_multi[:n]
                    prob_dict_multi[model_name] = prob_dict_multi[model_name][:n, :]

        tag = test_name.replace(" ", "")

        # Produce overlays
        if prob_dict_multi and y_ref_multi is not None:
            _plot_multiclass_overlay(y_ref_multi, prob_dict_multi, out_dir, tag)
        if prob_dict_bin and y_ref_bin is not None:
            _plot_binary_overlay(y_ref_bin, prob_dict_bin, out_dir, tag)

if __name__ == "__main__":
    main()
