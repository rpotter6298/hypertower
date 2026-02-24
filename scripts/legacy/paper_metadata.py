
import pandas as pd
from classes import HyperTower, ClinicalData, list_names, build_papila_clinical
from pathlib import Path
import shutil, json, textwrap
from datetime import datetime
import numpy as np
from typing import Dict, List, Tuple
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import label_binarize

def _proba_from_model(model, X):
    if hasattr(model[-1], "predict_proba"):
        return model.predict_proba(X)
    dec = model.decision_function(X)
    if dec.ndim == 1:  # binary margins -> make 2-col
        dec = np.stack([-dec, dec], axis=1)
    e = np.exp(dec - dec.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def _cv_auc_multiclass_per_class(X, y, groups, model, n_splits=5) -> np.ndarray:
    """
    Returns a length-3 array of mean OvR AUCs for Class0/1/2 across GroupKFold.
    Uses nan-safe means if a class is absent in a fold's test split.
    """
    gkf = GroupKFold(n_splits=n_splits)
    per_class_lists = [[], [], []]
    for tr, te in gkf.split(X, y, groups):
        model.fit(X[tr], y[tr])
        proba = _proba_from_model(model, X[te])
        y_te = y[te]
        y_bin = label_binarize(y_te, classes=[0, 1, 2])  # (n,3)
        for k in range(3):
            yk = y_bin[:, k]
            if yk.min() != yk.max():  # both classes present
                per_class_lists[k].append(roc_auc_score(yk, proba[:, k]))
            else:
                per_class_lists[k].append(np.nan)
    return np.array([np.nanmean(per_class_lists[k]) for k in range(3)], dtype=float)

def _cv_auc_binary(X, y, groups, model, n_splits=5) -> float:
    mask = np.isin(y, [0, 1])
    Xb, yb, gb = X[mask], y[mask], groups[mask]
    gkf = GroupKFold(n_splits=n_splits)
    aucs = []
    for tr, te in gkf.split(Xb, yb, gb):
        model.fit(Xb[tr], yb[tr])
        if hasattr(model[-1], "predict_proba"):
            p = model.predict_proba(Xb[te])[:, 1]
        else:
            p = model.decision_function(Xb[te])
            # logistic squash for safety
            if np.ptp(p) > 0:
                p = 1.0 / (1.0 + np.exp(-p))
            else:
                p = np.full_like(p, 0.5, dtype=float)
        # only compute if both classes present
        if len(np.unique(yb[te])) == 2:
            aucs.append(roc_auc_score(yb[te], p))
        else:
            aucs.append(np.nan)
    return float(np.nanmean(aucs))



# -----------------------------------
# 1) Build Clinical Data (paper-faithful)
# -----------------------------------
IMAGE_DIR = "Papila/FundusImages"
CLINICAL_DIR = "Papila/ClinicalData"
LABEL_COL = "Diagnosis"
CAT_COLS = ["Gender", "Phakic/Pseudophakic"]

paper_auc = {
  "TEST3_multiclass": {  # Class0=Healthy, Class1=Glaucoma, Class2=Suspect
    "LogReg": {"Class0": 0.67, "Class1": 0.66, "Class2": 0.67},  # from Fig. 7 (rounded)
    "kNN":   {"Class0": 0.72, "Class1": 0.70, "Class2": 0.76},  # your read of Fig. 7
    "RF":     {"Class0": 0.66, "Class1": 0.66, "Class2": 0.67},  # from Fig. 7 (rounded)
    "SVM":    {"Class0": 0.66, "Class1": 0.65, "Class2": 0.66},  # from Fig. 7 (rounded)
  },
  "TEST4_binary": {       # Healthy vs Glaucoma (Suspects removed)
    "LogReg": 0.71,       # from text/Fig. 7 range midpoint
    "kNN":   0.75,       # your read of Fig. 7
    "RF":     0.70,       # from Fig. 7 (rounded)
    "SVM":    0.69,       # from Fig. 7 (rounded)
  }
}


clinical = build_papila_clinical(
    image_dir=IMAGE_DIR,
    clinical_dir=CLINICAL_DIR,
    label_col=LABEL_COL,
    cat_cols=CAT_COLS,
)

# -----------------------------------
# 2) Feature matrix (no MD; IOP_corr already present)
# -----------------------------------
def build_feature_matrix(clinical) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Returns:
      X: features (N x D)
      y: labels (Diagnosis: 0 healthy, 1 glaucoma, 2 suspect)
      groups: patient IDs for GroupKFold
      feat_names: list of feature names in X order
    """
    df = clinical.df.copy()

    # Scalars used in paper-style baselines (no VF_MD)
    scalars = ["Age", "dioptre_1", "dioptre_2", "astigmatism",
               "Pachymetry", "Axial_Length", "IOP_corr"]

    # Categorical one-hot
    cats = ["Gender", "Phakic/Pseudophakic"]
    df_cats = pd.get_dummies(df[cats].astype("category"), drop_first=False, prefix=cats)

    # Combine
    X = pd.concat([df[scalars], df_cats], axis=1)

    # Median impute numerics (simple, consistent)
    for c in scalars:
        med = pd.to_numeric(X[c], errors="coerce").median()
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(med)

    y = df[LABEL_COL].astype(int).values
    groups = df["Patient ID"].astype(int).values
    feat_names = list(X.columns)
    return X.values.astype(np.float32), y, groups, feat_names

# ----------------------------
# 3) Model zoo (the four methods used in the paper)
# ----------------------------
def make_models(best_params: dict | None = None, random_state: int = 42) -> dict:
    """
    Build paper-like baseline models. If best_params is provided (a dict mapping
    model-name -> param dict with pipeline-style keys like 'clf__C'), those
    params are applied to the corresponding pipelines.
    """
    models = {
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=100,
                solver="lbfgs",
                multi_class="auto"
            ))
        ]),
        "kNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=5,
                weights="uniform",
                metric="minkowski",
                p=2
            ))
        ]),
        "RF": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=100,
                criterion="gini",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                bootstrap=True,
                # random_state left as default; set via best_params if desired
            ))
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                C=1.0,
                kernel="rbf",
                gamma="scale",
                probability=False
            ))
        ]),
    }

    # Apply overrides if provided
    if best_params:
        for name, params in best_params.items():
            if name in models and params:
                models[name].set_params(**params)

    return models


# -----------------------------------
# 4) CV AUCs (mean over 5 folds; GroupKFold by patient)
# -----------------------------------
def _cv_auc_multiclass(X, y, groups, model, n_splits=5) -> float:
    gkf = GroupKFold(n_splits=n_splits)
    aucs = []
    for tr, te in gkf.split(X, y, groups):
        model.fit(X[tr], y[tr])
        if hasattr(model[-1], "predict_proba"):
            proba = model.predict_proba(X[te])
        else:
            dec = model.decision_function(X[te])
            if dec.ndim == 1:
                dec = np.stack([-dec, dec], axis=1)
            e = np.exp(dec - dec.max(axis=1, keepdims=True))
            proba = e / e.sum(axis=1, keepdims=True)
        aucs.append(roc_auc_score(y[te], proba, multi_class="ovr", average="macro"))
    return float(np.mean(aucs))


def _cv_auc_binary(X, y, groups, model, n_splits=5) -> float:
    # Keep classes 0 (healthy) and 1 (glaucoma); drop suspects (2)
    mask = np.isin(y, [0, 1])
    Xb, yb, gb = X[mask], y[mask], groups[mask]

    gkf = GroupKFold(n_splits=n_splits)
    aucs = []
    for tr, te in gkf.split(Xb, yb, gb):
        model.fit(Xb[tr], yb[tr])
        if hasattr(model[-1], "predict_proba"):
            p = model.predict_proba(Xb[te])[:, 1]
        else:
            p = model.decision_function(Xb[te])
            # simple logistic squashing if needed
            if np.ptp(p) > 0:
                p = 1.0 / (1.0 + np.exp(-p))
            else:
                p = np.full_like(p, 0.5, dtype=float)
        aucs.append(roc_auc_score(yb[te], p))
    return float(np.mean(aucs))

# -----------------------------------
# 5) Run both tests (multiclass + binary) and print table
# -----------------------------------
def run_papila_clinical_baselines(clinical, n_splits: int = 5,
                                  random_state: int = 42,
                                  best_params: dict | None = None) -> pd.DataFrame:
    X, y, groups, feat_names = build_feature_matrix(clinical)
    models = make_models(best_params=best_params, random_state=random_state)

    rows = []
    for name, model in models.items():
        c0, c1, c2 = _cv_auc_multiclass_per_class(X, y, groups, model, n_splits=n_splits)
        auc_bin = _cv_auc_binary(X, y, groups, model, n_splits=n_splits)
        rows.append({"model": name, "Class0": c0, "Class1": c1, "Class2": c2, "Binary": auc_bin})

    df = pd.DataFrame(rows).set_index("model").sort_index()
    return df


results = run_papila_clinical_baselines(clinical, n_splits=5)
# print(results.to_string(float_format=lambda x: f"{x:.3f}"))















##############################
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
from sklearn.preprocessing import label_binarize
from sklearn.utils import check_random_state

# ==============================
# Helper: per-class & binary AUC with GroupKFold
# ==============================
def _proba_from_model(model, X):
    if hasattr(model[-1], "predict_proba"):
        return model.predict_proba(X)
    # decision_function fallback
    dec = model.decision_function(X)
    if dec.ndim == 1:  # binary margin -> 2-col probs
        dec = np.stack([-dec, dec], axis=1)
    e = np.exp(dec - dec.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def _cv_auc_perclass_and_binary(X, y, groups, model, n_splits=5):
    """
    Returns:
      per_class_auc: length-3 array (Class0, Class1, Class2) averaged over folds
      binary_auc: scalar (0 vs 1) averaged over folds
    """
    gkf = GroupKFold(n_splits=n_splits)

    # Hold fold-wise per-class AUCs (list of arrays of length 3)
    perclass_fold_scores = []
    binary_fold_scores = []

    for tr, te in gkf.split(X, y, groups):
        y_te = y[te]
        # Multiclass per-class (OvR)
        model.fit(X[tr], y[tr])
        proba = _proba_from_model(model, X[te])

        # One-vs-rest per-class AUCs (skip a class if absent in test fold)
        y_bin = label_binarize(y_te, classes=[0, 1, 2])  # shape (n, 3)
        perclass_scores = []
        for k in range(3):
            yk = y_bin[:, k]
            # Only compute if both 0 and 1 are present
            if yk.min() != yk.max():
                perclass_scores.append(roc_auc_score(yk, proba[:, k]))
            else:
                perclass_scores.append(np.nan)
        perclass_fold_scores.append(perclass_scores)

        # Binary AUC (0 vs 1; drop class 2)
        mask = np.isin(y_te, [0, 1])
        if mask.sum() > 0 and len(np.unique(y_te[mask])) == 2:
            # we need probabilities/margins for class 1 among (0,1)
            # Map proba[:, 1] if the model was trained 3-way; we restrict te samples to 0/1
            binary_p = proba[mask, 1]
            binary_y = y_te[mask]
            binary_fold_scores.append(roc_auc_score(binary_y, binary_p))
        else:
            binary_fold_scores.append(np.nan)

    # Average over folds (ignore NaNs if a class was missing in a fold)
    perclass_arr = np.array(perclass_fold_scores, dtype=float)  # (n_folds, 3)
    per_class_auc = np.nanmean(perclass_arr, axis=0)
    binary_auc = float(np.nanmean(np.array(binary_fold_scores, dtype=float)))
    return per_class_auc, binary_auc

# ==============================
# Distance-to-paper objective
# ==============================
def _distance_to_paper(model_name: str,
                       per_class_auc: np.ndarray,
                       binary_auc: float,
                       paper_auc: Dict,
                       w_mc: float = 1.0,
                       w_bin: float = 1.0) -> float:
    mc_targets = paper_auc["TEST3_multiclass"][model_name]
    tvec = np.array([mc_targets["Class0"], mc_targets["Class1"], mc_targets["Class2"]], dtype=float)
    mc_diff = np.nanmean(np.abs(per_class_auc - tvec))  # mean absolute difference over 3 classes

    bin_target = paper_auc["TEST4_binary"][model_name]
    bin_diff = abs(binary_auc - bin_target)

    return float(w_mc * mc_diff + w_bin * bin_diff)

# ==============================
# Parameter grids (paper-ish, not crazy-large)
# ==============================
def get_param_grids() -> Dict[str, List[dict]]:
    return {
        "LogReg": [
            {
                "clf__C": [0.01, 0.1, 1.0, 3.0, 10.0],
                "clf__class_weight": [None, "balanced"],
                "clf__max_iter": [200, 500],
                # lbfgs + l2 is implied
            }
        ],
        "kNN": [
            {
                "clf__n_neighbors": [3, 5, 7, 9, 11],
                "clf__weights": ["uniform", "distance"],
                "clf__p": [1, 2],  # Manhattan vs Euclidean
            }
        ],
        "RF": [
            {
                "clf__n_estimators": [200, 500, 1000],
                "clf__max_depth": [None, 5, 10, 20],
                "clf__max_features": ["sqrt", "log2", 0.5],
                "clf__min_samples_leaf": [1, 2, 5],
                "clf__class_weight": [None, "balanced"],
                # If you want determinism add: "clf__random_state": [42]
            }
        ],
        "SVM": [
            {
                "clf__C": [0.1, 1.0, 3.0, 10.0],
                "clf__gamma": ["scale", "auto", 0.1, 0.01, 0.001],
                "clf__kernel": ["rbf"],  # fixed to rbf as in paper-like default
            }
        ],
    }

# ==============================
# Grid search loop minimizing distance-to-paper
# ==============================
def search_params_to_match_paper(
    clinical,
    models: Dict[str, Pipeline],
    paper_auc: Dict,
    n_splits: int = 5,
    w_mc: float = 1.0,
    w_bin: float = 1.0,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    X, y, groups, feat_names = build_feature_matrix(clinical)
    grids = get_param_grids()

    summary_rows = []
    best_params_by_model = {}

    for name, base_model in models.items():
        if name not in grids:
            if verbose:
                print(f"[warn] No grid for {name}, skipping.")
            continue

        best_loss = np.inf
        best_params = None
        best_mc = None
        best_bin = None

        for param_set in ParameterGrid(grids[name]):
            model = clone(base_model).set_params(**param_set)
            per_class_auc, binary_auc = _cv_auc_perclass_and_binary(
                X, y, groups, model, n_splits=n_splits
            )
            loss = _distance_to_paper(
                name, per_class_auc, binary_auc, paper_auc, w_mc=w_mc, w_bin=w_bin
            )

            if verbose:
                mc_str = " / ".join(f"{a:.3f}" if np.isfinite(a) else "nan" for a in per_class_auc)
                print(f"[{name}] params={param_set} | mc per-class={mc_str} | bin={binary_auc:.3f} | loss={loss:.4f}")

            if loss < best_loss:
                best_loss = loss
                best_params = param_set
                best_mc = per_class_auc
                best_bin = binary_auc

        # store
        best_params_by_model[name] = best_params
        summary_rows.append({
            "model": name,
            "best_loss": best_loss,
            "best_params": json.dumps(best_params),
            "mc_Class0": float(best_mc[0]),
            "mc_Class1": float(best_mc[1]),
            "mc_Class2": float(best_mc[2]),
            "binary_auc": float(best_bin),
            "paper_mc_Class0": paper_auc["TEST3_multiclass"][name]["Class0"],
            "paper_mc_Class1": paper_auc["TEST3_multiclass"][name]["Class1"],
            "paper_mc_Class2": paper_auc["TEST3_multiclass"][name]["Class2"],
            "paper_binary": paper_auc["TEST4_binary"][name],
        })

    df = pd.DataFrame(summary_rows).set_index("model").sort_values("best_loss")
    return df, best_params_by_model

# ==============================
# Run the search
# ==============================
models = make_models(random_state=42)
df_match, best_params = search_params_to_match_paper(
    clinical=clinical,
    models=models,
    paper_auc=paper_auc,
    n_splits=5,
    w_mc=1.0,  # weight multiclass distance
    w_bin=1.0, # weight binary distance
    verbose=True
)

# print("\n=== Best params found (by minimal distance-to-paper) ===")
# print(df_match[["best_loss","best_params","mc_Class0","mc_Class1","mc_Class2","binary_auc",
#                 "paper_mc_Class0","paper_mc_Class1","paper_mc_Class2","paper_binary"]])

# print("\nBest param dicts:")
for k, v in best_params.items():
    print(k, "->", v)

results2 = run_papila_clinical_baselines(clinical, n_splits=5, random_state=42, best_params=best_params)
print(f"Default Settings: {results.round(2)}")
print(f"Best Params Settings: {results2.round(2)}")
print(f" Paper Results: {pd.DataFrame({
    model: {**vals, "Binary": paper_auc["TEST4_binary"][model]}
    for model, vals in paper_auc["TEST3_multiclass"].items()
}).T[["Class0","Class1","Class2","Binary"]]}")