#!/usr/bin/env python
"""
Phase 1: Reproduce PAPILA paper baseline results.

Runs classical ML classifiers on clinical data and/or a CNN on fundus images,
using the same 5-fold stratified CV scheme as the original paper.

Classifiers available (enable with flags):
  --knn    K-Nearest Neighbours
  --rf     Random Forest
  --svm    Support Vector Machine
  --logreg Logistic Regression
  --cnn    CNN (specify backbone with --backbone)

Clinical data loader is self-contained here — tweak the ClinicalLoader class
below without touching anything in the main v3 classes. This lets you match
the paper's preprocessing (or lack thereof) independently.

Usage examples:
  # All classical + our default clinical preprocessing
  python -m v3.scripts.main.phase_1_papila_reproduce --knn --rf --svm --logreg

  # Match paper more closely (no IOP correction, no feature engineering)
  python -m v3.scripts.main.phase_1_papila_reproduce --knn --rf --svm --logreg \
      --no-iop-corr --keep-raw-iop --no-cat-cols

  # CNN only, refugelike backbone
  python -m v3.scripts.main.phase_1_papila_reproduce --cnn --backbone refugelike

  # CNN with paper backbones
  python -m v3.scripts.main.phase_1_papila_reproduce --cnn \
      --backbone resnet50 --backbone-pretrained

  # Everything
  python -m v3.scripts.main.phase_1_papila_reproduce --knn --rf --svm --logreg \
      --cnn --backbone refugelike --output-dir analysis_data/papila_reproduce
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


# ---------------------------------------------------------------------------
# Standalone clinical data loader
# ---------------------------------------------------------------------------
# This loader is intentionally independent of the v3 clinical data pipeline
# so that we can tune preprocessing to match the original PAPILA paper without
# modifying the production classes.

class ClinicalLoader:
    """
    Standalone loader for PAPILA clinical data.

    Parameters
    ----------
    clinical_dir : str
        Path to Papila/ClinicalData directory.
    label_col : str
        Column containing ground-truth labels (default: "Diagnosis").
    cat_cols : list[str] | None
        Categorical columns to one-hot encode. Pass [] to disable.
    exclude_cols : list[str] | None
        Extra columns to drop from the feature matrix.
    iop_corr : bool
        Apply Perkins→Pneumatic IOP correction (ratio method). Default True.
    keep_raw_iop : bool
        If True, keep Perkins IOP column alongside corrected IOP. Default False.
    drop_suspects : bool
        Drop Diagnosis==2 (Suspect) rows — binary task only. Default True.
    """

    # PAPILA column names
    _PATIENT_COL = "Patient ID"
    _EYE_COL     = "eyeID"

    # These are always excluded from feature matrix
    _ALWAYS_EXCLUDE = {"ID", "Patient ID", "eyeID", "Diagnosis", "VF_MD"}

    def __init__(
        self,
        clinical_dir: str = "Papila/ClinicalData",
        label_col: str = "Diagnosis",
        cat_cols: Optional[List[str]] = None,
        exclude_cols: Optional[List[str]] = None,
        iop_corr: bool = True,
        keep_raw_iop: bool = False,
        drop_suspects: bool = True,
    ) -> None:
        self.clinical_dir  = Path(clinical_dir)
        self.label_col     = label_col
        self.cat_cols      = cat_cols if cat_cols is not None else ["Gender", "Phakic/Pseudophakic"]
        self.exclude_cols  = set(exclude_cols or [])
        self.iop_corr      = iop_corr
        self.keep_raw_iop  = keep_raw_iop
        self.drop_suspects = drop_suspects
        self._df: Optional[pd.DataFrame] = None

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = self._load()
        return self._df

    def _load(self) -> pd.DataFrame:
        # Load OD and OS files (xlsx, header on row 1)
        od_path = self.clinical_dir / "patient_data_od.xlsx"
        os_path = self.clinical_dir / "patient_data_os.xlsx"
        frames = []
        for path, eye in ((od_path, "OD"), (os_path, "OS")):
            if not path.exists():
                raise FileNotFoundError(f"Clinical data file not found: {path}")
            df = pd.read_excel(path, header=1)
            df["eyeID"] = eye
            # Normalise patient ID: '#002' → 2
            id_col = "Patient ID" if "Patient ID" in df.columns else "ID"
            df["Patient ID"] = (
                df[id_col].astype(str).str.extract(r"(\d+)")[0].astype(int)
            )
            frames.append(df)
        df = pd.concat(frames, ignore_index=True)

        # IOP: average Perkins and Pneumatic when both present, else use whichever is available
        has_perk = "Perkins" in df.columns
        has_pneu = "Pneumatic" in df.columns
        if has_perk and has_pneu:
            both = df["Perkins"].notna() & df["Pneumatic"].notna()
            df["IOP_raw"] = df["Perkins"].copy()
            df.loc[both, "IOP_raw"] = (df.loc[both, "Perkins"] + df.loc[both, "Pneumatic"]) / 2
            df.loc[~both & df["Pneumatic"].notna(), "IOP_raw"] = df.loc[~both & df["Pneumatic"].notna(), "Pneumatic"]
            df = df.drop(columns=["Perkins", "Pneumatic"])
        elif has_pneu:
            df = df.rename(columns={"Pneumatic": "IOP_raw"})
        elif has_perk:
            df = df.rename(columns={"Perkins": "IOP_raw"})

        if self.drop_suspects:
            df = df[df[self.label_col] != 2].reset_index(drop=True)

        return df

    def feature_matrix(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Returns (X, y, feature_names, patient_ids) at the eye level.

        Each eye is one row. Patient IDs are returned so that CV can split
        at the patient level (preventing OD/OS leakage across folds).

        X shape: (n_eyes, n_features)
        y: binary labels (0=Normal, 1=Glaucoma)
        patient_ids: (n_eyes,) int array — group labels for GroupKFold
        """
        df = self.df.copy()
        exclude = self._ALWAYS_EXCLUDE | self.exclude_cols

        numeric_cols = [
            c for c in df.columns
            if c not in exclude and c not in self.cat_cols
            and c not in ("eyeID",)
            and pd.to_numeric(df[c], errors="coerce").notna().any()
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

        X_num = df[numeric_cols].values.astype(np.float32)
        names = list(numeric_cols)

        parts = [X_num]
        cat_present = [c for c in (self.cat_cols or []) if c in df.columns]
        if cat_present:
            dummies = pd.get_dummies(df[cat_present].astype("category"),
                                     drop_first=False)
            parts.append(dummies.values.astype(np.float32))
            names.extend(list(dummies.columns))

        X = np.concatenate(parts, axis=1)
        y = (df[self.label_col].values.astype(int) == 1).astype(int)
        patient_ids = df["Patient ID"].values.astype(int)
        return X, y, names, patient_ids


# ---------------------------------------------------------------------------
# Adapter: wraps v3 DataBundle to match ClinicalLoader.feature_matrix() API
# ---------------------------------------------------------------------------

class _BundleLoaderAdapter:
    """Thin wrapper around a v3 DataBundle for use in phase_1 classical CV."""

    def __init__(self, bundle, label_col: str, drop_suspects: bool = True):
        self._bundle = bundle
        self.label_col = label_col
        self.drop_suspects = drop_suspects
        self._df_cache: Optional[pd.DataFrame] = None

    @property
    def df(self) -> pd.DataFrame:
        if self._df_cache is None:
            df = self._bundle.df.copy()
            if self.drop_suspects:
                df = df[df[self.label_col] != 2].reset_index(drop=True)
            self._df_cache = df
        return self._df_cache

    def feature_matrix(self) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
        df = self.df.copy()
        patient_col = self._bundle.patient_col
        scalar_cols = [c for c in self._bundle.scalar_cols if c in df.columns]
        for col in scalar_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
        X_num = df[scalar_cols].values.astype(np.float32)
        names = list(scalar_cols)

        parts = [X_num]
        cat_present = [c for c in self._bundle.cat_cols if c in df.columns]
        if cat_present:
            dummies = pd.get_dummies(df[cat_present].astype("category"), drop_first=False)
            parts.append(dummies.values.astype(np.float32))
            names.extend(list(dummies.columns))

        X = np.concatenate(parts, axis=1)
        y = (df[self.label_col].values.astype(int) == 1).astype(int)
        groups = df[patient_col].values.astype(int)
        return X, y, names, groups


# ---------------------------------------------------------------------------
# Shared CV utilities
# ---------------------------------------------------------------------------

def _oof_scores(model, X, y, groups, n_splits, seed, patient_level_cv=True):
    if patient_level_cv:
        splitter = StratifiedGroupKFold(n_splits=n_splits)
        split_iter = splitter.split(X, y, groups=groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, y)
    scores = np.zeros(len(y), dtype=float)
    preds  = np.zeros(len(y), dtype=int)
    for tr_idx, te_idx in split_iter:
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr = y[tr_idx]
        if np.unique(ytr).size < 2:
            continue
        m = clone(model)
        m.fit(Xtr, ytr)
        preds[te_idx] = m.predict(Xte)
        if hasattr(m, "predict_proba"):
            scores[te_idx] = m.predict_proba(Xte)[:, 1]
        elif hasattr(m, "decision_function"):
            scores[te_idx] = m.decision_function(Xte)
        else:
            scores[te_idx] = preds[te_idx].astype(float)
    return y.astype(int), scores, preds


def _cv_curves(model, X, y, groups, n_splits, seed, patient_level_cv=True):
    if patient_level_cv:
        splitter = StratifiedGroupKFold(n_splits=n_splits)
        split_iter = splitter.split(X, y, groups=groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, y)
    curves, fold_aucs, fold_accs = [], [], []
    for tr_idx, te_idx in split_iter:
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        if np.unique(ytr).size < 2 or np.unique(yte).size < 2:
            continue
        m = clone(model)
        m.fit(Xtr, ytr)
        if hasattr(m, "predict_proba"):
            sc = m.predict_proba(Xte)[:, 1]
        elif hasattr(m, "decision_function"):
            sc = m.decision_function(Xte)
        else:
            sc = m.predict(Xte).astype(float)
        fpr, tpr, _ = roc_curve(yte, sc, pos_label=1)
        curves.append((fpr, tpr, float(roc_auc_score(yte, sc))))
        fold_aucs.append(float(roc_auc_score(yte, sc)))
        fold_accs.append(float(accuracy_score(yte, m.predict(Xte))))
    return curves, fold_aucs, fold_accs


def _plot_mean_roc(curves, title, path):
    if not curves:
        return
    mean_fpr = np.linspace(0, 1, 200)
    tprs, aucs = [], []
    for fpr, tpr, auc_val in curves:
        tpr_i = np.interp(mean_fpr, fpr, tpr); tpr_i[0] = 0.0
        tprs.append(tpr_i); aucs.append(auc_val)
    mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
    std_tpr  = np.std(tprs, axis=0)
    mean_auc = float(np.mean(aucs)); std_auc = float(np.std(aucs))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(mean_fpr, mean_tpr, lw=2, label=f"AUC={mean_auc:.3f}±{std_auc:.3f}")
    ax.fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0),
                    np.minimum(mean_tpr + std_tpr, 1), alpha=0.2, color="grey")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, linestyle="--"); fig.tight_layout()
    fig.savefig(path, dpi=170); plt.close(fig)
    return mean_auc, std_auc


def _plot_overlay(all_curves: dict, title: str, path: Path):
    """all_curves: {model_name: (mean_fpr, mean_tpr, mean_auc, std_auc)}"""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    cmap = plt.get_cmap("tab10")
    for i, (name, (fpr, tpr, mean_auc, std_auc)) in enumerate(all_curves.items()):
        ax.plot(fpr, tpr, lw=2, color=cmap(i), label=f"{name} (AUC={mean_auc:.3f}±{std_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend(loc="upper left", fontsize="small")
    ax.grid(True, alpha=0.3, linestyle="--"); fig.tight_layout()
    fig.savefig(path, dpi=170); plt.close(fig)


def _print_result(name, aucs, accs):
    mu_auc = float(np.mean(aucs)); sd_auc = float(np.std(aucs))
    mu_acc = float(np.mean(accs)); sd_acc = float(np.std(accs))
    print(f"  {name:30s}  AUC={mu_auc:.3f}±{sd_auc:.3f}  ACC={mu_acc:.3f}±{sd_acc:.3f}")


# ---------------------------------------------------------------------------
# Classical classifier runners
# ---------------------------------------------------------------------------

def run_classical(
    name: str,
    model,
    loader: ClinicalLoader,
    out_dir: Path,
    n_splits: int,
    seed: int,
    patient_level_cv: bool = True,
) -> dict:
    X, y, feat_names, groups = loader.feature_matrix()
    curves, fold_aucs, fold_accs = _cv_curves(
        model, X, y, groups, n_splits, seed, patient_level_cv=patient_level_cv
    )

    sub = out_dir / name
    sub.mkdir(parents=True, exist_ok=True)
    res = _plot_mean_roc(curves, f"{name} ROC (mean ± SD)", sub / "roc_mean.png")
    mean_auc, std_auc = (res if res else (float("nan"), float("nan")))

    pd.DataFrame([{
        "model": name, "auc_mean": mean_auc, "auc_std": std_auc,
        "acc_mean": float(np.mean(fold_accs)), "acc_std": float(np.std(fold_accs)),
        "n_folds": len(fold_aucs),
    }]).to_csv(sub / "summary.csv", index=False)

    pd.DataFrame([{
        "fold": i+1, "auc": a, "acc": c
    } for i, (a, c) in enumerate(zip(fold_aucs, fold_accs))]).to_csv(
        sub / "fold_metrics.csv", index=False
    )

    _print_result(name, fold_aucs, fold_accs)

    # Return curve for overlay
    if curves:
        mean_fpr = np.linspace(0, 1, 200)
        tprs = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr, _ in curves]
        mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
        return {"fpr": mean_fpr, "tpr": mean_tpr, "auc_mean": mean_auc, "auc_std": std_auc}
    return {}


# ---------------------------------------------------------------------------
# CNN runner
# ---------------------------------------------------------------------------

def run_cnn(
    backbone: str,
    image_dir: str,
    clinical_dir: str,
    label_col: str,
    out_dir: Path,
    n_splits: int,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    freeze_ratio: float,
    augment: bool,
    device_str: str,
    drop_suspects: bool,
    preprocessor=None,
    img_size: int = 224,
    img_loader=None,
) -> dict:
    """
    Train a CNN-only (image only, no clinical data) baseline.

    CV strategy: StratifiedGroupKFold on patients (no OD/OS leakage).
    Within each outer fold, 20% of training patients are held out as a
    validation set for early stopping; the outer test fold is only
    evaluated once using the best-val checkpoint.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from PIL import Image

    from v3.classes.backbones import BACKBONES
    from v3.classes.image_loader import CachedImageLoader
    from v3.classes.utils import choose_device

    device = choose_device(device_str)
    spec = BACKBONES.get(backbone)
    if spec is None:
        raise ValueError(f"Unknown backbone: {backbone!r}. Available: {list(BACKBONES)}")

    # Load patient/eye table from clinical data (labels only — images are the input)
    loader_cd = ClinicalLoader(clinical_dir=clinical_dir, drop_suspects=drop_suspects)
    df = loader_cd.df[["Patient ID", "eyeID", label_col]].copy()
    df = df[df[label_col].isin([0, 1])].reset_index(drop=True)
    df["binary_label"] = (df[label_col] == 1).astype(int)

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    # If a cropper preprocessor is provided it already resizes to img_size,
    # so we skip the Resize in the transform to avoid a second interpolation.
    resize_in_tf = preprocessor is None
    eval_tf = transforms.Compose([
        *([ transforms.Resize((img_size, img_size)) ] if resize_in_tf else []),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_tf = transforms.Compose([
        *([ transforms.Resize((img_size, img_size)) ] if resize_in_tf else []),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.1, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]) if augment else eval_tf

    if img_loader is None:
        img_loader = CachedImageLoader(enabled=True, workers=4)

    class EyeDataset(Dataset):
        def __init__(self, records, transform):
            self.records   = records   # list of (pid, eye, label)
            self.transform = transform

        def warm(self):
            paths = [
                str(Path(image_dir) / f"RET{int(pid):03d}{eye.upper()}.jpg")
                for pid, eye, _ in self.records
            ]
            img_loader.warm(paths, preprocessor=preprocessor)

        def __len__(self): return len(self.records)

        def __getitem__(self, idx):
            pid, eye, label = self.records[idx]
            p = Path(image_dir) / f"RET{int(pid):03d}{eye.upper()}.jpg"
            if p.exists():
                img = img_loader.load(p, preprocessor=preprocessor)
            else:
                img = Image.new("RGB", (img_size, img_size))
            return self.transform(img), int(label)

    class CNNClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            bb_spec = BACKBONES[backbone]
            raw_model = bb_spec.ctor(weights=bb_spec.weights_default)
            feat_dim, self.backbone = bb_spec.strip(raw_model)
            if freeze_ratio > 0:
                blocks = bb_spec.blocks(self.backbone)
                n_freeze = int(len(blocks) * freeze_ratio)
                for blk in blocks[:n_freeze]:
                    for p in blk.parameters():
                        p.requires_grad_(False)
            self.head = nn.Linear(feat_dim, 2)

        def forward(self, x):
            return self.head(self.backbone(x))

    def _eval_loader(model, loader):
        model.eval()
        all_probs, all_y = [], []
        with torch.no_grad():
            for imgs, lbls in loader:
                probs = torch.softmax(model(imgs.to(device)), dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs.tolist())
                all_y.extend(lbls.numpy().tolist())
        return np.array(all_y), np.array(all_probs)

    # Build eye-level records and patient-level group array
    records_all = list(df[["Patient ID", "eyeID", "binary_label"]].itertuples(index=False, name=None))
    patient_ids = df["Patient ID"].values.astype(int)
    labels_arr  = df["binary_label"].values.astype(int)

    # Patient-level label for stratification in outer splitter
    pat_label_map = df.groupby("Patient ID")["binary_label"].first().to_dict()
    patient_labels = np.array([pat_label_map[p] for p in patient_ids])

    fold_aucs, fold_accs, curves = [], [], []
    outer = StratifiedGroupKFold(n_splits=n_splits)

    for fold, (trainval_idx, te_idx) in enumerate(
            outer.split(records_all, patient_labels, groups=patient_ids)):
        print(f"  [CNN {backbone}] fold {fold+1}/{n_splits}", flush=True)

        # Split trainval patients into train/val (80/20) for early stopping
        tv_patients   = np.unique(patient_ids[trainval_idx])
        tv_pat_labels = np.array([pat_label_map[p] for p in tv_patients])
        inner = StratifiedGroupKFold(n_splits=5)
        tr_pat_set, va_pat_set = next(iter(
            (set(tv_patients[ti]), set(tv_patients[vi]))
            for ti, vi in [next(inner.split(tv_patients, tv_pat_labels, groups=tv_patients))]
        ))

        tr_recs = [records_all[i] for i in trainval_idx if patient_ids[i] in tr_pat_set]
        va_recs = [records_all[i] for i in trainval_idx if patient_ids[i] in va_pat_set]
        te_recs = [records_all[i] for i in te_idx]

        tr_ds = EyeDataset(tr_recs, train_tf)
        va_ds = EyeDataset(va_recs, eval_tf)
        te_ds = EyeDataset(te_recs, eval_tf)
        for ds in (tr_ds, va_ds, te_ds):
            ds.warm()

        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
        va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model_cnn = CNNClassifier().to(device)
        opt = optim.Adam(filter(lambda p: p.requires_grad, model_cnn.parameters()), lr=lr)

        # Class-weighted loss: w_c = N / (N_c * C), matching paper eq. (2)
        tr_labels = [r[2] for r in tr_recs]
        n_total = len(tr_labels)
        n_classes = 2
        class_counts = np.bincount(tr_labels, minlength=n_classes).astype(float)
        class_counts = np.maximum(class_counts, 1)  # avoid div-by-zero
        weights = torch.tensor(
            n_total / (class_counts * n_classes), dtype=torch.float32
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)

        for ep in range(epochs):
            model_cnn.train()
            for imgs, lbls in tr_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                opt.zero_grad()
                criterion(model_cnn(imgs), lbls).backward()
                opt.step()

            if (ep + 1) % 5 == 0 or ep == epochs - 1:
                val_y, val_probs = _eval_loader(model_cnn, va_loader)
                val_auc = float(roc_auc_score(val_y, val_probs)) if val_y.size and len(np.unique(val_y)) > 1 else float("nan")
                print(f"    ep {ep+1:3d}/{epochs}  val_auc={val_auc:.3f}", flush=True)
        te_y, te_probs = _eval_loader(model_cnn, te_loader)
        if te_y.size and len(np.unique(te_y)) > 1:
            auc_val = float(roc_auc_score(te_y, te_probs))
            acc_val = float(accuracy_score(te_y, (te_probs >= 0.5).astype(int)))
            fold_aucs.append(auc_val)
            fold_accs.append(acc_val)
            fpr, tpr, _ = roc_curve(te_y, te_probs, pos_label=1)
            curves.append((fpr, tpr, auc_val))
            print(f"    fold {fold+1} TEST → AUC={auc_val:.3f}  ACC={acc_val:.3f}", flush=True)

    name = f"CNN ({backbone})"
    sub  = out_dir / f"cnn_{backbone}"
    sub.mkdir(parents=True, exist_ok=True)
    res = _plot_mean_roc(curves, f"{name} ROC (mean ± SD)", sub / "roc_mean.png")
    mean_auc, std_auc = (res if res else (float("nan"), float("nan")))

    pd.DataFrame([{
        "backbone": backbone, "auc_mean": mean_auc, "auc_std": std_auc,
        "acc_mean": float(np.mean(fold_accs)) if fold_accs else float("nan"),
        "acc_std":  float(np.std(fold_accs))  if fold_accs else float("nan"),
        "n_folds": len(fold_aucs),
    }]).to_csv(sub / "summary.csv", index=False)

    pd.DataFrame([{
        "fold": i+1, "auc": a, "acc": c
    } for i, (a, c) in enumerate(zip(fold_aucs, fold_accs))]).to_csv(
        sub / "fold_metrics.csv", index=False
    )

    _print_result(name, fold_aucs, fold_accs)

    if curves:
        mean_fpr = np.linspace(0, 1, 200)
        tprs = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr, _ in curves]
        mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
        return {"fpr": mean_fpr, "tpr": mean_tpr, "auc_mean": mean_auc, "auc_std": std_auc}
    return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Which classifiers to run
    ap.add_argument("--knn", action="store_true", help="Run K-Nearest Neighbours")
    ap.add_argument("--rf",  action="store_true", help="Run Random Forest")
    ap.add_argument("--svm", action="store_true", help="Run SVM")
    ap.add_argument("--logreg", action="store_true", help="Run Logistic Regression")
    ap.add_argument("--cnn", action="store_true", help="Run CNN")
    ap.add_argument("--all", action="store_true", help="Run all classifiers")

    # Data paths
    ap.add_argument("--image-dir",    default="Papila/FundusImages")
    ap.add_argument("--clinical-dir", default="Papila/ClinicalData")
    ap.add_argument("--label-col",    default="Diagnosis")
    ap.add_argument("--output-dir",   default="analysis_data/papila_reproduce")
    ap.add_argument("--tag",          default=None,
                    help="Optional suffix appended to --output-dir (e.g. 'paper_matched').")

    # Clinical data loader options
    ap.add_argument("--no-iop-corr",    action="store_true",
                    help="Skip IOP correction (use raw Perkins/Pneumatic values)")
    ap.add_argument("--keep-raw-iop",   action="store_true",
                    help="Keep raw IOP column alongside corrected IOP")
    ap.add_argument("--no-cat-cols",    action="store_true",
                    help="Exclude categorical columns (Gender, Phakic/Pseudophakic)")
    ap.add_argument("--exclude-cols",   nargs="*", default=[],
                    help="Additional columns to exclude from clinical feature matrix")
    ap.add_argument("--keep-suspects",  action="store_true",
                    help="Include Suspect (label 2) rows (default: drop them)")
    ap.add_argument("--hypertower-loader", action="store_true",
                    help="Use the v3 HyperTower clinical data bundle (better IOP correction) "
                         "instead of the standalone ClinicalLoader.")

    # CV
    ap.add_argument("--n-splits",  type=int, default=5)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--paper-cv",  action="store_true",
                    help="Use eye-level StratifiedKFold (matches paper's likely methodology) "
                         "instead of patient-level GroupKFold (our cleaner default).")

    # CNN image cropping (GT or UNet, same flags as main hypertower)
    ap.add_argument("--img-crop-manifest",  default=None,
                    help="Path to crop manifest CSV (enables cropping).")
    ap.add_argument("--img-crop-gt",        action="store_true",
                    help="Use GT segmentations to crop (requires --img-crop-manifest).")
    ap.add_argument("--img-crop-weights",   default=None,
                    help="UNet weights path for disc cropping (requires --img-crop-manifest).")
    ap.add_argument("--img-crop-scale",     type=float, default=2.5)
    ap.add_argument("--img-crop-size",      type=int,   default=200,
                    help="Crop target size in pixels (default 200, matching PAPILA paper).")
    ap.add_argument("--img-crop-cache",     default="cache_data/phase1_crops")
    ap.add_argument("--persist-img-crop-cache", action="store_true")

    # CNN options
    ap.add_argument("--backbone",           default="refugelike",
                    help="CNN backbone key (refugelike, resnet50, densenet121, vgg16, "
                         "efficientnet_b0, inception_v3, mobilenet_v2, refuge_densenet, ...)")
    ap.add_argument("--backbones",          nargs="+", default=None,
                    help="Run multiple backbones sequentially, sharing the image cache. "
                         "Overrides --backbone. e.g. --backbones resnet50 densenet121 vgg16")
    ap.add_argument("--epochs",             type=int,   default=15)
    ap.add_argument("--batch-size",         type=int,   default=16)
    ap.add_argument("--lr",                 type=float, default=1e-4)
    ap.add_argument("--freeze-ratio",       type=float, default=0.0,
                    help="Fraction of backbone blocks to freeze (0=finetune all, 1=freeze all)")
    ap.add_argument("--augment",            action="store_true")
    ap.add_argument("--device",             default="auto",
                    choices=["auto", "cpu", "cuda"])

    # Classical ML hyperparameters
    ap.add_argument("--knn-k",          type=int,   default=5)
    ap.add_argument("--rf-n-estimators", type=int,  default=500)
    ap.add_argument("--svm-c",          type=float, default=1.0)
    ap.add_argument("--lr-c",           type=float, default=1.0)
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()

    if args.all:
        args.knn = args.rf = args.svm = args.logreg = args.cnn = True

    if not any([args.knn, args.rf, args.svm, args.logreg, args.cnn]):
        ap.error("Specify at least one classifier: --knn --rf --svm --logreg --cnn (or --all)")

    out_dir = Path(args.output_dir)
    if args.tag:
        out_dir = out_dir / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build clinical data loader
    if args.hypertower_loader:
        from v3.classes.papila_builders import build_papila_data
        bundle = build_papila_data(
            image_dir=args.image_dir,
            clinical_dir=args.clinical_dir,
            label_col=args.label_col,
            cat_cols=["Gender", "Phakic/Pseudophakic"],
            n_splits=args.n_splits,
            random_seed=args.seed,
            iop_corr_method="ratio",
            iop_drop_raw=True,
        )
        loader = _BundleLoaderAdapter(bundle, label_col=args.label_col,
                                      drop_suspects=not args.keep_suspects)
        print(f"Clinical data: {len(loader.df)} rows  [HyperTower loader]  "
              f"(suspects {'kept' if args.keep_suspects else 'dropped'})")
    else:
        loader = ClinicalLoader(
            clinical_dir=args.clinical_dir,
            label_col=args.label_col,
            cat_cols=[] if args.no_cat_cols else None,
            exclude_cols=list(args.exclude_cols or []),
            iop_corr=not args.no_iop_corr,
            keep_raw_iop=args.keep_raw_iop,
            drop_suspects=not args.keep_suspects,
        )
        print(f"Clinical data: {len(loader.df)} rows  "
              f"(suspects {'kept' if args.keep_suspects else 'dropped'})")

    X, y, feat_names, groups = loader.feature_matrix()
    n_patients = len(np.unique(groups))
    print(f"Feature matrix: {X.shape}  ({n_patients} patients)  class balance: {dict(zip(*np.unique(y, return_counts=True)))}")

    overlay_curves: dict = {}
    all_results: list = []

    t0 = time.time()

    # KNN
    if args.knn:
        print("\n--- KNN ---")
        model = Pipeline([
            ("scale", StandardScaler()),
            ("knn",   KNeighborsClassifier(n_neighbors=args.knn_k)),
        ])
        r = run_classical("KNN", model, loader, out_dir, args.n_splits, args.seed, patient_level_cv=not args.paper_cv)
        if r:
            overlay_curves["KNN"] = (r["fpr"], r["tpr"], r["auc_mean"], r["auc_std"])

    # Random Forest
    if args.rf:
        print("\n--- Random Forest ---")
        model = RandomForestClassifier(
            n_estimators=args.rf_n_estimators, max_features="sqrt",
            random_state=args.seed, n_jobs=-1,
        )
        r = run_classical("Random Forest", model, loader, out_dir, args.n_splits, args.seed, patient_level_cv=not args.paper_cv)
        if r:
            overlay_curves["Random Forest"] = (r["fpr"], r["tpr"], r["auc_mean"], r["auc_std"])

    # SVM
    if args.svm:
        print("\n--- SVM ---")
        model = Pipeline([
            ("scale", StandardScaler()),
            ("svm",   SVC(kernel="rbf", C=args.svm_c, gamma="scale",
                          probability=True, random_state=args.seed)),
        ])
        r = run_classical("SVM", model, loader, out_dir, args.n_splits, args.seed, patient_level_cv=not args.paper_cv)
        if r:
            overlay_curves["SVM"] = (r["fpr"], r["tpr"], r["auc_mean"], r["auc_std"])

    # Logistic Regression
    if args.logreg:
        print("\n--- Logistic Regression ---")
        model = Pipeline([
            ("scale",  StandardScaler()),
            ("logreg", LogisticRegression(C=args.lr_c, max_iter=1000,
                                          solver="lbfgs")),
        ])
        r = run_classical("Logistic Regression", model, loader, out_dir, args.n_splits, args.seed, patient_level_cv=not args.paper_cv)
        if r:
            overlay_curves["Logistic Regression"] = (r["fpr"], r["tpr"], r["auc_mean"], r["auc_std"])

    # CNN
    if args.cnn:
        from v3.classes.croppers import build_image_preprocessor_from_args
        from v3.classes.image_loader import CachedImageLoader as _CachedImageLoader
        cnn_preprocessor = build_image_preprocessor_from_args(args)
        backbones_to_run = args.backbones if args.backbones else [args.backbone]
        shared_img_loader = _CachedImageLoader(enabled=True, workers=4)
        for backbone in backbones_to_run:
            print(f"\n--- CNN ({backbone})"
                  + (" [cropped]" if cnn_preprocessor else "") + " ---")
            r = run_cnn(
                backbone=backbone,
                image_dir=args.image_dir,
                clinical_dir=args.clinical_dir,
                label_col=args.label_col,
                out_dir=out_dir,
                n_splits=args.n_splits,
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                freeze_ratio=args.freeze_ratio,
                augment=args.augment,
                device_str=args.device,
                drop_suspects=not args.keep_suspects,
                preprocessor=cnn_preprocessor,
                img_size=args.img_crop_size if cnn_preprocessor else 224,
                img_loader=shared_img_loader,
            )
            if r:
                overlay_curves[f"CNN ({backbone})"] = (r["fpr"], r["tpr"], r["auc_mean"], r["auc_std"])

    # Overlay ROC
    if len(overlay_curves) > 1:
        _plot_overlay(overlay_curves, "PAPILA Reproduce — Clinical + CNN ROC", out_dir / "roc_overlay.png")
        print(f"\nOverlay ROC saved: {out_dir / 'roc_overlay.png'}")

    print(f"\nDone in {time.time()-t0:.1f}s — results in {out_dir}")


if __name__ == "__main__":
    main()
