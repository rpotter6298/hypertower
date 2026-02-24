#!/usr/bin/env python3
"""Train a CNN (resnet50 backbone), extract logits, and train RF on logits+metadata with 5-fold CV."""
from __future__ import annotations

import random
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classes import build_papila_clinical
from classes.backbones import BACKBONES, load_backbone_weights


# ---------------------------
# Config (edit in IDE)
# ---------------------------
IMAGE_DIR = "Papila/FundusImages"
CLINICAL_DIR = "Papila/ClinicalData"
LABEL_COL = "Diagnosis"
CAT_COLS = ["Gender", "Phakic/Pseudophakic"]
EVAL_MODE = "binary"  # "binary" or "multiclass"
N_SPLITS = 5
FOLD_SEED = 42
HOLDOUT_SEED = 123
HOLDOUT_PATIENTS_PER_CLASS = 6

BACKBONE_NAME = "resnet50"
BATCH_SIZE = 8
EPOCHS = 40
LR = 1e-4
WEIGHT_DECAY = 1e-5

RF_TREES = 500
RF_MAX_DEPTH = None
RF_MIN_SAMPLES_LEAF = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("analysis_data/basic_analysis/cnn_logits_rf_cv")
PRINT_EPOCH_REPORT = True
EPOCH_REPORT_EVERY = 1


class PapilaImageDataset(Dataset):
    def __init__(
        self, clinical, df: pd.DataFrame, label_col: str, img_transform
    ) -> None:
        self.clinical = clinical
        self.df = df.reset_index(drop=True)
        self.label_col = label_col
        self.img_transform = img_transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.clinical.get_image_path(row)
        image = Image.open(img_path).convert("RGB")
        x_img = self.img_transform(image)
        y = int(row[self.label_col])
        x_md = self.clinical.vectorize_row(row).astype(np.float32)
        return x_img, y, x_md


class CNNHead(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int) -> None:
        super().__init__()
        spec = BACKBONES[backbone_name]
        backbone = spec.ctor(weights=spec.weights_default)
        if backbone_name.startswith("refuge"):
            load_backbone_weights(backbone_name, backbone)
        out_dim, backbone = spec.strip(backbone)
        self.backbone = backbone
        self.head = nn.Linear(out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _auc_score(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> float:
    try:
        if num_classes == 2:
            return float(roc_auc_score(y_true, probs[:, 1]))
        return float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")


def _prepare_clinical() -> Tuple[object, pd.DataFrame]:
    clinical = build_papila_clinical(
        image_dir=IMAGE_DIR,
        clinical_dir=CLINICAL_DIR,
        label_col=LABEL_COL,
        cat_cols=CAT_COLS,
        n_splits=N_SPLITS,
        random_seed=FOLD_SEED,
    )
    df = clinical.df.copy()
    if EVAL_MODE == "binary":
        df = df[df[LABEL_COL].isin([0, 1])].reset_index(drop=True)
    return clinical, df


def _split_holdout_by_patient(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(HOLDOUT_SEED)
    patient_label = (
        df.groupby("Patient ID")[LABEL_COL]
        .agg(lambda s: int(s.mode().iloc[0]))
        .reset_index()
    )
    holdout_patients = []
    for lbl, grp in patient_label.groupby(LABEL_COL):
        candidates = grp["Patient ID"].to_numpy()
        n = min(HOLDOUT_PATIENTS_PER_CLASS, len(candidates))
        if n <= 0:
            continue
        selected = rng.choice(candidates, size=n, replace=False)
        holdout_patients.extend(selected.tolist())
    holdout_patients = sorted(set(holdout_patients))
    holdout_df = df[df["Patient ID"].isin(holdout_patients)].reset_index(drop=True)
    train_df = df[~df["Patient ID"].isin(holdout_patients)].reset_index(drop=True)
    return train_df, holdout_df


def _rebuild_clinical_from_df(clinical, df: pd.DataFrame) -> object:
    clinical.frames = [df.copy()]
    clinical.df = df.copy()
    clinical._infer_or_validate_feature_types()
    clinical._compute_numeric_stats()
    clinical._build_cat_maps()
    clinical._compute_feature_dim()
    clinical._build_kfold_indices()
    return clinical


def _train_cnn(
    model: nn.Module, loader: DataLoader, num_classes: int, fold: int
) -> None:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        for x_img, y, _x_md in loader:
            x_img = x_img.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x_img)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * int(y.size(0))
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.size(0))

        if PRINT_EPOCH_REPORT and ((epoch + 1) % EPOCH_REPORT_EVERY == 0):
            avg_loss = running_loss / max(total, 1)
            train_acc = correct / max(total, 1)
            print(
                f"[fold {fold + 1}/{N_SPLITS}] epoch {epoch + 1}/{EPOCHS} "
                f"train_loss={avg_loss:.4f} train_acc={train_acc:.4f}"
            )


def _infer_logits(
    model: nn.Module, loader: DataLoader
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits_all, probs_all, y_all, md_all = [], [], [], []
    with torch.no_grad():
        for x_img, y, x_md in loader:
            x_img = x_img.to(DEVICE)
            logits = model(x_img).cpu().numpy()
            probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
            logits_all.append(logits)
            probs_all.append(probs)
            y_all.append(y.numpy())
            md_all.append(x_md.numpy())
    return (
        np.concatenate(y_all, axis=0),
        np.concatenate(logits_all, axis=0),
        np.concatenate(md_all, axis=0),
    )


def main() -> None:
    _set_seed(FOLD_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    num_classes = 2 if EVAL_MODE == "binary" else 3
    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    clinical, df = _prepare_clinical()
    train_df, holdout_df = _split_holdout_by_patient(df)
    clinical = _rebuild_clinical_from_df(clinical, train_df)
    holdout_df.to_csv(OUTPUT_DIR / "holdout_patients.csv", index=False)

    rows: List[Dict[str, object]] = []
    holdout_rows: List[Dict[str, object]] = []

    for fold in range(N_SPLITS):
        print(f"\n[info] Starting fold {fold + 1}/{N_SPLITS}")
        fold_train_df, fold_val_df = clinical.get_split_dfs(fold)
        ds_train = PapilaImageDataset(clinical, fold_train_df, LABEL_COL, train_tf)
        ds_val = PapilaImageDataset(clinical, fold_val_df, LABEL_COL, eval_tf)
        ds_holdout = PapilaImageDataset(clinical, holdout_df, LABEL_COL, eval_tf)

        dl_train = DataLoader(
            ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )
        dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        dl_holdout = DataLoader(
            ds_holdout, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )

        model = CNNHead(BACKBONE_NAME, num_classes=num_classes).to(DEVICE)
        _train_cnn(model, dl_train, num_classes=num_classes, fold=fold)

        y_tr, log_tr, md_tr = _infer_logits(
            model,
            DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
        )
        y_va, log_va, md_va = _infer_logits(model, dl_val)
        y_ho, log_ho, md_ho = _infer_logits(model, dl_holdout)

        np.save(OUTPUT_DIR / f"fold{fold}_train_logits.npy", log_tr)
        np.save(OUTPUT_DIR / f"fold{fold}_val_logits.npy", log_va)
        np.save(OUTPUT_DIR / f"fold{fold}_holdout_logits.npy", log_ho)

        X_tr = np.concatenate([log_tr, md_tr], axis=1)
        X_va = np.concatenate([log_va, md_va], axis=1)
        X_ho = np.concatenate([log_ho, md_ho], axis=1)

        rf = RandomForestClassifier(
            n_estimators=RF_TREES,
            max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            class_weight="balanced",
            random_state=FOLD_SEED + fold,
            n_jobs=-1,
        )
        rf.fit(X_tr, y_tr)

        p_va = rf.predict_proba(X_va)
        p_ho = rf.predict_proba(X_ho)
        pred_va = np.argmax(p_va, axis=1)
        pred_ho = np.argmax(p_ho, axis=1)

        rows.append(
            {
                "fold": fold,
                "val_acc": float(accuracy_score(y_va, pred_va)),
                "val_auc": _auc_score(y_va, p_va, num_classes),
                "n_val": int(len(y_va)),
            }
        )
        holdout_rows.append(
            {
                "fold": fold,
                "holdout_acc": float(accuracy_score(y_ho, pred_ho)),
                "holdout_auc": _auc_score(y_ho, p_ho, num_classes),
                "n_holdout": int(len(y_ho)),
            }
        )
        print(
            f"[info] Fold {fold + 1} RF: val_acc={rows[-1]['val_acc']:.4f} val_auc={rows[-1]['val_auc']:.4f} "
            f"| holdout_acc={holdout_rows[-1]['holdout_acc']:.4f} holdout_auc={holdout_rows[-1]['holdout_auc']:.4f}"
        )

    fold_df = pd.DataFrame(rows)
    holdout_df = pd.DataFrame(holdout_rows)
    fold_df.to_csv(OUTPUT_DIR / "rf_val_metrics.csv", index=False)
    holdout_df.to_csv(OUTPUT_DIR / "rf_holdout_metrics.csv", index=False)

    print("\nRF validation metrics:")
    print(fold_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nRF holdout metrics:")
    print(holdout_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(
        f"\nMeans: val_acc={fold_df['val_acc'].mean():.4f}, val_auc={fold_df['val_auc'].mean():.4f}, "
        f"holdout_acc={holdout_df['holdout_acc'].mean():.4f}, holdout_auc={holdout_df['holdout_auc'].mean():.4f}"
    )
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
