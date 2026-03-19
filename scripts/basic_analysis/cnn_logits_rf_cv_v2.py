#!/usr/bin/env python3
"""v2 port of cnn_logits_rf_cv: CNN logit extraction + RF CV using the v2 PAPILA stack.

Replaces the hardcoded-config v1 version.  Data loading, fold splitting, and
feature preparation all go through the v2 stack so that --exclude-cols,
--iop-corr-method, etc. are first-class options.

Usage:
    python scripts/basic_analysis/cnn_logits_rf_cv_v2.py \
        --eval-mode binary --backbone resnet50 --epochs 40 \
        --exclude-cols Phakic/Pseudophakic Axial_Length \
        --run-name cnn_rf_nocrop_binary
"""
from __future__ import annotations

import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classes.v2.papila_builders import build_papila_data
from classes.v2.backbones import BACKBONES, load_backbone_weights
from classes.v2.dataset import ClinicalDataset, _ClinicalView
from classes.v2.split_manager import PatientFirstSplitManager
from classes.v2.transforms import build_backbone_transform, build_eval_transform


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

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
        return self.head(self.backbone(x))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auc_score(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> float:
    try:
        if num_classes == 2:
            return float(roc_auc_score(y_true, probs[:, 1]))
        return float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")


def _train_cnn(
    model: nn.Module,
    loader: DataLoader,
    args,
    fold: int,
    device: torch.device,
) -> None:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        running_loss = total = correct = 0
        for x_img, _meta, y in loader:
            x_img = x_img.to(device)
            y = y.to(device=device, dtype=torch.long)
            optimizer.zero_grad()
            logits = model(x_img)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * int(y.size(0))
            correct += int((logits.argmax(1) == y).sum().item())
            total += int(y.size(0))
        if (epoch + 1) % args.log_every == 0:
            print(
                f"  [fold {fold+1}] epoch {epoch+1}/{args.epochs} "
                f"loss={running_loss/max(total,1):.4f} acc={correct/max(total,1):.4f}",
                flush=True,
            )


def _infer_logits(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (y_true, logits, metadata_vectors)."""
    model.eval()
    y_all, logits_all, md_all = [], [], []
    with torch.no_grad():
        for x_img, x_md, y in loader:
            logits = model(x_img.to(device)).cpu().numpy()
            y_np = y.numpy() if torch.is_tensor(y) else np.asarray(y)
            md_np = x_md.numpy() if torch.is_tensor(x_md) else np.asarray(x_md)
            y_all.append(y_np)
            logits_all.append(logits)
            md_all.append(md_np)
    return (
        np.concatenate(y_all),
        np.concatenate(logits_all),
        np.concatenate(md_all),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="CNN logit extraction + RF CV (v2 PAPILA stack)"
    )
    ap.add_argument("--image-dir",    default="Papila/FundusImages")
    ap.add_argument("--clinical-dir", default="Papila/ClinicalData")
    ap.add_argument("--label-col",    default="Diagnosis")
    ap.add_argument("--cat-cols",     nargs="*", default=["Gender", "Phakic/Pseudophakic"])
    ap.add_argument("--exclude-cols", nargs="*", default=[],
                    help="Feature columns to drop from the clinical feature matrix.")
    ap.add_argument("--eval-mode",    choices=["binary", "multiclass"], default="binary")
    ap.add_argument("--n-splits",     type=int,   default=5)
    ap.add_argument("--fold-seed",    type=int,   default=42)
    ap.add_argument("--holdout-per-class", type=int, default=6,
                    help="Patients per class reserved for holdout (0 disables).")
    ap.add_argument("--holdout-seed", type=int,   default=123)
    ap.add_argument("--backbone",     default="resnet50")
    ap.add_argument("--batch-size",   type=int,   default=8)
    ap.add_argument("--epochs",       type=int,   default=40)
    ap.add_argument("--lr",           type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--rf-trees",     type=int,   default=500)
    ap.add_argument("--rf-max-depth", type=int,   default=None)
    ap.add_argument("--rf-min-samples-leaf", type=int, default=1)
    ap.add_argument("--num-workers",  type=int,   default=0)
    ap.add_argument("--device",       choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--seed",         type=int,   default=1234)
    ap.add_argument("--log-every",    type=int,   default=5)
    ap.add_argument("--run-name",     default=None)
    ap.add_argument("--output-root",  default="analysis_data/basic_analysis/cnn_logits_rf_cv_v2")
    ap.add_argument("--iop-corr-method", choices=["ratio", "ols", "lad", "multi"], default="ratio")
    ap.add_argument("--iop-drop-raw",    action="store_true", default=False)
    ap.add_argument("--in-memory-cache", action="store_true", default=True,
                    help="Cache all images in RAM before training (default: on).")
    ap.add_argument("--no-in-memory-cache", action="store_false", dest="in_memory_cache",
                    help="Disable in-memory image cache.")
    ap.add_argument("--cache-workers",   type=int, default=4,
                    help="Threads for prebuilding image cache (default: 4).")
    return ap


def _prebuild_image_cache(df: "pd.DataFrame", data, n_workers: int,
                          resize: int = 256) -> dict:
    """Load, convert to RGB, resize to `resize`px short edge, and cache as uint8 arrays.

    Storing pre-resized images means the per-batch transform only has to do
    CenterCrop + augmentation + ToTensor + Normalize on a small image rather
    than resizing a full-resolution fundus image every step.
    """
    paths = list({str(data.get_image_path(row)) for _, row in df.iterrows()})
    cache: dict = {}
    print(f"[cache] Preloading {len(paths)} images (resize={resize}px) "
          f"with {n_workers} threads...", flush=True)

    def _load(p: str):
        img = Image.open(p)
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        scale = resize / min(w, h)
        img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
        return p, np.asarray(img, dtype=np.uint8)

    with ThreadPoolExecutor(max_workers=max(1, n_workers)) as ex:
        for path, arr in ex.map(_load, paths):
            cache[path] = arr

    ex_shape = cache[paths[0]].shape
    print(f"[cache] Done — {len(cache)} images in RAM "
          f"({ex_shape[1]}×{ex_shape[0]} each).", flush=True)
    return cache


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}", flush=True)

    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- data ----------
    print("Loading PAPILA data...", flush=True)
    data = build_papila_data(
        image_dir=args.image_dir,
        clinical_dir=args.clinical_dir,
        label_col=args.label_col,
        cat_cols=list(args.cat_cols),
        n_splits=args.n_splits,
        random_seed=args.fold_seed,
        iop_corr_method=args.iop_corr_method,
        iop_drop_raw=args.iop_drop_raw,
        exclude_cols=list(args.exclude_cols or []),
    )
    print(f"Loaded: {len(data.df)} rows  feature_dim={data.feature_dim}", flush=True)

    df_mode = data.df.copy()
    if args.eval_mode == "binary":
        df_mode = df_mode[df_mode[args.label_col].isin([0, 1])].reset_index(drop=True)
    num_classes = 2 if args.eval_mode == "binary" else int(df_mode[args.label_col].nunique())
    print(
        f"eval_mode={args.eval_mode}  num_classes={num_classes}  "
        f"rows={len(df_mode)}  patients={df_mode['Patient ID'].nunique()}",
        flush=True,
    )

    # ---------- splits ----------
    split_manager = PatientFirstSplitManager(
        patient_col="Patient ID", label_col=args.label_col
    )
    split_args = SimpleNamespace(
        eval_mode=args.eval_mode,
        holdout_per_class=args.holdout_per_class,
        holdout_seed=args.holdout_seed,
        n_splits=args.n_splits,
        fold_seed=args.fold_seed,
    )
    clinical_ns = SimpleNamespace(df=df_mode, label_col=args.label_col)
    plans = split_manager.build_plans(clinical=clinical_ns, args=split_args, profile=None)
    n_folds = min(args.n_splits, len(plans))

    if plans and plans[0].holdout is not None:
        plans[0].holdout.to_csv(out_dir / "holdout_patients.csv", index=False)

    # ---------- image cache ----------
    image_cache = None
    if args.in_memory_cache:
        image_cache = _prebuild_image_cache(df_mode, data, args.cache_workers)

    # ---------- transforms ----------
    train_tf = build_backbone_transform(args.backbone, augment=True)
    eval_tf  = build_eval_transform(args.backbone)

    rows: List[Dict] = []
    holdout_rows: List[Dict] = []

    for fold, split in enumerate(plans[:n_folds]):
        print(f"\n[info] Fold {fold+1}/{n_folds}", flush=True)
        random.seed(args.seed + fold * 100)
        np.random.seed(args.seed + fold * 100)
        torch.manual_seed(args.seed + fold * 100)

        view_train = _ClinicalView(data, split.train)
        view_val   = _ClinicalView(data, split.val)

        dl_train = DataLoader(
            ClinicalDataset(view_train, train_tf, image_cache=image_cache),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        )
        dl_val = DataLoader(
            ClinicalDataset(view_val, eval_tf, image_cache=image_cache),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        )
        dl_holdout = None
        if split.holdout is not None and not split.holdout.empty:
            dl_holdout = DataLoader(
                ClinicalDataset(_ClinicalView(data, split.holdout), eval_tf,
                                image_cache=image_cache),
                batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
            )

        # Train CNN
        model = CNNHead(args.backbone, num_classes=num_classes).to(device)
        _train_cnn(model, dl_train, args, fold=fold, device=device)

        # Extract logits (re-run train without augmentation for RF features)
        dl_train_eval = DataLoader(
            ClinicalDataset(view_train, eval_tf, image_cache=image_cache),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        )
        y_tr, log_tr, md_tr = _infer_logits(model, dl_train_eval, device)
        y_va, log_va, md_va = _infer_logits(model, dl_val, device)

        np.save(out_dir / f"fold{fold}_train_logits.npy", log_tr)
        np.save(out_dir / f"fold{fold}_val_logits.npy",   log_va)

        X_tr = np.concatenate([log_tr, md_tr], axis=1)
        X_va = np.concatenate([log_va, md_va], axis=1)

        rf = RandomForestClassifier(
            n_estimators=args.rf_trees,
            max_depth=args.rf_max_depth,
            min_samples_leaf=args.rf_min_samples_leaf,
            class_weight="balanced",
            random_state=args.fold_seed + fold,
            n_jobs=-1,
        )
        rf.fit(X_tr, y_tr)

        p_va   = rf.predict_proba(X_va)
        pred_va = np.argmax(p_va, axis=1)
        rows.append({
            "fold": fold,
            "val_acc": float(accuracy_score(y_va, pred_va)),
            "val_auc": _auc_score(y_va, p_va, num_classes),
            "n_val":   int(len(y_va)),
        })

        if dl_holdout is not None:
            y_ho, log_ho, md_ho = _infer_logits(model, dl_holdout, device)
            np.save(out_dir / f"fold{fold}_holdout_logits.npy", log_ho)
            X_ho   = np.concatenate([log_ho, md_ho], axis=1)
            p_ho   = rf.predict_proba(X_ho)
            pred_ho = np.argmax(p_ho, axis=1)
            holdout_rows.append({
                "fold": fold,
                "holdout_acc": float(accuracy_score(y_ho, pred_ho)),
                "holdout_auc": _auc_score(y_ho, p_ho, num_classes),
                "n_holdout":   int(len(y_ho)),
            })
            print(
                f"[info] Fold {fold+1} RF: val_acc={rows[-1]['val_acc']:.4f}"
                f" val_auc={rows[-1]['val_auc']:.4f}"
                f" | holdout_acc={holdout_rows[-1]['holdout_acc']:.4f}"
                f" holdout_auc={holdout_rows[-1]['holdout_auc']:.4f}",
                flush=True,
            )
        else:
            print(
                f"[info] Fold {fold+1} RF: val_acc={rows[-1]['val_acc']:.4f}"
                f" val_auc={rows[-1]['val_auc']:.4f}",
                flush=True,
            )

    # ---------- save + summarise ----------
    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(out_dir / "rf_val_metrics.csv", index=False)
    print("\nRF validation metrics:")
    print(fold_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if holdout_rows:
        ho_df = pd.DataFrame(holdout_rows)
        ho_df.to_csv(out_dir / "rf_holdout_metrics.csv", index=False)
        print("\nRF holdout metrics:")
        print(ho_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(
            f"\nMeans: val_acc={fold_df['val_acc'].mean():.4f}"
            f"  val_auc={fold_df['val_auc'].mean():.4f}"
            f"  holdout_acc={ho_df['holdout_acc'].mean():.4f}"
            f"  holdout_auc={ho_df['holdout_auc'].mean():.4f}"
        )
    else:
        print(
            f"\nMeans: val_acc={fold_df['val_acc'].mean():.4f}"
            f"  val_auc={fold_df['val_auc'].mean():.4f}"
        )

    print(f"\nSaved outputs to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
