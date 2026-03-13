#!/usr/bin/env python3
"""
Compare all-eye baseline (ImageTower) vs SiameseImageTower bilateral model.

Key differences from compare_siamese_tower.py (v1):
  - Baseline trains on ALL eye samples (both OD + OS rows) — matches the
    original grid-search training regime, not just OD-from-bilateral.
  - No holdout set: pure k-fold CV is sufficient for architecture comparison.
  - Both models evaluated at patient level on the same bilateral val set:
      - Baseline: runs on OD and OS separately, mean-pools probabilities.
      - Siamese:  runs on both eyes simultaneously.
  - Extended metrics at best-epoch snapshots: kappa, MCC, macro-F1,
    per-class recall, and ECE (Expected Calibration Error).
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classes.v2 import (
    ImageTower,
    PatientFirstSplitManager,
    SiameseImageTower,
    SlotDataset,
    build_papila_data,
    build_papila_profile,
    slot_collate,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class BaselineCNN(nn.Module):
    """Single-eye image tower with a linear head."""

    def __init__(self, *, backbone: str, freeze_ratio: float, num_classes: int, augment: bool):
        super().__init__()
        self.tower = ImageTower(
            backbone=backbone,
            freeze_ratio=freeze_ratio,
            augment=augment,
            use_se=False,
        )
        self.head = nn.Linear(self.tower.out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.tower(x))


class SiameseCNN(nn.Module):
    """Bilateral image model using SiameseImageTower (shared backbone)."""

    def __init__(self, *, backbone: str, freeze_ratio: float, num_classes: int, augment: bool):
        super().__init__()
        self.tower = SiameseImageTower(
            backbone=backbone,
            freeze_ratio=freeze_ratio,
            augment=augment,
            use_se=False,
        )
        self.head = nn.Linear(self.tower.out_dim, num_classes)

    def forward(self, x_od: torch.Tensor, x_os: torch.Tensor) -> torch.Tensor:
        return self.head(self.tower(x_od, x_os))


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def filter_eye_samples(samples: list[dict]) -> list[dict]:
    """Keep any single-eye sample with a valid image and label (OD or OS)."""
    return [s for s in samples if s.get("image_1") is not None and s.get("label_1") is not None]


def filter_bilateral_samples(samples: list[dict]) -> list[dict]:
    """Keep only patient-level samples where both eyes are present."""
    return [
        s for s in samples
        if s.get("image_1") is not None
        and s.get("image_2") is not None
        and s.get("label_1") is not None
    ]


def make_loader(
    samples: list[dict],
    slots: dict,
    *,
    image_transform,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    ds = SlotDataset(samples, slots, image_transform=image_transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=slot_collate,
    )


def to_label_tensor(labels, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(labels):
        return labels.to(device=device, dtype=torch.long)
    return torch.as_tensor(labels, dtype=torch.long, device=device)


def _drop_mixed_label_patients(df, *, patient_col: str, label_col: str):
    per_patient = (
        df.groupby(patient_col)[label_col]
        .agg(lambda s: set(pd.to_numeric(s, errors="coerce").dropna().astype(int).tolist()))
    )
    mixed = [pid for pid, labels in per_patient.items() if len(labels) > 1]
    if not mixed:
        return df, []
    return df[~df[patient_col].isin(mixed)].reset_index(drop=True), mixed


def _relabel_mixed_patients_to_max(df, *, patient_col: str, label_col: str):
    """Set all rows for each patient to that patient's max observed label."""
    out = df.copy()
    labels = pd.to_numeric(out[label_col], errors="coerce")
    patient_max = labels.groupby(out[patient_col]).transform("max")
    changed_rows = int((labels != patient_max).fillna(False).sum())
    out[label_col] = patient_max.astype(int)
    per_patient_unique = (
        out.groupby(patient_col)[label_col]
        .nunique(dropna=True)
    )
    mixed_patients = per_patient_unique[per_patient_unique > 1].index.tolist()
    return out.reset_index(drop=True), changed_rows, mixed_patients


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def compute_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error: weighted mean of |confidence - accuracy| per bin."""
    if y_true.size == 0:
        return float("nan")
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (confidences >= lo) & (confidences <= hi if i == n_bins - 1 else confidences < hi)
        if not mask.any():
            continue
        bin_acc  = float((predictions[mask] == y_true[mask]).mean())
        bin_conf = float(confidences[mask].mean())
        ece += float(mask.sum()) / n * abs(bin_conf - bin_acc)
    return float(ece)


def compute_extended_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    num_classes: int,
    n_bins: int = 10,
    preds_override: Optional[np.ndarray] = None,
) -> dict:
    """
    Returns kappa, mcc, macro_f1, per_class_recall (np.ndarray), ece.
    All float('nan') on empty input or single-class edge cases.
    """
    nan = float("nan")
    if y_true.size == 0:
        return dict(kappa=nan, mcc=nan, macro_f1=nan,
                    per_class_recall=np.full(num_classes, nan), ece=nan)
    preds = preds_override if preds_override is not None else probs.argmax(axis=1)
    try:
        kappa = float(cohen_kappa_score(y_true, preds))
    except Exception:
        kappa = nan
    try:
        mcc = float(matthews_corrcoef(y_true, preds))
    except Exception:
        mcc = nan
    try:
        macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
    except Exception:
        macro_f1 = nan
    try:
        pcr = recall_score(
            y_true, preds, average=None,
            labels=list(range(num_classes)), zero_division=0,
        ).astype(float)
    except Exception:
        pcr = np.full(num_classes, nan)
    ece = compute_ece(y_true, probs, n_bins=n_bins)
    return dict(kappa=kappa, mcc=mcc, macro_f1=macro_f1, per_class_recall=pcr, ece=ece)


def tune_binary_threshold(y_true: np.ndarray, p1: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.5
    grid = np.linspace(0.0, 1.0, 1001)
    best_t = 0.5
    best_acc = -1.0
    for t in grid:
        pred = (p1 >= t).astype(int)
        acc = float((pred == y_true).mean())
        if acc > best_acc or (acc == best_acc and abs(t - 0.5) < abs(best_t - 0.5)):
            best_acc = acc
            best_t = float(t)
    return best_t


def multiclass_acc_with_bias(y_true: np.ndarray, probs: np.ndarray, bias: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    logits = np.log(np.clip(probs, 1e-8, 1.0)) + bias.reshape(1, -1)
    pred = np.argmax(logits, axis=1)
    return float((pred == y_true).mean())


def tune_multiclass_bias(y_true: np.ndarray, probs: np.ndarray, *, iters: int = 2) -> np.ndarray:
    if y_true.size == 0 or probs.size == 0:
        return np.zeros((0,), dtype=float)
    c = probs.shape[1]
    bias = np.zeros((c,), dtype=float)
    grid = np.linspace(-1.0, 1.0, 41)
    for _ in range(iters):
        for k in range(c):
            best_v = bias[k]
            best_acc = multiclass_acc_with_bias(y_true, probs, bias)
            old = bias[k]
            for v in grid:
                bias[k] = float(v)
                acc = multiclass_acc_with_bias(y_true, probs, bias)
                if acc > best_acc or (acc == best_acc and abs(v) < abs(best_v)):
                    best_acc = acc
                    best_v = float(v)
            bias[k] = best_v
            if np.isnan(best_acc):
                bias[k] = old
    return bias


def _svf(vec) -> Optional[str]:
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=float)
    if arr.size == 0:
        return None
    return "|".join(f"{float(v):.4f}" for v in arr.tolist())


def _score_arrays(y_true: np.ndarray, probs: np.ndarray, num_classes: int):
    """Score pre-collected arrays. Returns (acc, auc, n)."""
    if y_true.size == 0:
        return float("nan"), float("nan"), 0
    acc = float((probs.argmax(1) == y_true).mean())
    try:
        auc = (
            float(roc_auc_score(y_true, probs[:, 1]))
            if num_classes == 2
            else float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
        )
    except Exception:
        auc = float("nan")
    return acc, auc, int(len(y_true))


# ---------------------------------------------------------------------------
# Train / collect
# ---------------------------------------------------------------------------

def train_baseline_epoch(model: BaselineCNN, loader: DataLoader, opt, device):
    """Train on single-eye batches (image_1)."""
    model.train()
    total_loss = total_correct = total_n = 0
    for batch in loader:
        x = batch.get("image_1")
        y = batch.get("label_1")
        if not torch.is_tensor(x):
            continue
        y = to_label_tensor(y, device)
        x = x.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        bs = y.shape[0]
        total_loss    += float(loss.item()) * bs
        total_correct += int((logits.argmax(1) == y).sum())
        total_n       += bs
    return (
        total_loss    / total_n if total_n else float("nan"),
        total_correct / total_n if total_n else float("nan"),
    )


def train_siamese_epoch(model: SiameseCNN, loader: DataLoader, opt, device):
    """Train on bilateral patient batches (image_1 = OD, image_2 = OS)."""
    model.train()
    total_loss = total_correct = total_n = 0
    for batch in loader:
        x1 = batch.get("image_1")
        x2 = batch.get("image_2")
        y  = batch.get("label_1")
        if not torch.is_tensor(x1) or not torch.is_tensor(x2):
            continue
        y = to_label_tensor(y, device)
        logits = model(x1.to(device), x2.to(device))
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        bs = y.shape[0]
        total_loss    += float(loss.item()) * bs
        total_correct += int((logits.argmax(1) == y).sum())
        total_n       += bs
    return (
        total_loss    / total_n if total_n else float("nan"),
        total_correct / total_n if total_n else float("nan"),
    )


def collect_probs_baseline_bilateral(
    model: BaselineCNN,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate baseline on bilateral (patient-level) samples.

    For each patient, runs the single-eye model on OD (image_1) and OS
    (image_2) separately, then mean-pools the softmax probabilities.
    Returns (y_true [N], probs [N, C]) at patient level.
    """
    model.eval()
    y_chunks, p_chunks = [], []
    with torch.no_grad():
        for batch in loader:
            x1 = batch.get("image_1")
            x2 = batch.get("image_2")
            y  = batch.get("label_1")
            if not torch.is_tensor(x1) or not torch.is_tensor(x2):
                continue
            y_t = to_label_tensor(y, device)
            p_od = F.softmax(model(x1.to(device)), dim=1)
            p_os = F.softmax(model(x2.to(device)), dim=1)
            p    = 0.5 * (p_od + p_os)
            y_chunks.append(y_t.cpu().numpy())
            p_chunks.append(p.cpu().numpy())
    if not y_chunks:
        return np.array([], dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(y_chunks), np.concatenate(p_chunks, axis=0)


def collect_probs_siamese(
    model: SiameseCNN,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate siamese on bilateral (patient-level) samples."""
    model.eval()
    y_chunks, p_chunks = [], []
    with torch.no_grad():
        for batch in loader:
            x1 = batch.get("image_1")
            x2 = batch.get("image_2")
            y  = batch.get("label_1")
            if not torch.is_tensor(x1) or not torch.is_tensor(x2):
                continue
            y_t = to_label_tensor(y, device)
            p   = F.softmax(model(x1.to(device), x2.to(device)), dim=1)
            y_chunks.append(y_t.cpu().numpy())
            p_chunks.append(p.cpu().numpy())
    if not y_chunks:
        return np.array([], dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(y_chunks), np.concatenate(p_chunks, axis=0)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

def _nan() -> float:
    return float("nan")


@dataclass
class FoldResult:
    mode: str
    fold: int
    # Epoch where each model hit its peak val AUC
    best_epoch_base: int
    best_epoch_siam: int
    # Baseline metrics at best-val epoch (patient-level, prob-aggregated)
    base_val_auc:    float
    base_val_acc:    float
    base_val_kappa:  float
    base_val_mcc:    float
    base_val_f1:     float
    base_val_recall: Optional[str]   # pipe-delimited per-class recall
    base_val_ece:    float
    base_val_threshold: float
    base_val_bias: Optional[str]
    base_val_n:      int
    # Siamese metrics at best-val epoch
    siam_val_auc:    float
    siam_val_acc:    float
    siam_val_kappa:  float
    siam_val_mcc:    float
    siam_val_f1:     float
    siam_val_recall: Optional[str]
    siam_val_ece:    float
    siam_val_threshold: float
    siam_val_bias: Optional[str]
    siam_val_n:      int
    # Train sample sizes (informational)
    base_train_n:    int
    siam_train_n:    int


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _f(v) -> Optional[float]:
    """Nan-safe float serialiser."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), 6)


def _sv(vec) -> Optional[str]:
    """Serialise a numeric vector to a pipe-delimited string."""
    if vec is None:
        return None
    return "|".join(f"{float(v):.4f}" for v in vec)


# ---------------------------------------------------------------------------
# Main fold runner
# ---------------------------------------------------------------------------

def run_fold(
    fold: int,
    split,
    mode: str,
    args,
    device: torch.device,
    data,
    num_classes: int,
    profile_eye,
    profile_patient,
    fold_dir: Path,
) -> FoldResult:
    # ---- build samples -------------------------------------------------------
    # Baseline trains on ALL eye-level samples (OD + OS as separate rows)
    eye_train = filter_eye_samples(
        profile_eye.build_samples(df=split.train, clinical=data)
    )
    # Siamese trains on bilateral patient-level samples
    bilat_train = filter_bilateral_samples(
        profile_patient.build_samples(df=split.train, clinical=data)
    )
    # Val: bilateral patients only — shared between both model evaluations
    bilat_val = filter_bilateral_samples(
        profile_patient.build_samples(df=split.val, clinical=data)
    )

    if len(bilat_val) == 0:
        print(f"  [fold {fold+1}] WARNING: no bilateral val samples; skipping fold.", flush=True)
        nan = _nan()
        return FoldResult(
            mode=mode, fold=fold,
            best_epoch_base=0, best_epoch_siam=0,
            base_val_auc=nan, base_val_acc=nan, base_val_kappa=nan,
            base_val_mcc=nan, base_val_f1=nan, base_val_recall=None,
            base_val_ece=nan, base_val_threshold=nan, base_val_bias=None, base_val_n=0,
            siam_val_auc=nan, siam_val_acc=nan, siam_val_kappa=nan,
            siam_val_mcc=nan, siam_val_f1=nan, siam_val_recall=None,
            siam_val_ece=nan, siam_val_threshold=nan, siam_val_bias=None, siam_val_n=0,
            base_train_n=len(eye_train), siam_train_n=len(bilat_train),
        )

    # ---- models --------------------------------------------------------------
    baseline = BaselineCNN(
        backbone=args.backbone, freeze_ratio=args.freeze_ratio,
        num_classes=num_classes, augment=args.augment,
    ).to(device)
    siamese = SiameseCNN(
        backbone=args.backbone, freeze_ratio=args.freeze_ratio,
        num_classes=num_classes, augment=args.augment,
    ).to(device)

    slots_eye     = profile_eye.slot_descriptors()
    slots_patient = profile_patient.slot_descriptors()
    loader_kw     = dict(batch_size=args.batch_size, num_workers=args.num_workers)

    # ---- loaders -------------------------------------------------------------
    # Baseline uses eye-level transform; siamese shares the same backbone
    # transform so both models see the same normalisation at eval time.
    train_base = make_loader(
        eye_train, slots_eye,
        image_transform=baseline.tower.transform, shuffle=True, **loader_kw,
    )
    train_siam = make_loader(
        bilat_train, slots_patient,
        image_transform=siamese.tower.transform, shuffle=True, **loader_kw,
    )
    # Shared val loader — both models read from this
    val_loader = make_loader(
        bilat_val, slots_patient,
        image_transform=baseline.tower.transform, shuffle=False, **loader_kw,
    )

    opt_base = torch.optim.Adam(baseline.parameters(), lr=args.lr)
    opt_siam = torch.optim.Adam(siamese.parameters(),  lr=args.lr)

    # ---- epoch log -----------------------------------------------------------
    epoch_fields = [
        "fold", "epoch",
        "base_train_loss", "base_train_acc",
        "base_val_auc",    "base_val_acc",    "base_val_n",
        "base_val_threshold",
        "base_val_bias",
        "siam_train_loss", "siam_train_acc",
        "siam_val_auc",    "siam_val_acc",    "siam_val_n",
        "siam_val_threshold",
        "siam_val_bias",
        "is_best_base",    "is_best_siam",
    ]
    epoch_fp = (fold_dir / "epoch_log.csv").open("w", newline="", encoding="utf-8")
    epoch_writer = csv.DictWriter(epoch_fp, fieldnames=epoch_fields)
    epoch_writer.writeheader()

    # ---- best-epoch trackers -------------------------------------------------
    best_base_auc = -1.0
    best_siam_auc = -1.0
    best_base_state: Optional[dict] = None
    best_siam_state: Optional[dict] = None
    best_epoch_base = 0
    best_epoch_siam = 0
    snap_base: dict = {}
    snap_siam: dict = {}

    print(
        f"  [fold {fold+1}]  base_train_n={len(eye_train)} (eye-level)  "
        f"siam_train_n={len(bilat_train)} (bilateral)  val_n={len(bilat_val)}",
        flush=True,
    )

    # ---- epoch loop ----------------------------------------------------------
    for epoch in range(args.epochs):
        bl_loss, bl_acc = train_baseline_epoch(baseline, train_base, opt_base, device)
        si_loss, si_acc = train_siamese_epoch(siamese,   train_siam, opt_siam,  device)

        y_b, p_b = collect_probs_baseline_bilateral(baseline, val_loader, device)
        y_s, p_s = collect_probs_siamese(siamese,             val_loader, device)

        b_acc, b_auc, b_n = _score_arrays(y_b, p_b, num_classes)
        s_acc, s_auc, s_n = _score_arrays(y_s, p_s, num_classes)
        b_thr = 0.5
        s_thr = 0.5
        b_bias = None
        s_bias = None
        b_ext_preds = None
        s_ext_preds = None
        if args.tune_binary_threshold and num_classes == 2 and b_n > 0 and s_n > 0:
            b_thr = tune_binary_threshold(y_b, p_b[:, 1])
            s_thr = tune_binary_threshold(y_s, p_s[:, 1])
            b_ext_preds = (p_b[:, 1] >= b_thr).astype(int)
            s_ext_preds = (p_s[:, 1] >= s_thr).astype(int)
            b_acc = float((b_ext_preds == y_b).mean())
            s_acc = float((s_ext_preds == y_s).mean())
        elif args.tune_multiclass_bias and num_classes > 2 and b_n > 0 and s_n > 0:
            b_bias = tune_multiclass_bias(y_b, p_b)
            s_bias = tune_multiclass_bias(y_s, p_s)
            b_logits = np.log(np.clip(p_b, 1e-8, 1.0)) + b_bias.reshape(1, -1)
            s_logits = np.log(np.clip(p_s, 1e-8, 1.0)) + s_bias.reshape(1, -1)
            b_ext_preds = np.argmax(b_logits, axis=1)
            s_ext_preds = np.argmax(s_logits, axis=1)
            b_acc = float((b_ext_preds == y_b).mean())
            s_acc = float((s_ext_preds == y_s).mean())

        # Independent best-epoch update per model
        is_best_base = (not np.isnan(b_auc)) and (b_auc > best_base_auc)
        if is_best_base:
            best_base_auc   = b_auc
            best_base_state = copy.deepcopy(baseline.state_dict())
            best_epoch_base = epoch + 1
            ext = compute_extended_metrics(
                y_b, p_b, num_classes, n_bins=args.ece_bins, preds_override=b_ext_preds
            )
            snap_base = dict(
                auc=b_auc, acc=b_acc, n=b_n,
                kappa=ext["kappa"], mcc=ext["mcc"], macro_f1=ext["macro_f1"],
                per_class_recall=ext["per_class_recall"], ece=ext["ece"], threshold=b_thr, bias=b_bias,
            )

        is_best_siam = (not np.isnan(s_auc)) and (s_auc > best_siam_auc)
        if is_best_siam:
            best_siam_auc   = s_auc
            best_siam_state = copy.deepcopy(siamese.state_dict())
            best_epoch_siam = epoch + 1
            ext = compute_extended_metrics(
                y_s, p_s, num_classes, n_bins=args.ece_bins, preds_override=s_ext_preds
            )
            snap_siam = dict(
                auc=s_auc, acc=s_acc, n=s_n,
                kappa=ext["kappa"], mcc=ext["mcc"], macro_f1=ext["macro_f1"],
                per_class_recall=ext["per_class_recall"], ece=ext["ece"], threshold=s_thr, bias=s_bias,
            )

        epoch_writer.writerow({
            "fold": fold, "epoch": epoch + 1,
            "base_train_loss": _f(bl_loss), "base_train_acc": _f(bl_acc),
            "base_val_auc":    _f(b_auc),   "base_val_acc":   _f(b_acc),   "base_val_n": b_n,
            "base_val_threshold": _f(b_thr if num_classes == 2 else float("nan")),
            "base_val_bias": _svf(b_bias if num_classes > 2 else None),
            "siam_train_loss": _f(si_loss), "siam_train_acc": _f(si_acc),
            "siam_val_auc":    _f(s_auc),   "siam_val_acc":   _f(s_acc),   "siam_val_n": s_n,
            "siam_val_threshold": _f(s_thr if num_classes == 2 else float("nan")),
            "siam_val_bias": _svf(s_bias if num_classes > 2 else None),
            "is_best_base": int(is_best_base),
            "is_best_siam": int(is_best_siam),
        })
        epoch_fp.flush()

        if args.log_every > 0 and (epoch + 1) % args.log_every == 0:
            print(
                f"    ep {epoch+1:>3}/{args.epochs}  "
                f"base val AUC={b_auc:.4f}  siam val AUC={s_auc:.4f}  "
                f"(best base={best_base_auc:.4f} @ep{best_epoch_base}  "
                f"best siam={best_siam_auc:.4f} @ep{best_epoch_siam})",
                flush=True,
            )

    epoch_fp.close()

    if args.save_checkpoints:
        if best_base_state is not None:
            torch.save(best_base_state, fold_dir / "best_baseline.pt")
        if best_siam_state is not None:
            torch.save(best_siam_state, fold_dir / "best_siamese.pt")

    nan = _nan()

    print(
        f"  [fold {fold+1}] BEST  "
        f"base AUC={snap_base.get('auc', nan):.4f} "
        f"kappa={snap_base.get('kappa', nan):.4f} "
        f"F1={snap_base.get('macro_f1', nan):.4f} "
        f"ECE={snap_base.get('ece', nan):.4f} @ep{best_epoch_base}  |  "
        f"siam AUC={snap_siam.get('auc', nan):.4f} "
        f"kappa={snap_siam.get('kappa', nan):.4f} "
        f"F1={snap_siam.get('macro_f1', nan):.4f} "
        f"ECE={snap_siam.get('ece', nan):.4f} @ep{best_epoch_siam}",
        flush=True,
    )

    return FoldResult(
        mode=mode, fold=fold,
        best_epoch_base=best_epoch_base, best_epoch_siam=best_epoch_siam,
        base_val_auc=snap_base.get("auc", nan),
        base_val_acc=snap_base.get("acc", nan),
        base_val_kappa=snap_base.get("kappa", nan),
        base_val_mcc=snap_base.get("mcc", nan),
        base_val_f1=snap_base.get("macro_f1", nan),
        base_val_recall=_sv(snap_base.get("per_class_recall")),
        base_val_ece=snap_base.get("ece", nan),
        base_val_threshold=snap_base.get("threshold", nan),
        base_val_bias=_svf(snap_base.get("bias")),
        base_val_n=snap_base.get("n", 0),
        siam_val_auc=snap_siam.get("auc", nan),
        siam_val_acc=snap_siam.get("acc", nan),
        siam_val_kappa=snap_siam.get("kappa", nan),
        siam_val_mcc=snap_siam.get("mcc", nan),
        siam_val_f1=snap_siam.get("macro_f1", nan),
        siam_val_recall=_sv(snap_siam.get("per_class_recall")),
        siam_val_ece=snap_siam.get("ece", nan),
        siam_val_threshold=snap_siam.get("threshold", nan),
        siam_val_bias=_svf(snap_siam.get("bias")),
        siam_val_n=snap_siam.get("n", 0),
        base_train_n=len(eye_train),
        siam_train_n=len(bilat_train),
    )


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _summary(results: list[FoldResult]) -> dict:
    def _ms(vals):
        v = np.array([x for x in vals if not np.isnan(float(x)) if x is not None], dtype=float)
        return (
            float(np.mean(v)) if v.size else None,
            float(np.std(v))  if v.size else None,
        )

    metrics = ["auc", "acc", "kappa", "mcc", "f1", "ece", "threshold"]
    out = {}
    for label, prefix in [("baseline_best_val", "base_val"), ("siamese_best_val", "siam_val")]:
        sub = {}
        for m in metrics:
            vals = [getattr(r, f"{prefix}_{m}") for r in results]
            mean, std = _ms(vals)
            sub[f"{m}_mean"] = mean
            if m in ("auc", "f1", "kappa"):
                sub[f"{m}_std"] = std
        out[label] = sub

    # Per-fold deltas (siamese − baseline)
    delta = {}
    for m in ["auc", "f1", "kappa"]:
        pairs = [
            getattr(r, f"siam_val_{m}") - getattr(r, f"base_val_{m}")
            for r in results
            if not np.isnan(float(getattr(r, f"base_val_{m}")))
            and not np.isnan(float(getattr(r, f"siam_val_{m}")))
        ]
        delta[f"{m}_mean"] = float(np.mean(pairs)) if pairs else None
        delta[f"{m}_std"]  = float(np.std(pairs))  if pairs else None
    out["delta_val"] = delta

    out["n_folds_completed"] = len(results)
    out["base_train_mode"]   = "eye-level (all OD+OS samples)"
    out["siam_train_mode"]   = "patient-level (bilateral only)"
    out["eval_mode"]         = "patient-level bilateral (both models, same val set)"
    return out


def _print_summary(mode: str, s: dict) -> None:
    def f(v):
        return "nan" if v is None else f"{v:.4f}"

    bv = s["baseline_best_val"]
    sv = s["siamese_best_val"]
    dv = s["delta_val"]

    print(f"\n=== Summary [{mode}] — best-epoch, patient-level bilateral val ===")
    print(f"  {'':22s}  {'AUC':>8}  {'ACC':>8}  {'Kappa':>8}  {'F1-mac':>8}  {'ECE':>8}  {'Thr':>8}")
    print(
        f"  {'baseline (eye-lvl tr)':22s}  "
        f"{f(bv['auc_mean']):>8}  {f(bv['acc_mean']):>8}  "
        f"{f(bv['kappa_mean']):>8}  {f(bv['f1_mean']):>8}  {f(bv['ece_mean']):>8}  {f(bv['threshold_mean']):>8}"
    )
    print(
        f"  {'siamese (bilateral tr)':22s}  "
        f"{f(sv['auc_mean']):>8}  {f(sv['acc_mean']):>8}  "
        f"{f(sv['kappa_mean']):>8}  {f(sv['f1_mean']):>8}  {f(sv['ece_mean']):>8}  {f(sv['threshold_mean']):>8}"
    )
    print(
        f"  {'delta (siam − base)':22s}  "
        f"{f(dv['auc_mean']):>8}  {'':>8}  "
        f"{f(dv['kappa_mean']):>8}  {f(dv['f1_mean']):>8}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "All-eye baseline (ImageTower) vs SiameseImageTower bilateral model. "
            "Pure k-fold CV — no holdout set."
        )
    )
    ap.add_argument("--image-dir",    default="Papila/FundusImages")
    ap.add_argument("--clinical-dir", default="Papila/ClinicalData")
    ap.add_argument("--label-col",    default="Diagnosis")
    ap.add_argument("--cat-cols",     nargs="*", default=["Gender", "Phakic/Pseudophakic"])
    ap.add_argument("--eval-mode",    choices=["binary", "multiclass"], default="multiclass")
    ap.add_argument(
        "--eval-modes", nargs="+", choices=["binary", "multiclass"], default=None,
        help="Run multiple eval modes in one pass.",
    )
    ap.add_argument("--n-splits",          type=int,   default=5)
    ap.add_argument("--fold-seed",         type=int,   default=42)
    ap.add_argument("--folds",             type=int,   default=5)
    ap.add_argument("--epochs",            type=int,   default=40)
    ap.add_argument("--batch-size",        type=int,   default=8)
    ap.add_argument("--lr",                type=float, default=1e-4)
    ap.add_argument("--backbone",          default="refugelike")
    ap.add_argument("--freeze-ratio",      type=float, default=0.0)
    ap.add_argument("--augment",           action="store_true")
    ap.add_argument("--num-workers",       type=int,   default=0)
    ap.add_argument("--device",            choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--seed",              type=int,   default=1234)
    ap.add_argument("--run-name",          default=None)
    ap.add_argument("--output-root",       default="analysis_data/basic_analysis")
    ap.add_argument(
        "--exclude-mixed-patients",
        dest="exclude_mixed_patients",
        action="store_true",
        help="Drop patients whose two eyes have different labels before splitting.",
    )
    ap.add_argument(
        "--include-mixed-patients",
        dest="exclude_mixed_patients",
        action="store_false",
        help="Keep mixed-label patients (default behavior).",
    )
    ap.add_argument(
        "--keep-mixed-raw-labels",
        action="store_true",
        help="When mixed patients are included, keep original per-eye labels (default is relabel to patient max label).",
    )
    ap.set_defaults(exclude_mixed_patients=False)
    ap.add_argument("--log-every",       type=int, default=5)
    ap.add_argument(
        "--tune-binary-threshold",
        action="store_true",
        help="Tune per-model binary threshold on validation each epoch and use it for ACC/F1/Kappa/MCC/recall.",
    )
    ap.add_argument(
        "--tune-multiclass-bias",
        action="store_true",
        help="Tune per-model multiclass log-prob bias on validation each epoch and use it for ACC/F1/Kappa/MCC/recall.",
    )
    ap.add_argument("--ece-bins",        type=int, default=10,
                    help="Number of bins for ECE calibration calculation.")
    ap.add_argument("--save-checkpoints", action="store_true",
                    help="Save best model state dicts (disabled by default to save disk).")
    return ap.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = choose_device(args.device)
    seed_everything(args.seed)

    print(f"Device: {device}", flush=True)
    print("Loading PAPILA data...", flush=True)
    data = build_papila_data(
        image_dir=args.image_dir,
        clinical_dir=args.clinical_dir,
        label_col=args.label_col,
        cat_cols=list(args.cat_cols),
        n_splits=args.n_splits,
        random_seed=args.fold_seed,
    )
    print(f"Loaded: {len(data.df)} rows", flush=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"siamese_v2_{ts}"
    out_dir = Path(args.output_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_modes = args.eval_modes if args.eval_modes else [args.eval_mode]

    all_results: dict[str, list[FoldResult]] = {}
    summaries:   dict[str, dict]             = {}

    for mode in eval_modes:
        df_mode = data.df.copy()

        if args.exclude_mixed_patients:
            before = df_mode["Patient ID"].nunique()
            df_mode, mixed = _drop_mixed_label_patients(
                df_mode, patient_col="Patient ID", label_col=args.label_col
            )
            print(
                f"[{mode}] dropped {len(mixed)} mixed-label patients "
                f"({before} → {df_mode['Patient ID'].nunique()})",
                flush=True,
            )
        else:
            if args.keep_mixed_raw_labels:
                print(f"[{mode}] keeping mixed-label patients with raw per-eye labels.", flush=True)
            else:
                before_rows = len(df_mode)
                df_mode, changed_rows, still_mixed = _relabel_mixed_patients_to_max(
                    df_mode, patient_col="Patient ID", label_col=args.label_col
                )
                print(
                    f"[{mode}] included mixed-label patients; relabeled to patient max severity "
                    f"(changed_rows={changed_rows}, rows={before_rows}->{len(df_mode)}, remaining_mixed={len(still_mixed)}).",
                    flush=True,
                )

        if mode == "binary":
            df_mode = df_mode[df_mode[args.label_col].isin([0, 1])].reset_index(drop=True)

        num_classes = 2 if mode == "binary" else int(df_mode[args.label_col].nunique())
        print(
            f"\n[{mode}] num_classes={num_classes}  rows={len(df_mode)}  "
            f"patients={df_mode['Patient ID'].nunique()}",
            flush=True,
        )

        split_manager = PatientFirstSplitManager(
            patient_col="Patient ID", label_col=args.label_col
        )
        split_args = SimpleNamespace(
            eval_mode=mode,
            holdout_per_class=0,   # No holdout by design
            holdout_seed=123,
            n_splits=args.n_splits,
            fold_seed=args.fold_seed,
        )
        clinical_ns = SimpleNamespace(df=df_mode, label_col=args.label_col)
        plans = split_manager.build_plans(clinical=clinical_ns, args=split_args, profile=None)
        n_folds = min(args.folds, len(plans))

        # Two profiles: eye-level for baseline training, patient-level for
        # siamese training and shared bilateral val evaluation.
        profile_eye     = build_papila_profile(
            patient_col="Patient ID", label_col=args.label_col, sample_mode="eye"
        )
        profile_patient = build_papila_profile(
            patient_col="Patient ID", label_col=args.label_col, sample_mode="patient"
        )

        mode_dir = out_dir / mode
        mode_dir.mkdir(exist_ok=True)

        fold_results: list[FoldResult] = []
        for fold in range(n_folds):
            fold_seed = args.seed + fold * 100
            seed_everything(fold_seed)
            fold_dir = mode_dir / f"fold{fold}"
            fold_dir.mkdir(exist_ok=True)

            print(f"\n[{mode}] fold {fold+1}/{n_folds}", flush=True)
            result = run_fold(
                fold=fold,
                split=plans[fold],
                mode=mode,
                args=args,
                device=device,
                data=data,
                num_classes=num_classes,
                profile_eye=profile_eye,
                profile_patient=profile_patient,
                fold_dir=fold_dir,
            )
            fold_results.append(result)

        # Write per-mode fold CSV
        fold_csv = out_dir / f"{mode}_fold_results.csv"
        csv_fields = list(FoldResult.__dataclass_fields__.keys())
        with fold_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=csv_fields)
            w.writeheader()
            for r in fold_results:
                w.writerow({k: getattr(r, k) for k in csv_fields})

        summary = _summary(fold_results)
        _print_summary(mode, summary)

        all_results[mode] = fold_results
        summaries[mode]   = summary

    payload = {
        "run_name":  run_name,
        "timestamp": ts,
        "config":    vars(args),
        "summaries": summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nOutputs written to: {out_dir}")


if __name__ == "__main__":
    main()
