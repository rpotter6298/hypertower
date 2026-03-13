#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classes.v2 import (
    Bridge,
    ImageTower,
    PatientFirstSplitManager,
    SlotDataset,
    build_papila_data,
    build_papila_profile,
    slot_collate,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class FoldMetrics:
    mode: str
    fold: int
    baseline_acc: float
    baseline_auc: float
    bilateral_acc: float
    bilateral_auc: float
    baseline_n: int
    bilateral_n: int
    os_acc: float
    os_auc: float
    os_n: int
    holdout_baseline_acc: float
    holdout_baseline_auc: float
    holdout_bilateral_acc: float
    holdout_bilateral_auc: float
    holdout_baseline_n: int
    holdout_bilateral_n: int
    holdout_os_acc: float
    holdout_os_auc: float
    holdout_os_n: int


class EyeLevelCNN(nn.Module):
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
        feats = self.tower(x)
        return self.head(feats)


class BilateralFusionCNN(nn.Module):
    def __init__(
        self,
        *,
        backbone: str,
        freeze_ratio: float,
        num_classes: int,
        augment: bool,
        use_se: bool,
        fusion_dim: int,
    ):
        super().__init__()
        self.tower_od = ImageTower(
            backbone=backbone,
            freeze_ratio=freeze_ratio,
            augment=augment,
            use_se=False,
        )
        self.tower_os = ImageTower(
            backbone=backbone,
            freeze_ratio=freeze_ratio,
            augment=augment,
            use_se=False,
        )
        self.bridge = Bridge(
            img_dim=self.tower_od.out_dim,
            meta_dim=self.tower_os.out_dim,
            num_classes=num_classes,
            fusion_dim=fusion_dim,
            mode="fused",
            use_se=use_se,
        )

    def forward(self, od: torch.Tensor, os: torch.Tensor) -> torch.Tensor:
        f_od = self.tower_od(od)
        f_os = self.tower_os(os)
        out_fused, _, _ = self.bridge(f_od, f_os)
        return out_fused


def patient_to_single_eye_samples(patient_samples: list[dict], eye_key: str) -> list[dict]:
    out = []
    for s in patient_samples:
        img = s.get(eye_key)
        lbl = s.get("label_1")
        if img is None or lbl is None:
            continue
        out.append({"id_1": s.get("id_1"), "image_1": img, "label_1": lbl})
    return out


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


def filter_eye_samples(samples: list[dict]) -> list[dict]:
    return [s for s in samples if s.get("image_1") is not None and s.get("label_1") is not None]


def filter_bilateral_samples(samples: list[dict]) -> list[dict]:
    return [
        s
        for s in samples
        if s.get("image_1") is not None and s.get("image_2") is not None and s.get("label_1") is not None
    ]


def to_label_tensor(labels, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(labels):
        return labels.to(device=device, dtype=torch.long)
    return torch.as_tensor(labels, dtype=torch.long, device=device)


def train_eye_epoch(
    model: EyeLevelCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    for batch in loader:
        x = batch.get("image_1")
        y = batch.get("label_1")
        if not torch.is_tensor(x):
            continue
        y = to_label_tensor(y, device)
        x = x.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bs = int(y.shape[0])
        total_loss += float(loss.item()) * bs
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_n += bs
    avg_loss = float(total_loss / total_n) if total_n > 0 else float("nan")
    acc = float(total_correct / total_n) if total_n > 0 else float("nan")
    return avg_loss, acc, total_n


def train_bilateral_epoch(
    model: BilateralFusionCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    for batch in loader:
        x1 = batch.get("image_1")
        x2 = batch.get("image_2")
        y = batch.get("label_1")
        if not torch.is_tensor(x1) or not torch.is_tensor(x2):
            continue
        y = to_label_tensor(y, device)
        x1 = x1.to(device)
        x2 = x2.to(device)
        logits = model(x1, x2)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bs = int(y.shape[0])
        total_loss += float(loss.item()) * bs
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_n += bs
    avg_loss = float(total_loss / total_n) if total_n > 0 else float("nan")
    acc = float(total_correct / total_n) if total_n > 0 else float("nan")
    return avg_loss, acc, total_n


def evaluate_eye(model: EyeLevelCNN, loader: DataLoader, device: torch.device, num_classes: int):
    model.eval()
    y_true = []
    y_prob = []
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for batch in loader:
            x = batch.get("image_1")
            y = batch.get("label_1")
            if not torch.is_tensor(x):
                continue
            y_t = to_label_tensor(y, device)
            logits = model(x.to(device))
            bs = int(y_t.shape[0])
            total_loss += float(F.cross_entropy(logits, y_t).item()) * bs
            total_n += bs
            probs = F.softmax(logits, dim=1).cpu().numpy()
            y_true.append(y_t.cpu().numpy())
            y_prob.append(probs)
    acc, auc, n = _score_arrays(y_true, y_prob, num_classes)
    avg_loss = float(total_loss / total_n) if total_n > 0 else float("nan")
    return avg_loss, acc, auc, n


def evaluate_bilateral(
    model: BilateralFusionCNN,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
):
    model.eval()
    y_true = []
    y_prob = []
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for batch in loader:
            x1 = batch.get("image_1")
            x2 = batch.get("image_2")
            y = batch.get("label_1")
            if not torch.is_tensor(x1) or not torch.is_tensor(x2):
                continue
            y_t = to_label_tensor(y, device)
            logits = model(x1.to(device), x2.to(device))
            bs = int(y_t.shape[0])
            total_loss += float(F.cross_entropy(logits, y_t).item()) * bs
            total_n += bs
            probs = F.softmax(logits, dim=1).cpu().numpy()
            y_true.append(y_t.cpu().numpy())
            y_prob.append(probs)
    acc, auc, n = _score_arrays(y_true, y_prob, num_classes)
    avg_loss = float(total_loss / total_n) if total_n > 0 else float("nan")
    return avg_loss, acc, auc, n


def evaluate_two_single_merge(
    model_od: EyeLevelCNN,
    model_os: EyeLevelCNN,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
):
    model_od.eval()
    model_os.eval()
    y_true = []
    y_prob = []
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for batch in loader:
            x1 = batch.get("image_1")
            x2 = batch.get("image_2")
            y = batch.get("label_1")
            if not torch.is_tensor(x1) or not torch.is_tensor(x2):
                continue
            y_t = to_label_tensor(y, device)
            p1 = F.softmax(model_od(x1.to(device)), dim=1)
            p2 = F.softmax(model_os(x2.to(device)), dim=1)
            p = 0.5 * (p1 + p2)
            bs = int(y_t.shape[0])
            total_loss += float(F.nll_loss(torch.log(p.clamp_min(1e-8)), y_t).item()) * bs
            total_n += bs
            y_true.append(y_t.cpu().numpy())
            y_prob.append(p.cpu().numpy())
    acc, auc, n = _score_arrays(y_true, y_prob, num_classes)
    avg_loss = float(total_loss / total_n) if total_n > 0 else float("nan")
    return avg_loss, acc, auc, n


def collect_binary_probs_eye(model: EyeLevelCNN, loader: DataLoader, device: torch.device):
    model.eval()
    y_true = []
    p1 = []
    with torch.no_grad():
        for batch in loader:
            x = batch.get("image_1")
            y = batch.get("label_1")
            if not torch.is_tensor(x):
                continue
            y_t = to_label_tensor(y, device)
            probs = F.softmax(model(x.to(device)), dim=1)[:, 1]
            y_true.append(y_t.cpu().numpy())
            p1.append(probs.cpu().numpy())
    if not y_true:
        return np.array([]), np.array([])
    return np.concatenate(y_true), np.concatenate(p1)


def collect_binary_probs_bilateral(model: BilateralFusionCNN, loader: DataLoader, device: torch.device):
    model.eval()
    y_true = []
    p1 = []
    with torch.no_grad():
        for batch in loader:
            x1 = batch.get("image_1")
            x2 = batch.get("image_2")
            y = batch.get("label_1")
            if not torch.is_tensor(x1) or not torch.is_tensor(x2):
                continue
            y_t = to_label_tensor(y, device)
            probs = F.softmax(model(x1.to(device), x2.to(device)), dim=1)[:, 1]
            y_true.append(y_t.cpu().numpy())
            p1.append(probs.cpu().numpy())
    if not y_true:
        return np.array([]), np.array([])
    return np.concatenate(y_true), np.concatenate(p1)


def collect_binary_probs_merge(model_od: EyeLevelCNN, model_os: EyeLevelCNN, loader: DataLoader, device: torch.device):
    model_od.eval()
    model_os.eval()
    y_true = []
    p1 = []
    with torch.no_grad():
        for batch in loader:
            x1 = batch.get("image_1")
            x2 = batch.get("image_2")
            y = batch.get("label_1")
            if not torch.is_tensor(x1) or not torch.is_tensor(x2):
                continue
            y_t = to_label_tensor(y, device)
            p_od = F.softmax(model_od(x1.to(device)), dim=1)[:, 1]
            p_os = F.softmax(model_os(x2.to(device)), dim=1)[:, 1]
            p = 0.5 * (p_od + p_os)
            y_true.append(y_t.cpu().numpy())
            p1.append(p.cpu().numpy())
    if not y_true:
        return np.array([]), np.array([])
    return np.concatenate(y_true), np.concatenate(p1)


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


def binary_acc_at_threshold(y_true: np.ndarray, p1: np.ndarray, t: float) -> float:
    if y_true.size == 0:
        return float("nan")
    pred = (p1 >= t).astype(int)
    return float((pred == y_true).mean())


def collect_probs_eye(model: EyeLevelCNN, loader: DataLoader, device: torch.device):
    model.eval()
    y_true = []
    probs_all = []
    with torch.no_grad():
        for batch in loader:
            x = batch.get("image_1")
            y = batch.get("label_1")
            if not torch.is_tensor(x):
                continue
            y_t = to_label_tensor(y, device)
            probs = F.softmax(model(x.to(device)), dim=1).cpu().numpy()
            y_true.append(y_t.cpu().numpy())
            probs_all.append(probs)
    if not y_true:
        return np.array([]), np.zeros((0, 0), dtype=float)
    return np.concatenate(y_true), np.concatenate(probs_all, axis=0)


def collect_probs_bilateral(model: BilateralFusionCNN, loader: DataLoader, device: torch.device):
    model.eval()
    y_true = []
    probs_all = []
    with torch.no_grad():
        for batch in loader:
            x1 = batch.get("image_1")
            x2 = batch.get("image_2")
            y = batch.get("label_1")
            if not torch.is_tensor(x1) or not torch.is_tensor(x2):
                continue
            y_t = to_label_tensor(y, device)
            probs = F.softmax(model(x1.to(device), x2.to(device)), dim=1).cpu().numpy()
            y_true.append(y_t.cpu().numpy())
            probs_all.append(probs)
    if not y_true:
        return np.array([]), np.zeros((0, 0), dtype=float)
    return np.concatenate(y_true), np.concatenate(probs_all, axis=0)


def collect_probs_merge(model_od: EyeLevelCNN, model_os: EyeLevelCNN, loader: DataLoader, device: torch.device):
    model_od.eval()
    model_os.eval()
    y_true = []
    probs_all = []
    with torch.no_grad():
        for batch in loader:
            x1 = batch.get("image_1")
            x2 = batch.get("image_2")
            y = batch.get("label_1")
            if not torch.is_tensor(x1) or not torch.is_tensor(x2):
                continue
            y_t = to_label_tensor(y, device)
            p_od = F.softmax(model_od(x1.to(device)), dim=1)
            p_os = F.softmax(model_os(x2.to(device)), dim=1)
            probs = (0.5 * (p_od + p_os)).cpu().numpy()
            y_true.append(y_t.cpu().numpy())
            probs_all.append(probs)
    if not y_true:
        return np.array([]), np.zeros((0, 0), dtype=float)
    return np.concatenate(y_true), np.concatenate(probs_all, axis=0)


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
            # small stabilization around baseline
            if np.isnan(best_acc):
                bias[k] = old
    return bias


def _serialize_vec(vec: np.ndarray | None) -> str | None:
    if vec is None:
        return None
    if vec.size == 0:
        return None
    return "|".join(f"{float(v):.4f}" for v in vec.tolist())


def _score_arrays(y_true_chunks, y_prob_chunks, num_classes: int):
    if not y_true_chunks:
        return float("nan"), float("nan"), 0
    y = np.concatenate(y_true_chunks, axis=0)
    p = np.concatenate(y_prob_chunks, axis=0)
    acc = float((p.argmax(axis=1) == y).mean())
    try:
        if num_classes == 2:
            auc = float(roc_auc_score(y, p[:, 1]))
        else:
            auc = float(roc_auc_score(y, p, multi_class="ovr", average="macro"))
    except Exception:
        auc = float("nan")
    return acc, auc, int(y.shape[0])


def _drop_mixed_label_patients(df: pd.DataFrame, *, patient_col: str, label_col: str):
    per_patient = (
        df.groupby(patient_col)[label_col]
        .agg(lambda s: set(pd.to_numeric(s, errors="coerce").dropna().astype(int).tolist()))
    )
    mixed_ids = [pid for pid, labels in per_patient.items() if len(labels) > 1]
    if not mixed_ids:
        return df, []
    keep = ~df[patient_col].isin(mixed_ids)
    return df[keep].reset_index(drop=True), mixed_ids


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is not available.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Compare eye-level single-tower CNN vs patient-level bilateral dual-tower fusion model."
    )
    ap.add_argument("--image-dir", default="Papila/FundusImages")
    ap.add_argument("--clinical-dir", default="Papila/ClinicalData")
    ap.add_argument("--label-col", default="Diagnosis")
    ap.add_argument("--cat-cols", nargs="*", default=["Gender", "Phakic/Pseudophakic"])
    ap.add_argument("--eval-mode", choices=["multiclass", "binary"], default="binary")
    ap.add_argument(
        "--eval-modes",
        nargs="+",
        choices=["multiclass", "binary"],
        default=None,
        help="Optional list of modes to run in one pass (e.g. --eval-modes binary multiclass).",
    )
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--fold-seed", type=int, default=42)
    ap.add_argument("--holdout-per-class", type=int, default=0)
    ap.add_argument("--holdout-seed", type=int, default=123)
    ap.add_argument("--folds", type=int, default=5, help="How many folds to run (<= n-splits).")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--freeze-ratio", type=float, default=0.0)
    ap.add_argument("--augment", action="store_true", help="Enable image augmentation during training.")
    ap.add_argument("--fusion-dim", type=int, default=256)
    ap.add_argument("--bridge-se", action="store_true", help="Enable SE in bilateral bridge.")
    ap.add_argument(
        "--bilateral-method",
        choices=["bridge", "two-single-merge"],
        default="bridge",
        help="Bilateral comparator: learned dual-tower bridge or two separate single-eye models merged by prob average.",
    )
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--output-root", default="analysis_data/basic_analysis")
    ap.add_argument(
        "--exclude-binary-mixed-patients",
        action="store_true",
        help="Drop patients with mixed eye labels (any disagreement across eyes) before splitting.",
    )
    ap.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="If > 0, print epoch progress every N epochs within each fold.",
    )
    ap.add_argument(
        "--tune-binary-threshold",
        action="store_true",
        help="In binary mode, tune decision thresholds on validation probs and apply them to val/holdout accuracy.",
    )
    ap.add_argument(
        "--tune-multiclass-bias",
        action="store_true",
        help="In multiclass mode, tune per-class log-prob biases on validation and apply to val/holdout accuracy.",
    )
    return ap.parse_args()


def _serialize_float(v: float) -> Optional[float]:
    return None if np.isnan(v) else float(v)


def _rows_from_metrics(metrics: list[FoldMetrics]) -> list[dict]:
    rows = []
    for m in metrics:
        rows.append(
            {
                "mode": m.mode,
                "fold": m.fold,
                "baseline_acc": _serialize_float(m.baseline_acc),
                "baseline_auc": _serialize_float(m.baseline_auc),
                "bilateral_acc": _serialize_float(m.bilateral_acc),
                "bilateral_auc": _serialize_float(m.bilateral_auc),
                "baseline_n": m.baseline_n,
                "bilateral_n": m.bilateral_n,
                "os_acc": _serialize_float(m.os_acc),
                "os_auc": _serialize_float(m.os_auc),
                "os_n": m.os_n,
                "holdout_baseline_acc": _serialize_float(m.holdout_baseline_acc),
                "holdout_baseline_auc": _serialize_float(m.holdout_baseline_auc),
                "holdout_bilateral_acc": _serialize_float(m.holdout_bilateral_acc),
                "holdout_bilateral_auc": _serialize_float(m.holdout_bilateral_auc),
                "holdout_baseline_n": m.holdout_baseline_n,
                "holdout_bilateral_n": m.holdout_bilateral_n,
                "holdout_os_acc": _serialize_float(m.holdout_os_acc),
                "holdout_os_auc": _serialize_float(m.holdout_os_auc),
                "holdout_os_n": m.holdout_os_n,
            }
        )
    return rows


def _summary_for_mode(mode: str, metrics: list[FoldMetrics]) -> dict:
    b_accs = np.array([m.baseline_acc for m in metrics], dtype=float)
    b_aucs = np.array([m.baseline_auc for m in metrics], dtype=float)
    d_accs = np.array([m.bilateral_acc for m in metrics], dtype=float)
    d_aucs = np.array([m.bilateral_auc for m in metrics], dtype=float)
    hb_accs = np.array([m.holdout_baseline_acc for m in metrics], dtype=float)
    hb_aucs = np.array([m.holdout_baseline_auc for m in metrics], dtype=float)
    hd_accs = np.array([m.holdout_bilateral_acc for m in metrics], dtype=float)
    hd_aucs = np.array([m.holdout_bilateral_auc for m in metrics], dtype=float)

    return {
        "mode": mode,
        "baseline": {
            "acc_mean": _serialize_float(float(np.nanmean(b_accs))),
            "acc_std": _serialize_float(float(np.nanstd(b_accs))),
            "auc_mean": _serialize_float(float(np.nanmean(b_aucs))),
            "auc_std": _serialize_float(float(np.nanstd(b_aucs))),
        },
        "bilateral": {
            "acc_mean": _serialize_float(float(np.nanmean(d_accs))),
            "acc_std": _serialize_float(float(np.nanstd(d_accs))),
            "auc_mean": _serialize_float(float(np.nanmean(d_aucs))),
            "auc_std": _serialize_float(float(np.nanstd(d_aucs))),
        },
        "delta_bilateral_minus_baseline": {
            "acc_mean": _serialize_float(float(np.nanmean(d_accs - b_accs))),
            "auc_mean": _serialize_float(float(np.nanmean(d_aucs - b_aucs))),
        },
        "holdout_baseline": {
            "acc_mean": _serialize_float(float(np.nanmean(hb_accs))),
            "acc_std": _serialize_float(float(np.nanstd(hb_accs))),
            "auc_mean": _serialize_float(float(np.nanmean(hb_aucs))),
            "auc_std": _serialize_float(float(np.nanstd(hb_aucs))),
        },
        "holdout_bilateral": {
            "acc_mean": _serialize_float(float(np.nanmean(hd_accs))),
            "acc_std": _serialize_float(float(np.nanstd(hd_accs))),
            "auc_mean": _serialize_float(float(np.nanmean(hd_aucs))),
            "auc_std": _serialize_float(float(np.nanstd(hd_aucs))),
        },
        "holdout_delta_bilateral_minus_baseline": {
            "acc_mean": _serialize_float(float(np.nanmean(hd_accs - hb_accs))),
            "auc_mean": _serialize_float(float(np.nanmean(hd_aucs - hb_aucs))),
        },
    }


def _print_summary(mode: str, summary: dict) -> None:
    def fmt(v):
        return "nan" if v is None else f"{v:.4f}"

    print(f"\n=== Summary ({mode}) ===")
    print(
        "baseline_eye_cnn   "
        f"acc={fmt(summary['baseline']['acc_mean'])}±{fmt(summary['baseline']['acc_std'])} "
        f"auc={fmt(summary['baseline']['auc_mean'])}±{fmt(summary['baseline']['auc_std'])}"
    )
    print(
        "bilateral_dual_img "
        f"acc={fmt(summary['bilateral']['acc_mean'])}±{fmt(summary['bilateral']['acc_std'])} "
        f"auc={fmt(summary['bilateral']['auc_mean'])}±{fmt(summary['bilateral']['auc_std'])}"
    )
    print(
        "delta(bilateral-baseline) "
        f"acc={fmt(summary['delta_bilateral_minus_baseline']['acc_mean'])} "
        f"auc={fmt(summary['delta_bilateral_minus_baseline']['auc_mean'])}"
    )
    if summary["holdout_baseline"]["acc_mean"] is not None:
        print(
            "holdout baseline_eye_cnn   "
            f"acc={fmt(summary['holdout_baseline']['acc_mean'])}±{fmt(summary['holdout_baseline']['acc_std'])} "
            f"auc={fmt(summary['holdout_baseline']['auc_mean'])}±{fmt(summary['holdout_baseline']['auc_std'])}"
        )
        print(
            "holdout bilateral_dual_img "
            f"acc={fmt(summary['holdout_bilateral']['acc_mean'])}±{fmt(summary['holdout_bilateral']['acc_std'])} "
            f"auc={fmt(summary['holdout_bilateral']['auc_mean'])}±{fmt(summary['holdout_bilateral']['auc_std'])}"
        )
        print(
            "holdout delta(bilateral-baseline) "
            f"acc={fmt(summary['holdout_delta_bilateral_minus_baseline']['acc_mean'])} "
            f"auc={fmt(summary['holdout_delta_bilateral_minus_baseline']['auc_mean'])}"
        )


def run_mode(args, mode: str, device: torch.device, data, out_dir: Path) -> tuple[list[FoldMetrics], dict]:
    df_mode = data.df.copy()
    if args.exclude_binary_mixed_patients:
        before_rows = len(df_mode)
        before_patients = int(df_mode["Patient ID"].nunique())
        df_mode, mixed_ids = _drop_mixed_label_patients(
            df_mode, patient_col="Patient ID", label_col=args.label_col
        )
        print(
            f"[mode={mode}] excluded {len(mixed_ids)} mixed-label patients "
            f"(rows {before_rows}->{len(df_mode)}, patients {before_patients}->{df_mode['Patient ID'].nunique()})",
            flush=True,
        )

    if mode == "binary":
        df_mode = df_mode[df_mode[args.label_col].isin([0, 1])].reset_index(drop=True)

    num_classes = 2 if mode == "binary" else int(df_mode[args.label_col].nunique())
    print(
        f"\n[mode={mode}] preparing splits (num_classes={num_classes}, rows={len(df_mode)}, patients={df_mode['Patient ID'].nunique()})...",
        flush=True,
    )

    split_manager = PatientFirstSplitManager(patient_col="Patient ID", label_col=args.label_col)
    split_args = SimpleNamespace(
        eval_mode=mode,
        holdout_per_class=args.holdout_per_class,
        holdout_seed=args.holdout_seed,
        n_splits=args.n_splits,
        fold_seed=args.fold_seed,
    )
    clinical_for_split = SimpleNamespace(df=df_mode, label_col=args.label_col)
    plans = split_manager.build_plans(clinical=clinical_for_split, args=split_args, profile=None)
    n_folds = min(args.folds, len(plans))

    profile_eye = build_papila_profile(patient_col="Patient ID", label_col=args.label_col, sample_mode="eye")
    profile_patient = build_papila_profile(patient_col="Patient ID", label_col=args.label_col, sample_mode="patient")

    fold_metrics: list[FoldMetrics] = []
    mode_dir = out_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    for fold in range(n_folds):
        split = plans[fold]
        holdout_df = split.holdout
        fold_seed = args.seed + fold * 100
        seed_everything(fold_seed)
        print(
            f"[mode={mode}] fold {fold+1}/{n_folds}: building models/loaders...",
            flush=True,
        )
        fold_dir = mode_dir / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_log_path = fold_dir / "train.log"
        epoch_log_path = fold_dir / "epoch_log.csv"

        baseline_model = None
        bilateral_model = None
        model_od = None
        model_os = None
        if args.bilateral_method == "bridge":
            baseline_model = EyeLevelCNN(
                backbone=args.backbone,
                freeze_ratio=args.freeze_ratio,
                num_classes=num_classes,
                augment=args.augment,
            ).to(device)
            bilateral_model = BilateralFusionCNN(
                backbone=args.backbone,
                freeze_ratio=args.freeze_ratio,
                num_classes=num_classes,
                augment=args.augment,
                use_se=args.bridge_se,
                fusion_dim=args.fusion_dim,
            ).to(device)
        else:
            model_od = EyeLevelCNN(
                backbone=args.backbone,
                freeze_ratio=args.freeze_ratio,
                num_classes=num_classes,
                augment=args.augment,
            ).to(device)
            model_os = EyeLevelCNN(
                backbone=args.backbone,
                freeze_ratio=args.freeze_ratio,
                num_classes=num_classes,
                augment=args.augment,
            ).to(device)
            baseline_model = model_od

        eye_train_samples = filter_eye_samples(profile_eye.build_samples(df=split.train, clinical=data))
        eye_val_samples = filter_eye_samples(profile_eye.build_samples(df=split.val, clinical=data))
        bilat_train_samples = filter_bilateral_samples(profile_patient.build_samples(df=split.train, clinical=data))
        bilat_val_samples = filter_bilateral_samples(profile_patient.build_samples(df=split.val, clinical=data))
        od_train_samples = patient_to_single_eye_samples(bilat_train_samples, "image_1")
        od_val_samples = patient_to_single_eye_samples(bilat_val_samples, "image_1")
        os_train_samples = patient_to_single_eye_samples(bilat_train_samples, "image_2")
        os_val_samples = patient_to_single_eye_samples(bilat_val_samples, "image_2")

        eye_holdout_samples = []
        bilat_holdout_samples = []
        od_holdout_samples = []
        os_holdout_samples = []
        if holdout_df is not None and not holdout_df.empty:
            eye_holdout_samples = filter_eye_samples(profile_eye.build_samples(df=holdout_df, clinical=data))
            bilat_holdout_samples = filter_bilateral_samples(
                profile_patient.build_samples(df=holdout_df, clinical=data)
            )
            od_holdout_samples = patient_to_single_eye_samples(bilat_holdout_samples, "image_1")
            os_holdout_samples = patient_to_single_eye_samples(bilat_holdout_samples, "image_2")
        baseline_train = None
        baseline_val = None
        if args.bilateral_method == "bridge":
            baseline_train = make_loader(
                eye_train_samples,
                profile_eye.slot_descriptors(),
                image_transform=baseline_model.tower.transform,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
            )
            baseline_val = make_loader(
                eye_val_samples,
                profile_eye.slot_descriptors(),
                image_transform=baseline_model.tower.transform,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
        else:
            baseline_train = make_loader(
                od_train_samples,
                profile_eye.slot_descriptors(),
                image_transform=model_od.tower.transform,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
            )
            baseline_val = make_loader(
                od_val_samples,
                profile_eye.slot_descriptors(),
                image_transform=model_od.tower.transform,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
        bilateral_train = make_loader(
            bilat_train_samples,
            profile_patient.slot_descriptors(),
            image_transform=(bilateral_model.tower_od.transform if bilateral_model is not None else model_od.tower.transform),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        bilateral_val = make_loader(
            bilat_val_samples,
            profile_patient.slot_descriptors(),
            image_transform=(bilateral_model.tower_od.transform if bilateral_model is not None else model_od.tower.transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        od_train = None
        od_val = None
        os_train = None
        os_val = None
        if args.bilateral_method == "two-single-merge":
            od_train = make_loader(
                od_train_samples,
                profile_eye.slot_descriptors(),
                image_transform=model_od.tower.transform,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
            )
            os_train = make_loader(
                os_train_samples,
                profile_eye.slot_descriptors(),
                image_transform=model_os.tower.transform,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
            )
            od_val = make_loader(
                od_val_samples,
                profile_eye.slot_descriptors(),
                image_transform=model_od.tower.transform,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
            os_val = make_loader(
                os_val_samples,
                profile_eye.slot_descriptors(),
                image_transform=model_os.tower.transform,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )

        baseline_holdout = None
        bilateral_holdout = None
        os_holdout = None
        if eye_holdout_samples:
            if args.bilateral_method == "bridge":
                baseline_holdout = make_loader(
                    eye_holdout_samples,
                    profile_eye.slot_descriptors(),
                    image_transform=baseline_model.tower.transform,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                )
            else:
                baseline_holdout = make_loader(
                    od_holdout_samples,
                    profile_eye.slot_descriptors(),
                    image_transform=model_od.tower.transform,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                )
                os_holdout = make_loader(
                    os_holdout_samples,
                    profile_eye.slot_descriptors(),
                    image_transform=model_os.tower.transform,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                )
        if bilat_holdout_samples:
            bilateral_holdout = make_loader(
                bilat_holdout_samples,
                profile_patient.slot_descriptors(),
                image_transform=(bilateral_model.tower_od.transform if bilateral_model is not None else model_od.tower.transform),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )

        opt_base = None
        opt_bilat = None
        opt_od = None
        opt_os = None
        if args.bilateral_method == "bridge":
            opt_base = torch.optim.Adam(baseline_model.parameters(), lr=args.lr)
            opt_bilat = torch.optim.Adam(bilateral_model.parameters(), lr=args.lr)
        else:
            opt_od = torch.optim.Adam(model_od.parameters(), lr=args.lr)
            opt_os = torch.optim.Adam(model_os.parameters(), lr=args.lr)
        epoch_fields = [
            "mode",
            "fold",
            "epoch",
            "baseline_train_loss",
            "baseline_train_acc",
            "baseline_train_n",
            "baseline_val_loss",
            "baseline_val_acc",
            "baseline_val_auc",
            "baseline_val_n",
            "baseline_threshold",
            "baseline_bias",
            "bilateral_train_loss",
            "bilateral_train_acc",
            "bilateral_train_n",
            "bilateral_val_loss",
            "bilateral_val_acc",
            "bilateral_val_auc",
            "bilateral_val_n",
            "bilateral_threshold",
            "bilateral_bias",
            "os_val_loss",
            "os_val_acc",
            "os_val_auc",
            "os_val_n",
            "os_threshold",
            "os_bias",
            "holdout_baseline_loss",
            "holdout_baseline_acc",
            "holdout_baseline_auc",
            "holdout_baseline_n",
            "holdout_bilateral_loss",
            "holdout_bilateral_acc",
            "holdout_bilateral_auc",
            "holdout_bilateral_n",
            "holdout_os_loss",
            "holdout_os_acc",
            "holdout_os_auc",
            "holdout_os_n",
        ]
        epoch_fp = epoch_log_path.open("w", newline="", encoding="utf-8")
        epoch_writer = csv.DictWriter(epoch_fp, fieldnames=epoch_fields)
        epoch_writer.writeheader()

        print(
            f"[mode={mode}] fold {fold+1}/{n_folds}: training "
            f"(epochs={args.epochs}, baseline_train_n={len(eye_train_samples)}, "
            f"bilateral_train_n={len(bilat_train_samples)}, method={args.bilateral_method})",
            flush=True,
        )
        b_loss = b_acc = b_auc = float("nan")
        b_n = 0
        b_thr = 0.5
        b_bias = None
        os_loss = os_acc = os_auc = float("nan")
        os_n = 0
        os_thr = 0.5
        os_bias = None
        d_loss = d_acc = d_auc = float("nan")
        d_n = 0
        d_thr = 0.5
        d_bias = None
        hb_loss = hb_acc = hb_auc = float("nan")
        hb_n = 0
        hos_loss = hos_acc = hos_auc = float("nan")
        hos_n = 0
        hd_loss = hd_acc = hd_auc = float("nan")
        hd_n = 0
        with train_log_path.open("w", encoding="utf-8") as train_log:
            for epoch in range(args.epochs):
                bt_loss = bt_acc = float("nan")
                bt_n = 0
                ot_loss = ot_acc = float("nan")
                ot_n = 0
                if args.bilateral_method == "bridge":
                    bt_loss, bt_acc, bt_n = train_eye_epoch(baseline_model, baseline_train, opt_base, device)
                else:
                    bt_loss, bt_acc, bt_n = train_eye_epoch(model_od, od_train, opt_od, device)
                    ot_loss, ot_acc, ot_n = train_eye_epoch(model_os, os_train, opt_os, device)
                if args.bilateral_method == "bridge":
                    dt_loss, dt_acc, dt_n = train_bilateral_epoch(bilateral_model, bilateral_train, opt_bilat, device)
                else:
                    dt_loss = float(np.nanmean([bt_loss, ot_loss]))
                    dt_acc = float(np.nanmean([bt_acc, ot_acc]))
                    dt_n = int(min(bt_n, ot_n))
                b_loss, b_acc, b_auc, b_n = evaluate_eye(baseline_model, baseline_val, device, num_classes)
                if args.bilateral_method == "two-single-merge":
                    os_loss, os_acc, os_auc, os_n = evaluate_eye(model_os, os_val, device, num_classes)
                if args.bilateral_method == "bridge":
                    d_loss, d_acc, d_auc, d_n = evaluate_bilateral(bilateral_model, bilateral_val, device, num_classes)
                else:
                    d_loss, d_acc, d_auc, d_n = evaluate_two_single_merge(
                        model_od, model_os, bilateral_val, device, num_classes
                    )

                if args.tune_binary_threshold and num_classes == 2:
                    yb, pb = collect_binary_probs_eye(baseline_model, baseline_val, device)
                    b_thr = tune_binary_threshold(yb, pb)
                    b_acc = binary_acc_at_threshold(yb, pb, b_thr)
                    if args.bilateral_method == "bridge":
                        yd, pd = collect_binary_probs_bilateral(bilateral_model, bilateral_val, device)
                    else:
                        yd, pd = collect_binary_probs_merge(model_od, model_os, bilateral_val, device)
                    d_thr = tune_binary_threshold(yd, pd)
                    d_acc = binary_acc_at_threshold(yd, pd, d_thr)
                    if args.bilateral_method == "two-single-merge":
                        yo, po = collect_binary_probs_eye(model_os, os_val, device)
                        os_thr = tune_binary_threshold(yo, po)
                        os_acc = binary_acc_at_threshold(yo, po, os_thr)
                elif args.tune_multiclass_bias and num_classes > 2:
                    yb, pb = collect_probs_eye(baseline_model, baseline_val, device)
                    b_bias = tune_multiclass_bias(yb, pb)
                    b_acc = multiclass_acc_with_bias(yb, pb, b_bias)
                    if args.bilateral_method == "bridge":
                        yd, pd = collect_probs_bilateral(bilateral_model, bilateral_val, device)
                    else:
                        yd, pd = collect_probs_merge(model_od, model_os, bilateral_val, device)
                    d_bias = tune_multiclass_bias(yd, pd)
                    d_acc = multiclass_acc_with_bias(yd, pd, d_bias)
                    if args.bilateral_method == "two-single-merge":
                        yo, po = collect_probs_eye(model_os, os_val, device)
                        os_bias = tune_multiclass_bias(yo, po)
                        os_acc = multiclass_acc_with_bias(yo, po, os_bias)

                hb_loss = hb_acc = hb_auc = float("nan")
                hb_n = 0
                hos_loss = hos_acc = hos_auc = float("nan")
                hos_n = 0
                hd_loss = hd_acc = hd_auc = float("nan")
                hd_n = 0
                if baseline_holdout is not None:
                    hb_loss, hb_acc, hb_auc, hb_n = evaluate_eye(baseline_model, baseline_holdout, device, num_classes)
                if os_holdout is not None:
                    hos_loss, hos_acc, hos_auc, hos_n = evaluate_eye(model_os, os_holdout, device, num_classes)
                if bilateral_holdout is not None:
                    if args.bilateral_method == "bridge":
                        hd_loss, hd_acc, hd_auc, hd_n = evaluate_bilateral(
                            bilateral_model, bilateral_holdout, device, num_classes
                        )
                    else:
                        hd_loss, hd_acc, hd_auc, hd_n = evaluate_two_single_merge(
                            model_od, model_os, bilateral_holdout, device, num_classes
                        )
                if args.tune_binary_threshold and num_classes == 2:
                    if baseline_holdout is not None:
                        yhb, phb = collect_binary_probs_eye(baseline_model, baseline_holdout, device)
                        hb_acc = binary_acc_at_threshold(yhb, phb, b_thr)
                    if os_holdout is not None:
                        yho, pho = collect_binary_probs_eye(model_os, os_holdout, device)
                        hos_acc = binary_acc_at_threshold(yho, pho, os_thr)
                    if bilateral_holdout is not None:
                        if args.bilateral_method == "bridge":
                            yhd, phd = collect_binary_probs_bilateral(bilateral_model, bilateral_holdout, device)
                        else:
                            yhd, phd = collect_binary_probs_merge(model_od, model_os, bilateral_holdout, device)
                        hd_acc = binary_acc_at_threshold(yhd, phd, d_thr)
                elif args.tune_multiclass_bias and num_classes > 2:
                    if baseline_holdout is not None and b_bias is not None:
                        yhb, phb = collect_probs_eye(baseline_model, baseline_holdout, device)
                        hb_acc = multiclass_acc_with_bias(yhb, phb, b_bias)
                    if os_holdout is not None and os_bias is not None:
                        yho, pho = collect_probs_eye(model_os, os_holdout, device)
                        hos_acc = multiclass_acc_with_bias(yho, pho, os_bias)
                    if bilateral_holdout is not None and d_bias is not None:
                        if args.bilateral_method == "bridge":
                            yhd, phd = collect_probs_bilateral(bilateral_model, bilateral_holdout, device)
                        else:
                            yhd, phd = collect_probs_merge(model_od, model_os, bilateral_holdout, device)
                        hd_acc = multiclass_acc_with_bias(yhd, phd, d_bias)

                row = {
                    "mode": mode,
                    "fold": fold,
                    "epoch": epoch + 1,
                    "baseline_train_loss": _serialize_float(bt_loss),
                    "baseline_train_acc": _serialize_float(bt_acc),
                    "baseline_train_n": bt_n,
                    "baseline_val_loss": _serialize_float(b_loss),
                    "baseline_val_acc": _serialize_float(b_acc),
                    "baseline_val_auc": _serialize_float(b_auc),
                    "baseline_val_n": b_n,
                    "baseline_threshold": _serialize_float(b_thr if num_classes == 2 else float("nan")),
                    "baseline_bias": _serialize_vec(b_bias if num_classes > 2 else None),
                    "bilateral_train_loss": _serialize_float(dt_loss),
                    "bilateral_train_acc": _serialize_float(dt_acc),
                    "bilateral_train_n": dt_n,
                    "bilateral_val_loss": _serialize_float(d_loss),
                    "bilateral_val_acc": _serialize_float(d_acc),
                    "bilateral_val_auc": _serialize_float(d_auc),
                    "bilateral_val_n": d_n,
                    "bilateral_threshold": _serialize_float(d_thr if num_classes == 2 else float("nan")),
                    "bilateral_bias": _serialize_vec(d_bias if num_classes > 2 else None),
                    "os_val_loss": _serialize_float(os_loss),
                    "os_val_acc": _serialize_float(os_acc),
                    "os_val_auc": _serialize_float(os_auc),
                    "os_val_n": os_n,
                    "os_threshold": _serialize_float(os_thr if (num_classes == 2 and args.bilateral_method == "two-single-merge") else float("nan")),
                    "os_bias": _serialize_vec(os_bias if (num_classes > 2 and args.bilateral_method == "two-single-merge") else None),
                    "holdout_baseline_loss": _serialize_float(hb_loss),
                    "holdout_baseline_acc": _serialize_float(hb_acc),
                    "holdout_baseline_auc": _serialize_float(hb_auc),
                    "holdout_baseline_n": hb_n,
                    "holdout_bilateral_loss": _serialize_float(hd_loss),
                    "holdout_bilateral_acc": _serialize_float(hd_acc),
                    "holdout_bilateral_auc": _serialize_float(hd_auc),
                    "holdout_bilateral_n": hd_n,
                    "holdout_os_loss": _serialize_float(hos_loss),
                    "holdout_os_acc": _serialize_float(hos_acc),
                    "holdout_os_auc": _serialize_float(hos_auc),
                    "holdout_os_n": hos_n,
                }
                epoch_writer.writerow(row)
                epoch_fp.flush()

                line_train = (
                    f"[mode={mode} fold={fold+1}/{n_folds} epoch={epoch+1}/{args.epochs}] "
                    f"train baseline(loss={bt_loss:.4f}, acc={bt_acc:.4f}, n={bt_n}) "
                    f"bilateral(loss={dt_loss:.4f}, acc={dt_acc:.4f}, n={dt_n})"
                )
                line_val = (
                    f"[mode={mode} fold={fold+1}/{n_folds} epoch={epoch+1}/{args.epochs}] "
                    f"val baseline(loss={b_loss:.4f}, acc={b_acc:.4f}, auc={b_auc:.4f}, n={b_n}) "
                    f"bilateral(loss={d_loss:.4f}, acc={d_acc:.4f}, auc={d_auc:.4f}, n={d_n})"
                )
                if args.bilateral_method == "two-single-merge":
                    line_val += f" os(loss={os_loss:.4f}, acc={os_acc:.4f}, auc={os_auc:.4f}, n={os_n})"
                print(line_train, flush=True)
                print(line_val, flush=True)
                train_log.write(line_train + "\n")
                train_log.write(line_val + "\n")
                if hb_n > 0 or hd_n > 0:
                    line_holdout = (
                        f"[mode={mode} fold={fold+1}/{n_folds} epoch={epoch+1}/{args.epochs}] "
                        f"holdout baseline(loss={hb_loss:.4f}, acc={hb_acc:.4f}, auc={hb_auc:.4f}, n={hb_n}) "
                        f"bilateral(loss={hd_loss:.4f}, acc={hd_acc:.4f}, auc={hd_auc:.4f}, n={hd_n})"
                    )
                    if args.bilateral_method == "two-single-merge":
                        line_holdout += (
                            f" os(loss={hos_loss:.4f}, acc={hos_acc:.4f}, auc={hos_auc:.4f}, n={hos_n})"
                        )
                    print(line_holdout, flush=True)
                    train_log.write(line_holdout + "\n")
                if args.log_every > 0 and ((epoch + 1) % args.log_every == 0 or (epoch + 1) == args.epochs):
                    print(
                        f"[mode={mode}] fold {fold+1}/{n_folds}: epoch {epoch+1}/{args.epochs} checkpoint",
                        flush=True,
                    )
        epoch_fp.close()

        fold_metrics.append(
            FoldMetrics(
                mode=mode,
                fold=fold,
                baseline_acc=b_acc,
                baseline_auc=b_auc,
                bilateral_acc=d_acc,
                bilateral_auc=d_auc,
                baseline_n=b_n,
                bilateral_n=d_n,
                os_acc=os_acc,
                os_auc=os_auc,
                os_n=os_n,
                holdout_baseline_acc=hb_acc,
                holdout_baseline_auc=hb_auc,
                holdout_bilateral_acc=hd_acc,
                holdout_bilateral_auc=hd_auc,
                holdout_baseline_n=hb_n,
                holdout_bilateral_n=hd_n,
                holdout_os_acc=hos_acc,
                holdout_os_auc=hos_auc,
                holdout_os_n=hos_n,
            )
        )
        msg = (
            f"mode={mode} fold={fold} baseline(val loss={b_loss:.4f}, acc={b_acc:.4f}, auc={b_auc:.4f}, n={b_n}) "
            f"bilateral(val loss={d_loss:.4f}, acc={d_acc:.4f}, auc={d_auc:.4f}, n={d_n})"
        )
        if args.bilateral_method == "two-single-merge":
            msg += f" | os(val loss={os_loss:.4f}, acc={os_acc:.4f}, auc={os_auc:.4f}, n={os_n})"
        if hb_n > 0 or hd_n > 0:
            msg += (
                f" | holdout baseline(loss={hb_loss:.4f}, acc={hb_acc:.4f}, auc={hb_auc:.4f}, n={hb_n}) "
                f"bilateral(loss={hd_loss:.4f}, acc={hd_acc:.4f}, auc={hd_auc:.4f}, n={hd_n})"
            )
            if args.bilateral_method == "two-single-merge":
                msg += (
                    f" os(loss={hos_loss:.4f}, acc={hos_acc:.4f}, auc={hos_auc:.4f}, n={hos_n})"
                )
        print(msg)

    summary = _summary_for_mode(mode, fold_metrics)
    _print_summary(mode, summary)
    return fold_metrics, summary


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
    print(f"Loaded PAPILA data: rows={len(data.df)}", flush=True)

    eval_modes = args.eval_modes if args.eval_modes else [args.eval_mode]
    print(f"Eval modes: {eval_modes}", flush=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"dual_eye_compare_{ts}"
    out_dir = Path(args.output_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summaries = {}
    for mode in eval_modes:
        fold_metrics, summary = run_mode(args, mode, device, data, out_dir)
        rows = _rows_from_metrics(fold_metrics)
        all_rows.extend(rows)
        summaries[mode] = summary

        mode_csv = out_dir / f"{mode}_fold_metrics.csv"
        if rows:
            with mode_csv.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    all_csv = out_dir / "all_fold_metrics.csv"
    if all_rows:
        with all_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    payload = {
        "run_name": run_name,
        "timestamp": ts,
        "config": vars(args),
        "summaries": summaries,
    }
    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nOutputs written to: {out_dir}")


if __name__ == "__main__":
    main()
