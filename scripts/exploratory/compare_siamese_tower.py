#!/usr/bin/env python3
"""
Compare single-eye OD baseline vs SiameseImageTower bilateral model.

Key differences from compare_dual_eye_towers.py:
  - Uses SiameseImageTower (shared backbone, f_mean + f_delta output).
  - Reports BEST-epoch val metrics per fold (not final-epoch), with the
    corresponding holdout metrics snapped at the same checkpoint.
  - Both models are always evaluated on patient-level samples (matched n).
  - Optionally includes two-single-merge as a second reference point.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
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
    """Single-eye (OD) image tower with a linear head."""

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
    """
    Bilateral image model using SiameseImageTower.
    forward(x_od, x_os) -> logits
    """

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

def filter_od_samples(patient_samples: list[dict]) -> list[dict]:
    """Extract patient-level OD-only samples (image_1 = OD)."""
    out = []
    for s in patient_samples:
        if s.get("image_1") is not None and s.get("label_1") is not None:
            out.append({"id_1": s.get("id_1"), "image_1": s["image_1"], "label_1": s["label_1"]})
    return out


def filter_bilateral_samples(patient_samples: list[dict]) -> list[dict]:
    return [
        s for s in patient_samples
        if s.get("image_1") is not None
        and s.get("image_2") is not None
        and s.get("label_1") is not None
    ]


def make_loader(samples, slots, *, image_transform, batch_size, shuffle, num_workers) -> DataLoader:
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
    import pandas as pd
    per_patient = (
        df.groupby(patient_col)[label_col]
        .agg(lambda s: set(pd.to_numeric(s, errors="coerce").dropna().astype(int).tolist()))
    )
    mixed = [pid for pid, labels in per_patient.items() if len(labels) > 1]
    if not mixed:
        return df, []
    return df[~df[patient_col].isin(mixed)].reset_index(drop=True), mixed


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------

def _score(y_true_chunks, y_prob_chunks, num_classes: int):
    if not y_true_chunks:
        return float("nan"), float("nan"), 0
    y = np.concatenate(y_true_chunks)
    p = np.concatenate(y_prob_chunks)
    acc = float((p.argmax(1) == y).mean())
    try:
        auc = (
            float(roc_auc_score(y, p[:, 1]))
            if num_classes == 2
            else float(roc_auc_score(y, p, multi_class="ovr", average="macro"))
        )
    except Exception:
        auc = float("nan")
    return acc, auc, int(len(y))


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------

def train_baseline_epoch(model, loader, opt, device):
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
        total_loss += float(loss.item()) * bs
        total_correct += int((logits.argmax(1) == y).sum())
        total_n += bs
    return (total_loss / total_n if total_n else float("nan"),
            total_correct / total_n if total_n else float("nan"))


def train_siamese_epoch(model, loader, opt, device):
    model.train()
    total_loss = total_correct = total_n = 0
    for batch in loader:
        x1 = batch.get("image_1")
        x2 = batch.get("image_2")
        y = batch.get("label_1")
        if not torch.is_tensor(x1) or not torch.is_tensor(x2):
            continue
        y = to_label_tensor(y, device)
        logits = model(x1.to(device), x2.to(device))
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        bs = y.shape[0]
        total_loss += float(loss.item()) * bs
        total_correct += int((logits.argmax(1) == y).sum())
        total_n += bs
    return (total_loss / total_n if total_n else float("nan"),
            total_correct / total_n if total_n else float("nan"))


def evaluate_baseline(model, loader, device, num_classes):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch.get("image_1")
            y = batch.get("label_1")
            if not torch.is_tensor(x):
                continue
            y_t = to_label_tensor(y, device)
            p = F.softmax(model(x.to(device)), dim=1).cpu().numpy()
            y_true.append(y_t.cpu().numpy())
            y_prob.append(p)
    return _score(y_true, y_prob, num_classes)


def evaluate_siamese(model, loader, device, num_classes):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for batch in loader:
            x1 = batch.get("image_1")
            x2 = batch.get("image_2")
            y = batch.get("label_1")
            if not torch.is_tensor(x1) or not torch.is_tensor(x2):
                continue
            y_t = to_label_tensor(y, device)
            p = F.softmax(model(x1.to(device), x2.to(device)), dim=1).cpu().numpy()
            y_true.append(y_t.cpu().numpy())
            y_prob.append(p)
    return _score(y_true, y_prob, num_classes)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    mode: str
    fold: int
    # Best-epoch validation metrics
    best_epoch: int
    baseline_best_val_auc: float
    baseline_best_val_acc: float
    siamese_best_val_auc: float
    siamese_best_val_acc: float
    # Holdout metrics at the respective best-epoch checkpoint
    baseline_holdout_auc: float
    baseline_holdout_acc: float
    baseline_holdout_n: int
    siamese_holdout_auc: float
    siamese_holdout_acc: float
    siamese_holdout_n: int
    # Sample sizes
    baseline_n: int
    siamese_n: int


def _nan() -> float:
    return float("nan")


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
    profile_od,
    profile_patient,
    fold_dir: Path,
) -> FoldResult:
    holdout_df = split.holdout

    # ---- build samples ----
    bilat_train = filter_bilateral_samples(profile_patient.build_samples(df=split.train, clinical=data))
    bilat_val   = filter_bilateral_samples(profile_patient.build_samples(df=split.val,   clinical=data))
    od_train    = filter_od_samples(bilat_train)
    od_val      = filter_od_samples(bilat_val)

    bilat_holdout = []
    od_holdout    = []
    if holdout_df is not None and not holdout_df.empty:
        bilat_holdout = filter_bilateral_samples(profile_patient.build_samples(df=holdout_df, clinical=data))
        od_holdout    = filter_od_samples(bilat_holdout)

    # ---- models ----
    baseline = BaselineCNN(
        backbone=args.backbone, freeze_ratio=args.freeze_ratio,
        num_classes=num_classes, augment=args.augment,
    ).to(device)
    siamese = SiameseCNN(
        backbone=args.backbone, freeze_ratio=args.freeze_ratio,
        num_classes=num_classes, augment=args.augment,
    ).to(device)

    slots_od      = profile_od.slot_descriptors()
    slots_patient = profile_patient.slot_descriptors()

    # ---- loaders ----
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers)
    train_base = make_loader(od_train,    slots_od,      image_transform=baseline.tower.transform, shuffle=True,  **loader_kw)
    val_base   = make_loader(od_val,      slots_od,      image_transform=baseline.tower.transform, shuffle=False, **loader_kw)
    train_siam = make_loader(bilat_train, slots_patient, image_transform=siamese.tower.transform,  shuffle=True,  **loader_kw)
    val_siam   = make_loader(bilat_val,   slots_patient, image_transform=siamese.tower.transform,  shuffle=False, **loader_kw)

    ho_base = (
        make_loader(od_holdout,    slots_od,      image_transform=baseline.tower.transform, shuffle=False, **loader_kw)
        if od_holdout else None
    )
    ho_siam = (
        make_loader(bilat_holdout, slots_patient, image_transform=siamese.tower.transform,  shuffle=False, **loader_kw)
        if bilat_holdout else None
    )

    opt_base = torch.optim.Adam(baseline.parameters(), lr=args.lr)
    opt_siam = torch.optim.Adam(siamese.parameters(),  lr=args.lr)

    # ---- epoch log ----
    epoch_log_path = fold_dir / "epoch_log.csv"
    epoch_fields = [
        "fold", "epoch",
        "base_train_loss", "base_train_acc",
        "base_val_auc", "base_val_acc", "base_val_n",
        "siam_train_loss", "siam_train_acc",
        "siam_val_auc", "siam_val_acc", "siam_val_n",
        "ho_base_auc", "ho_base_acc", "ho_base_n",
        "ho_siam_auc", "ho_siam_acc", "ho_siam_n",
    ]
    epoch_fp = epoch_log_path.open("w", newline="", encoding="utf-8")
    epoch_writer = csv.DictWriter(epoch_fp, fieldnames=epoch_fields)
    epoch_writer.writeheader()

    def _f(v):
        return None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), 6)

    # ---- best-epoch tracking ----
    best_base_auc = -1.0
    best_siam_auc = -1.0
    best_base_state: Optional[dict] = None
    best_siam_state: Optional[dict] = None
    best_base_val_acc = _nan()
    best_siam_val_acc = _nan()
    # Holdout metrics snapped at best-val checkpoint
    snap_ho_base_auc = _nan()
    snap_ho_base_acc = _nan()
    snap_ho_base_n   = 0
    snap_ho_siam_auc = _nan()
    snap_ho_siam_acc = _nan()
    snap_ho_siam_n   = 0
    best_epoch = 0

    print(
        f"  [fold {fold+1}] training {args.epochs} epochs | "
        f"baseline n_train={len(od_train)} n_val={len(od_val)} | "
        f"siamese  n_train={len(bilat_train)} n_val={len(bilat_val)}",
        flush=True,
    )

    for epoch in range(args.epochs):
        bl_loss, bl_acc = train_baseline_epoch(baseline, train_base, opt_base, device)
        si_loss, si_acc = train_siamese_epoch(siamese,  train_siam, opt_siam,  device)

        b_val_acc, b_val_auc, b_val_n = evaluate_baseline(baseline, val_base, device, num_classes)
        s_val_acc, s_val_auc, s_val_n = evaluate_siamese( siamese,  val_siam,  device, num_classes)

        # Holdout at this epoch (always evaluated for logging, cheaply)
        hb_auc, hb_acc, hb_n = (_nan(), _nan(), 0)
        hs_auc, hs_acc, hs_n = (_nan(), _nan(), 0)
        if ho_base is not None:
            hb_acc, hb_auc, hb_n = evaluate_baseline(baseline, ho_base, device, num_classes)
        if ho_siam is not None:
            hs_acc, hs_auc, hs_n = evaluate_siamese(siamese,   ho_siam, device, num_classes)

        # Best-epoch tracking: snapshot state independently per model
        if not np.isnan(b_val_auc) and b_val_auc > best_base_auc:
            best_base_auc   = b_val_auc
            best_base_val_acc = b_val_acc
            best_base_state = copy.deepcopy(baseline.state_dict())
            snap_ho_base_auc = hb_auc
            snap_ho_base_acc = hb_acc
            snap_ho_base_n   = hb_n

        if not np.isnan(s_val_auc) and s_val_auc > best_siam_auc:
            best_siam_auc   = s_val_auc
            best_siam_val_acc = s_val_acc
            best_siam_state = copy.deepcopy(siamese.state_dict())
            snap_ho_siam_auc = hs_auc
            snap_ho_siam_acc = hs_acc
            snap_ho_siam_n   = hs_n
            best_epoch = epoch + 1

        row = {
            "fold": fold, "epoch": epoch + 1,
            "base_train_loss": _f(bl_loss), "base_train_acc": _f(bl_acc),
            "base_val_auc": _f(b_val_auc), "base_val_acc": _f(b_val_acc), "base_val_n": b_val_n,
            "siam_train_loss": _f(si_loss), "siam_train_acc": _f(si_acc),
            "siam_val_auc": _f(s_val_auc), "siam_val_acc": _f(s_val_acc), "siam_val_n": s_val_n,
            "ho_base_auc": _f(hb_auc), "ho_base_acc": _f(hb_acc), "ho_base_n": hb_n,
            "ho_siam_auc": _f(hs_auc), "ho_siam_acc": _f(hs_acc), "ho_siam_n": hs_n,
        }
        epoch_writer.writerow(row)
        epoch_fp.flush()

        if args.log_every > 0 and (epoch + 1) % args.log_every == 0:
            print(
                f"    ep {epoch+1:>3}/{args.epochs}  "
                f"base val AUC={b_val_auc:.4f}  siam val AUC={s_val_auc:.4f}  "
                f"(best base={best_base_auc:.4f}  best siam={best_siam_auc:.4f})",
                flush=True,
            )

    epoch_fp.close()

    # Save best checkpoints
    if best_base_state is not None:
        torch.save(best_base_state, fold_dir / "best_baseline.pt")
    if best_siam_state is not None:
        torch.save(best_siam_state, fold_dir / "best_siamese.pt")

    result = FoldResult(
        mode=mode, fold=fold,
        best_epoch=best_epoch,
        baseline_best_val_auc=best_base_auc,
        baseline_best_val_acc=best_base_val_acc,
        siamese_best_val_auc=best_siam_auc,
        siamese_best_val_acc=best_siam_val_acc,
        baseline_holdout_auc=snap_ho_base_auc,
        baseline_holdout_acc=snap_ho_base_acc,
        baseline_holdout_n=snap_ho_base_n,
        siamese_holdout_auc=snap_ho_siam_auc,
        siamese_holdout_acc=snap_ho_siam_acc,
        siamese_holdout_n=snap_ho_siam_n,
        baseline_n=len(od_val),
        siamese_n=len(bilat_val),
    )

    print(
        f"  [fold {fold+1}] BEST  "
        f"base val AUC={best_base_auc:.4f} acc={best_base_val_acc:.4f}  "
        f"siam val AUC={best_siam_auc:.4f} acc={best_siam_val_acc:.4f}  "
        f"(siam best epoch={best_epoch})",
        flush=True,
    )
    if snap_ho_base_n > 0 or snap_ho_siam_n > 0:
        print(
            f"  [fold {fold+1}] HOUT  "
            f"base AUC={snap_ho_base_auc:.4f} acc={snap_ho_base_acc:.4f} (n={snap_ho_base_n})  "
            f"siam AUC={snap_ho_siam_auc:.4f} acc={snap_ho_siam_acc:.4f} (n={snap_ho_siam_n})",
            flush=True,
        )

    return result


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _summary(results: list[FoldResult]) -> dict:
    def _means(vals):
        v = np.array([x for x in vals if not np.isnan(x)], dtype=float)
        return (float(np.mean(v)) if len(v) else None,
                float(np.std(v))  if len(v) else None)

    b_val_aucs  = [r.baseline_best_val_auc  for r in results]
    s_val_aucs  = [r.siamese_best_val_auc   for r in results]
    b_ho_aucs   = [r.baseline_holdout_auc   for r in results]
    s_ho_aucs   = [r.siamese_holdout_auc    for r in results]
    b_val_accs  = [r.baseline_best_val_acc  for r in results]
    s_val_accs  = [r.siamese_best_val_acc   for r in results]
    b_ho_accs   = [r.baseline_holdout_acc   for r in results]
    s_ho_accs   = [r.siamese_holdout_acc    for r in results]

    deltas_val_auc = [s - b for b, s in zip(b_val_aucs, s_val_aucs)
                      if not np.isnan(b) and not np.isnan(s)]
    deltas_ho_auc  = [s - b for b, s in zip(b_ho_aucs, s_ho_aucs)
                      if not np.isnan(b) and not np.isnan(s)]

    bva_m, bva_s = _means(b_val_aucs)
    sva_m, sva_s = _means(s_val_aucs)
    bha_m, bha_s = _means(b_ho_aucs)
    sha_m, sha_s = _means(s_ho_aucs)

    return {
        "baseline_best_val":     {"auc_mean": bva_m, "auc_std": bva_s, "acc_mean": _means(b_val_accs)[0]},
        "siamese_best_val":      {"auc_mean": sva_m, "auc_std": sva_s, "acc_mean": _means(s_val_accs)[0]},
        "delta_val_auc":         {"mean": float(np.mean(deltas_val_auc)) if deltas_val_auc else None,
                                  "std":  float(np.std(deltas_val_auc))  if deltas_val_auc else None},
        "baseline_holdout":      {"auc_mean": bha_m, "auc_std": bha_s, "acc_mean": _means(b_ho_accs)[0]},
        "siamese_holdout":       {"auc_mean": sha_m, "auc_std": sha_s, "acc_mean": _means(s_ho_accs)[0]},
        "delta_holdout_auc":     {"mean": float(np.mean(deltas_ho_auc)) if deltas_ho_auc else None,
                                  "std":  float(np.std(deltas_ho_auc))  if deltas_ho_auc else None},
    }


def _print_summary(mode: str, s: dict) -> None:
    def f(v):
        return "nan" if v is None else f"{v:.4f}"

    bv = s["baseline_best_val"]
    sv = s["siamese_best_val"]
    dv = s["delta_val_auc"]
    bh = s["baseline_holdout"]
    sh = s["siamese_holdout"]
    dh = s["delta_holdout_auc"]

    print(f"\n=== Summary [{mode}] (best-epoch metrics) ===")
    print(f"  val   baseline  AUC={f(bv['auc_mean'])}±{f(bv['auc_std'])}  acc={f(bv['acc_mean'])}")
    print(f"  val   siamese   AUC={f(sv['auc_mean'])}±{f(sv['auc_std'])}  acc={f(sv['acc_mean'])}")
    print(f"  val   delta     AUC={f(dv['mean'])}±{f(dv['std'])}")
    print(f"  hout  baseline  AUC={f(bh['auc_mean'])}±{f(bh['auc_std'])}  acc={f(bh['acc_mean'])}")
    print(f"  hout  siamese   AUC={f(sh['auc_mean'])}±{f(sh['auc_std'])}  acc={f(sh['acc_mean'])}")
    print(f"  hout  delta     AUC={f(dh['mean'])}±{f(dh['std'])}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Baseline single-eye vs SiameseImageTower bilateral comparison."
    )
    ap.add_argument("--image-dir",    default="Papila/FundusImages")
    ap.add_argument("--clinical-dir", default="Papila/ClinicalData")
    ap.add_argument("--label-col",    default="Diagnosis")
    ap.add_argument("--cat-cols",     nargs="*", default=["Gender", "Phakic/Pseudophakic"])
    ap.add_argument("--eval-mode",    choices=["binary", "multiclass"], default="binary")
    ap.add_argument(
        "--eval-modes", nargs="+", choices=["binary", "multiclass"], default=None,
        help="Run multiple modes in one pass, e.g. --eval-modes binary multiclass",
    )
    ap.add_argument("--n-splits",          type=int,   default=5)
    ap.add_argument("--fold-seed",         type=int,   default=42)
    ap.add_argument("--holdout-per-class", type=int,   default=0)
    ap.add_argument("--holdout-seed",      type=int,   default=123)
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
        "--exclude-mixed-patients", action="store_true",
        help="Drop patients whose two eyes have different labels before splitting.",
    )
    ap.add_argument(
        "--log-every", type=int, default=5,
        help="Print epoch progress every N epochs (0 to disable).",
    )
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
    run_name = args.run_name or f"siamese_compare_{ts}"
    out_dir = Path(args.output_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_modes = args.eval_modes if args.eval_modes else [args.eval_mode]

    all_results: dict[str, list[FoldResult]] = {}
    summaries:   dict[str, dict] = {}

    for mode in eval_modes:
        import pandas as pd
        df_mode = data.df.copy()

        if args.exclude_mixed_patients:
            before = df_mode["Patient ID"].nunique()
            df_mode, mixed = _drop_mixed_label_patients(
                df_mode, patient_col="Patient ID", label_col=args.label_col
            )
            print(f"[{mode}] dropped {len(mixed)} mixed-label patients "
                  f"({before} -> {df_mode['Patient ID'].nunique()})", flush=True)

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
            holdout_per_class=args.holdout_per_class,
            holdout_seed=args.holdout_seed,
            n_splits=args.n_splits,
            fold_seed=args.fold_seed,
        )
        clinical_ns = SimpleNamespace(df=df_mode, label_col=args.label_col)
        plans = split_manager.build_plans(clinical=clinical_ns, args=split_args, profile=None)
        n_folds = min(args.folds, len(plans))

        profile_od      = build_papila_profile(
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
                profile_od=profile_od,
                profile_patient=profile_patient,
                fold_dir=fold_dir,
            )
            fold_results.append(result)

        # Write per-mode CSV
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
        "run_name": run_name,
        "timestamp": ts,
        "config": vars(args),
        "summaries": summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nOutputs written to: {out_dir}")


if __name__ == "__main__":
    main()
