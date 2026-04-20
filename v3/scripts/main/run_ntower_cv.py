#!/usr/bin/env python
"""
NTowerHT + HyperBridge ensemble cross-validation runner.

Reproduces phase5/embedding_mlp_head using the new module architecture:

  Stage 1 — per-eye NTowerHT  (eye-level samples, BCD training)
    img_tower + cd_tower → Bridge → z_fused  [B, fusion_dim]

  Stage 2 — HyperBridge(embedding_mlp)  (patient-level bilateral samples)
    cat([z_od, z_os]) → Linear(2*fusion_dim → hidden_dim) → ReLU → Dropout → Linear → logits

Training mirrors v3_hypertower ensemble+fused_head:
  - Warmup phases for NTowerHT (tower_warmup → fused_warmup → main)
  - NTowerHT frozen; HyperBridge trained on bilateral samples

Usage (phase5/embedding_mlp_head equivalent):
  python -m v3.scripts.main.run_ntower_cv \\
      --run-name ntower/ensemble_fused \\
      --eval-mode binary \\
      --epochs 30 \\
      --fusion-epochs 10 \\
      --in-memory-cache \\
      --augment \\
      --tune-binary-threshold \\
      --backbone refugelike \\
      --iop-corr-method ratio \\
      --iop-drop-raw \\
      --exclude-cols Axial_Length
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from v3.classes.hypertower_models import NTowerHT, train_ntower_epoch, collect_probs_ntower
from v3.classes.towerbase import _to_label_tensor
from v3.classes.bridges import HyperBridge
from v3.classes.image_towers import ImageEncoder
from v3.classes.clinical_towers import ClinicalEncoder
from v3.classes.papila_builders import build_papila_data
from v3.classes.profiles import build_papila_profile
from v3.classes.split_manager import PatientFirstSplitManager
from types import SimpleNamespace
from v3.classes.loader_factory import (
    filter_eye_samples,
    filter_bilateral_samples,
    make_loader,
    build_balanced_sampler,
)
from v3.classes.metrics import _score_arrays, compute_extended_metrics, tune_binary_threshold
from v3.classes.transforms import build_eval_transform
from v3.classes.utils import seed_everything, choose_device
from v3.classes.croppers import build_image_preprocessor_from_args
from v3.classes.image_loader import CachedImageLoader

REPO_ROOT    = Path(__file__).resolve().parents[3]
IMAGE_DIR    = REPO_ROOT / "Papila" / "FundusImages"
CLINICAL_DIR = REPO_ROOT / "Papila" / "ClinicalData"

# Batch key mapping for per-eye NTowerHT training
EYE_KEY_MAP = {"img": "image_1", "cd": "matrix_1"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--run-name",              default="ntower/ensemble_fused")
    ap.add_argument("--output-root",           default="v3/results")
    ap.add_argument("--eval-mode",             default="binary", choices=["binary", "multiclass"])
    ap.add_argument("--epochs",                type=int,   default=30,
                    help="NTowerHT main-phase epochs")
    ap.add_argument("--fusion-epochs",         type=int,   default=10,
                    help="HyperBridge training epochs (after NTowerHT is frozen)")
    ap.add_argument("--warmup-cd-epochs",      type=int,   default=40,
                    help="Pre-train cd tower + aux head only (no image tower)")
    ap.add_argument("--folds",                 type=int,   default=5)
    ap.add_argument("--fold-seed",             type=int,   default=100,
                    help="Seed for patient splits (rep00=100, rep01=200, ...)")
    ap.add_argument("--seed",                  type=int,   default=1234,
                    help="Seed for model init / per-fold RNG (matches V3HyperTower default)")
    ap.add_argument("--batch-size",            type=int,   default=16)
    ap.add_argument("--lr",                    type=float, default=1e-4)
    ap.add_argument("--backbone",              default="refugelike")
    ap.add_argument("--freeze-ratio",          type=float, default=0.0)
    ap.add_argument("--augment",               action="store_true")
    ap.add_argument("--fusion-dim",            type=int,   default=256)
    ap.add_argument("--hyper-hidden-dim",      type=int,   default=256,
                    help="HyperBridge hidden dim (default matches EmbeddingMLPEnsembleHT)")
    ap.add_argument("--cd-hidden-dim",         type=int,   default=128)
    ap.add_argument("--bcd-prob",              type=float, default=0.5)
    ap.add_argument("--warmup-tower-epochs",   type=int,   default=3)
    ap.add_argument("--warmup-fused-epochs",   type=int,   default=3)
    ap.add_argument("--label-col",             default="Diagnosis")
    ap.add_argument("--iop-corr-method",       default="ratio")
    ap.add_argument("--iop-drop-raw",          action="store_true")
    ap.add_argument("--exclude-cols",          nargs="*",  default=[])
    ap.add_argument("--num-workers",           type=int,   default=0)
    ap.add_argument("--in-memory-cache",       action="store_true")
    ap.add_argument("--tune-binary-threshold", action="store_true")
    ap.add_argument("--device",                default=None)
    return ap


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(args):
    return build_papila_data(
        image_dir=str(IMAGE_DIR),
        clinical_dir=str(CLINICAL_DIR),
        label_col=args.label_col,
        cat_cols=["Gender", "Phakic/Pseudophakic"],
        iop_corr_method=args.iop_corr_method,
        iop_drop_raw=args.iop_drop_raw,
        exclude_cols=args.exclude_cols or [],
    )


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def build_nt(data, num_classes: int, args) -> NTowerHT:
    """Per-eye NTowerHT: image + clinical → bridge."""
    img_enc = ImageEncoder(backbone=args.backbone, freeze_ratio=args.freeze_ratio, augment=args.augment)
    cd_enc  = ClinicalEncoder(clinical_data=data, hidden_dim=args.cd_hidden_dim)
    return NTowerHT(
        towers={"img": img_enc, "cd": cd_enc},
        num_classes=num_classes,
        fusion_dim=args.fusion_dim,
    )


def build_hb(num_classes: int, args) -> HyperBridge:
    """HyperBridge(embedding_mlp): cat([z_od, z_os]) → MLP → logits."""
    return HyperBridge(
        input_dims={"od": args.fusion_dim, "os": args.fusion_dim},
        num_classes=num_classes,
        hidden_dim=args.hyper_hidden_dim,
        mode="embedding_mlp",
    )




# ---------------------------------------------------------------------------
# Per-tower pre-warmup + HyperBridge training/inference
# ---------------------------------------------------------------------------

def train_tower_pre_warmup(
    nt: NTowerHT,
    tower_idx: int,
    loader,
    batch_key: str,
    opt,
    device: torch.device,
) -> tuple[float, float]:
    """Pre-train a single tower (by index) + its bridge aux head only.

    Everything else is frozen.  Caller is responsible for passing a loader
    that omits unnecessary slots (e.g. cd_only_loader strips image_1).
    """
    tower_name = list(nt.towers.keys())[tower_idx]

    for p in nt.parameters():
        p.requires_grad_(False)
    for p in nt.towers[tower_name].parameters():
        p.requires_grad_(True)
    for p in nt.bridge.aux_heads[tower_idx].parameters():
        p.requires_grad_(True)

    nt.train()
    total_loss = total_correct = total_n = 0
    for batch in loader:
        x = batch.get(batch_key)
        y = batch.get("label_1")
        if not torch.is_tensor(x):
            continue
        y_t    = _to_label_tensor(y, device)
        z      = nt.towers[tower_name](x.to(device))
        logits = nt.bridge.aux_heads[tower_idx](z)
        loss   = F.cross_entropy(logits, y_t)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss    += loss.item() * len(y_t)
        total_correct += int((logits.argmax(1) == y_t).sum())
        total_n       += len(y_t)

    for p in nt.parameters():
        p.requires_grad_(True)

    return (
        total_loss / total_n if total_n else float("nan"),
        total_correct / total_n if total_n else float("nan"),
    )


def _encode_eye(nt: NTowerHT, batch: dict, batch_key_map: dict[str, str], device) -> torch.Tensor:
    """Run all towers from a single-eye batch dict and return z_fused."""
    embeddings = {name: nt.towers[name](batch[key].to(device))
                  for name, key in batch_key_map.items()}
    return nt.encode(embeddings)


def train_hb_epoch(
    nt: NTowerHT,
    hb: HyperBridge,
    loader,
    opt,
    device: torch.device,
) -> tuple[float, float]:
    """Train HyperBridge with NTowerHT frozen.

    Each bilateral batch provides both eyes; we encode each through NTowerHT
    to get z_od, z_os, then train HyperBridge to fuse them.
    """
    nt.eval()
    hb.train()
    total_loss = total_correct = total_n = 0
    for batch in loader:
        x1 = batch.get("image_1"); m1 = batch.get("matrix_1")
        x2 = batch.get("image_2"); m2 = batch.get("matrix_2")
        y  = batch.get("label_1")
        if not all(torch.is_tensor(t) for t in (x1, m1, x2, m2)):
            continue
        y_t = _to_label_tensor(y, device)
        # Build per-eye batch dicts (keyed by batch key, as _encode_eye expects)
        od_batch = {"image_1": x1, "matrix_1": m1}
        os_batch = {"image_1": x2, "matrix_1": m2}
        with torch.no_grad():
            z_od = _encode_eye(nt, od_batch, EYE_KEY_MAP, device)
            z_os = _encode_eye(nt, os_batch, EYE_KEY_MAP, device)
        logits, _ = hb({"od": z_od, "os": z_os})
        loss = F.cross_entropy(logits, y_t)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss    += loss.item() * len(y_t)
        total_correct += int((logits.argmax(1) == y_t).sum())
        total_n       += len(y_t)
    return (
        total_loss / total_n if total_n else float("nan"),
        total_correct / total_n if total_n else float("nan"),
    )


def collect_probs_hb(
    nt: NTowerHT,
    hb: HyperBridge,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect HyperBridge predictions (patient-level)."""
    nt.eval(); hb.eval()
    y_all, p_all = [], []
    with torch.no_grad():
        for batch in loader:
            x1 = batch.get("image_1"); m1 = batch.get("matrix_1")
            x2 = batch.get("image_2"); m2 = batch.get("matrix_2")
            y  = batch.get("label_1")
            if not all(torch.is_tensor(t) for t in (x1, m1, x2, m2)):
                continue
            y_t  = _to_label_tensor(y, device)
            z_od = _encode_eye(nt, {"image_1": x1, "matrix_1": m1}, EYE_KEY_MAP, device)
            z_os = _encode_eye(nt, {"image_1": x2, "matrix_1": m2}, EYE_KEY_MAP, device)
            logits, _ = hb({"od": z_od, "os": z_os})
            y_all.append(y_t.cpu().numpy())
            p_all.append(F.softmax(logits, dim=1).cpu().numpy())
    if not y_all:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(y_all), np.concatenate(p_all, axis=0)


# ---------------------------------------------------------------------------
# Fold runner
# ---------------------------------------------------------------------------

def phase_for_epoch(epoch: int, warmup_tower: int, warmup_fused: int) -> str:
    if epoch < warmup_tower:
        return "tower_warmup"
    if epoch < warmup_tower + warmup_fused:
        return "fused_warmup"
    return "main"


def run_fold(fold: int, plans, data, num_classes: int, device, args,
             profile_eye, profile_patient, image_preprocessor, image_cache) -> dict:
    nan = float("nan")
    seed_everything(args.seed + fold * 100)

    split = plans[fold]  # PatientSplit with .train/.val/.test DataFrames

    # Eye-level splits (for NTowerHT training)
    eye_train = filter_eye_samples(profile_eye.build_samples(df=split.train, clinical=data))
    eye_val   = filter_eye_samples(profile_eye.build_samples(df=split.val,   clinical=data))

    # Patient-level splits (for HyperBridge training/eval)
    bilat_train = filter_bilateral_samples(profile_patient.build_samples(df=split.train, clinical=data))
    bilat_val   = filter_bilateral_samples(profile_patient.build_samples(df=split.val,   clinical=data))
    bilat_test  = filter_bilateral_samples(profile_patient.build_samples(
        df=split.test, clinical=data)) if split.test is not None else []

    if not bilat_val:
        print(f"  fold{fold+1}: no bilateral val samples, skipping.", flush=True)
        return {"fold": fold, "val_auc": nan, "val_acc": nan, "val_n": 0,
                "val_kappa": nan, "val_mcc": nan, "val_f1": nan,
                "val_threshold": 0.5, "test_auc": nan, "test_acc": nan, "test_n": nan}

    # Build Stage 1 model only — HyperBridge is built after Stage 1 completes,
    # matching phase5 where EmbeddingMLPEnsembleHT is constructed at Phase 2 start.
    # Building hb here would consume random state and shift all subsequent dropout ops.
    nt = build_nt(data, num_classes, args).to(device)
    opt_nt = torch.optim.Adam(nt.parameters(), lr=args.lr)

    slots_eye     = profile_eye.slot_descriptors()
    slots_patient = profile_patient.slot_descriptors()

    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                     image_cache=image_cache, persistent_workers=args.num_workers > 0)
    eval_transform = build_eval_transform(args.backbone)

    # Unified eye-level loader (all slots, shuffle=True — matches phase5 exactly)
    train_eye_loader = make_loader(
        eye_train, slots_eye, image_transform=nt.transform,
        image_preprocessor=image_preprocessor, shuffle=True, **loader_kw,
    )
    # cd-only loader for warmup: strips image_1 so image decoding is skipped entirely
    slots_cd_only = {k: v for k, v in slots_eye.items() if k != "image_1"}
    cd_warmup_loader = make_loader(
        eye_train, slots_cd_only, image_transform=None,
        image_preprocessor=None, shuffle=True,
        sampler=build_balanced_sampler(eye_train), **loader_kw,
    )

    val_eye_loader = make_loader(
        eye_val, slots_eye, image_transform=eval_transform,
        image_preprocessor=image_preprocessor, shuffle=False, **loader_kw,
    )
    train_bilat_loader = make_loader(
        bilat_train, slots_patient, image_transform=nt.transform,
        image_preprocessor=image_preprocessor, shuffle=True, **loader_kw,
    )
    val_bilat_loader = make_loader(
        bilat_val, slots_patient, image_transform=eval_transform,
        image_preprocessor=image_preprocessor, shuffle=False, **loader_kw,
    )
    test_bilat_loader = make_loader(
        bilat_test, slots_patient, image_transform=eval_transform,
        image_preprocessor=image_preprocessor, shuffle=False, **loader_kw,
    ) if bilat_test else None

    # ── Stage 1: train NTowerHT ───────────────────────────────────────────
    tower_names = list(nt.towers.keys())
    tower_keys  = list(EYE_KEY_MAP.values())

    # Per-tower pre-warmup using the cd-only loader (no image loading overhead)
    pre_warmup_epochs = [args.warmup_cd_epochs if name == "cd" else 0
                         for name in tower_names]
    warmup_loaders    = {"cd": cd_warmup_loader}
    for idx, (name, n_epochs) in enumerate(zip(tower_names, pre_warmup_epochs)):
        if n_epochs == 0:
            continue
        for epoch in range(n_epochs):
            tr_loss, tr_acc = train_tower_pre_warmup(
                nt, idx, warmup_loaders[name], tower_keys[idx], opt_nt, device,
            )
            print(
                f"  fold{fold+1} [NT] ep{epoch+1:03d}/{n_epochs} [{name}_warmup     ]"
                f"  loss={tr_loss:.4f}  tr_acc={tr_acc:.3f}",
                flush=True,
            )

    total_nt_epochs = args.warmup_tower_epochs + args.warmup_fused_epochs + args.epochs

    for epoch in range(total_nt_epochs):
        phase = phase_for_epoch(epoch, args.warmup_tower_epochs, args.warmup_fused_epochs)
        tr_loss, tr_acc = train_ntower_epoch(
            nt, train_eye_loader, opt_nt, device,
            batch_key_map=EYE_KEY_MAP, phase=phase, bcd_prob=args.bcd_prob,
        )
        y_v, p_v = collect_probs_ntower(nt, val_eye_loader, device, batch_key_map=EYE_KEY_MAP)
        _, val_auc, _ = _score_arrays(y_v, p_v, num_classes)
        print(
            f"  fold{fold+1} [NT] ep{epoch+1:03d}/{total_nt_epochs} [{phase:14s}]"
            f"  loss={tr_loss:.4f}  tr_acc={tr_acc:.3f}  val_auc={val_auc:.4f}",
            flush=True,
        )

    # Freeze NTowerHT (final-epoch weights, matching phase5 — no best-checkpoint restore)
    for p in nt.parameters():
        p.requires_grad_(False)

    # ── Stage 2: train HyperBridge ────────────────────────────────────────
    # Build here (not at fold start) to match phase5 random-state sequence
    hb = build_hb(num_classes, args).to(device)
    opt_hb = torch.optim.Adam(hb.parameters(), lr=args.lr)

    for epoch in range(args.fusion_epochs):
        tr_loss, tr_acc = train_hb_epoch(nt, hb, train_bilat_loader, opt_hb, device)
        y_v, p_v = collect_probs_hb(nt, hb, val_bilat_loader, device)
        _, val_auc, _ = _score_arrays(y_v, p_v, num_classes)
        print(
            f"  fold{fold+1} [HB] ep{epoch+1:02d}/{args.fusion_epochs} [fusion       ]"
            f"  loss={tr_loss:.4f}  tr_acc={tr_acc:.3f}  val_auc={val_auc:.4f}",
            flush=True,
        )
    # Final-epoch weights used (no best-checkpoint restore, matching phase5)

    # ── Final eval ────────────────────────────────────────────────────────
    y_val, p_val = collect_probs_hb(nt, hb, val_bilat_loader, device)
    val_acc, val_auc, val_n = _score_arrays(y_val, p_val, num_classes)
    ext = compute_extended_metrics(y_val, p_val, num_classes) if y_val.size else {}

    val_threshold = 0.5
    if args.tune_binary_threshold and num_classes == 2 and y_val.size >= 2:
        val_threshold = tune_binary_threshold(y_val, p_val[:, 1])

    test_auc = test_acc = test_n = nan
    if test_bilat_loader is not None:
        y_te, p_te = collect_probs_hb(nt, hb, test_bilat_loader, device)
        test_acc, test_auc, test_n = _score_arrays(y_te, p_te, num_classes)

    return {
        "fold":          fold,
        "val_auc":       val_auc,
        "val_acc":       val_acc,
        "val_n":         val_n,
        "val_kappa":     ext.get("kappa", nan),
        "val_mcc":       ext.get("mcc", nan),
        "val_f1":        ext.get("macro_f1", nan),
        "val_threshold": val_threshold,
        "test_auc":      test_auc,
        "test_acc":      test_acc,
        "test_n":        test_n,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap   = build_parser()
    args = ap.parse_args()

    num_classes = 2 if args.eval_mode == "binary" else 3
    device      = choose_device(args.device)
    print(f"Device: {device}", flush=True)

    print("Loading data ...", flush=True)
    data = load_data(args)
    print(f"  feature_dim={data.feature_dim}", flush=True)

    image_cache        = CachedImageLoader() if args.in_memory_cache else None
    image_preprocessor = build_image_preprocessor_from_args(args)

    profile_eye     = build_papila_profile(patient_col="Patient ID", label_col=args.label_col, sample_mode="eye")
    profile_patient = build_papila_profile(patient_col="Patient ID", label_col=args.label_col, sample_mode="patient")

    # Filter to binary labels before splitting (mirrors V3HyperTower)
    df_mode = data.df.copy()
    if args.eval_mode == "binary":
        df_mode = df_mode[df_mode[args.label_col].isin([0, 1])].reset_index(drop=True)

    split_mgr   = PatientFirstSplitManager(patient_col="Patient ID", label_col=args.label_col)
    split_args  = SimpleNamespace(eval_mode=args.eval_mode, n_splits=args.folds, fold_seed=args.fold_seed)
    clinical_ns = SimpleNamespace(df=df_mode, label_col=args.label_col)
    plans       = split_mgr.build_plans(clinical=clinical_ns, args=split_args, profile=None)

    out_dir = REPO_ROOT / args.output_root / args.run_name / "binary" / "ntower"
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []
    t0 = time.time()

    for fold in range(args.folds):
        split     = plans[fold]
        bilat_val = filter_bilateral_samples(profile_patient.build_samples(df=split.val, clinical=data))
        print(
            f"\n── fold {fold+1}/{args.folds}"
            f"  train_patients={split.train['Patient ID'].nunique()}"
            f"  val={len(bilat_val)} ──",
            flush=True,
        )
        result = run_fold(fold, plans, data, num_classes, device, args,
                          profile_eye, profile_patient, image_preprocessor, image_cache)
        fold_results.append(result)
        print(
            f"  fold{fold+1} DONE  val_auc={result['val_auc']:.4f}"
            f"  test_auc={result['test_auc']:.4f}",
            flush=True,
        )

    if fold_results:
        val_aucs  = [r["val_auc"]  for r in fold_results if not np.isnan(r["val_auc"])]
        test_aucs = [r["test_auc"] for r in fold_results if not np.isnan(r["test_auc"])]
        summary = {
            "run_name":       args.run_name,
            "backbone":       args.backbone,
            "nt_epochs":      args.epochs,
            "fusion_epochs":  args.fusion_epochs,
            "folds":          args.folds,
            "mean_val_auc":   float(np.mean(val_aucs))  if val_aucs  else float("nan"),
            "std_val_auc":    float(np.std(val_aucs))   if val_aucs  else float("nan"),
            "mean_test_auc":  float(np.mean(test_aucs)) if test_aucs else float("nan"),
            "std_test_auc":   float(np.std(test_aucs))  if test_aucs else float("nan"),
            "elapsed_s":      round(time.time() - t0, 1),
            "fold_results":   fold_results,
        }
        summary_path = out_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"\n{'='*60}", flush=True)
        print(f"Val  AUC: {summary['mean_val_auc']:.4f} ± {summary['std_val_auc']:.4f}", flush=True)
        print(f"Test AUC: {summary['mean_test_auc']:.4f} ± {summary['std_test_auc']:.4f}", flush=True)
        print(f"Saved: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
