#!/usr/bin/env python3
"""
Re-evaluate holdout accuracy using two threshold strategies:

  1. acc   — current behaviour: maximise raw accuracy on (imbalanced) val set
  2. youden — Youden's J = sensitivity + specificity − 1 on val set

For each fold the val probs (already saved) supply the threshold, then the
model is re-run on the holdout set to get the actual holdout accuracy under
each strategy.

Usage
-----
    python scripts/output_analysis/reeval_holdout_threshold.py \
        --run-dir analysis_data/pipeline_10x5 \
        --eval-mode binary

    # or a single nocrop run:
    python scripts/output_analysis/reeval_holdout_threshold.py \
        --run-dir analysis_data/pipeline_nocrop \
        --eval-mode binary \
        --fold-seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classes.v2.metrics import tune_multiclass_bias
from classes.v2.papila_builders import build_papila_data
from classes.v2.loader_factory import filter_bilateral_samples, make_loader
from classes.v2.models import SingleEyeHT, collect_probs_single_components
from classes.v2.profiles import build_papila_profile
from classes.v2.split_manager import PatientFirstSplitManager
from classes.v2.transforms import build_eval_transform
from classes.v2.utils import choose_device


# ---------------------------------------------------------------------------
# Threshold helpers
# ---------------------------------------------------------------------------

def _acc_threshold(y: np.ndarray, p1: np.ndarray) -> float:
    grid = np.linspace(0.0, 1.0, 1001)
    best_t, best_acc = 0.5, -1.0
    for t in grid:
        acc = float(((p1 >= t).astype(int) == y).mean())
        if acc > best_acc or (acc == best_acc and abs(t - 0.5) < abs(best_t - 0.5)):
            best_acc, best_t = acc, float(t)
    return best_t


def _youden_threshold(y: np.ndarray, p1: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y, p1)
    j = tpr + (1.0 - fpr) - 1.0
    return float(thresholds[np.argmax(j)])


def _apply_threshold(probs: np.ndarray, threshold: float, num_classes: int) -> np.ndarray:
    if num_classes == 2:
        return (probs[:, 1] >= threshold).astype(int)
    # multiclass: not applicable for a single scalar threshold
    return probs.argmax(axis=1)


# ---------------------------------------------------------------------------
# Per-fold evaluation
# ---------------------------------------------------------------------------

def eval_fold(
    fold_dir: Path,
    fold_idx: int,
    fold_seed: int,
    eval_mode: str,
    args,
    device: torch.device,
) -> dict | None:

    checkpoint = fold_dir / "best_single.pt"
    if not checkpoint.exists():
        print(f"  [skip] {fold_dir}: no best_single.pt")
        return None

    val_y_path = fold_dir / "y_true.npy"
    val_p_path = fold_dir / "probs_fused.npy"
    if not val_y_path.exists() or not val_p_path.exists():
        print(f"  [skip] {fold_dir}: no val probs")
        return None

    val_y  = np.load(val_y_path)
    val_p  = np.load(val_p_path)
    num_classes = val_p.shape[1]

    # ---- decision boundaries from val ----
    if num_classes == 2:
        t_acc    = _acc_threshold(val_y, val_p[:, 1])
        t_youden = _youden_threshold(val_y, val_p[:, 1])
    else:
        # multiclass: compare raw-acc-optimised bias vs balanced-acc-optimised bias
        # raw-acc bias: temporarily swap objective back to raw accuracy
        from sklearn.metrics import accuracy_score
        import copy

        def _tune_bias_raw(y, p):
            c = p.shape[1]
            bias = np.zeros(c)
            grid = np.linspace(-1.0, 1.0, 41)
            for _ in range(2):
                for k in range(c):
                    best_v, best_acc = bias[k], -1.0
                    old = bias[k]
                    for v in grid:
                        bias[k] = float(v)
                        logits = np.log(np.clip(p, 1e-8, 1.0)) + bias.reshape(1, -1)
                        acc = float((logits.argmax(1) == y).mean())
                        if acc > best_acc or (acc == best_acc and abs(v) < abs(best_v)):
                            best_acc, best_v = acc, float(v)
                    bias[k] = best_v
            return bias

        bias_raw = _tune_bias_raw(val_y, val_p)
        bias_bal = tune_multiclass_bias(val_y, val_p)  # balanced acc objective

    # ---- reconstruct holdout split ----
    data = build_papila_data(
        image_dir=args.image_dir,
        clinical_dir=args.clinical_dir,
        label_col=args.label_col,
        cat_cols=args.cat_cols,
        n_splits=args.n_splits,
        random_seed=fold_seed,
        iop_corr_method=getattr(args, "iop_corr_method", "ratio"),
    )
    df_mode = data.df.copy()
    if eval_mode == "binary":
        df_mode = df_mode[df_mode[args.label_col].isin([0, 1])].reset_index(drop=True)

    splitter = PatientFirstSplitManager(
        patient_col="Patient ID", label_col=args.label_col
    )
    split_args = SimpleNamespace(
        eval_mode=eval_mode,
        holdout_per_class=args.holdout_per_class,
        holdout_seed=args.holdout_seed,
        n_splits=args.n_splits,
        fold_seed=fold_seed,
    )
    plans = splitter.build_plans(
        clinical=SimpleNamespace(df=df_mode, label_col=args.label_col),
        args=split_args,
        profile=None,
    )
    split = plans[fold_idx]

    if split.holdout is None or split.holdout.empty:
        print(f"  [skip] {fold_dir}: no holdout")
        return None

    # ---- build holdout loader ----
    profile_patient = build_papila_profile(
        patient_col="Patient ID", label_col=args.label_col, sample_mode="patient"
    )
    holdout_samples = filter_bilateral_samples(
        profile_patient.build_samples(df=split.holdout, clinical=data)
    )
    if not holdout_samples:
        print(f"  [skip] {fold_dir}: no bilateral holdout samples")
        return None

    eval_transform = build_eval_transform(args.backbone)
    holdout_loader = make_loader(
        holdout_samples,
        profile_patient.slot_descriptors(),
        image_transform=eval_transform,
        image_preprocessor=None,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # ---- load model ----
    model = SingleEyeHT(
        backbone=args.backbone,
        freeze_ratio=0.0,
        augment=False,
        clinical_data=data,
        num_classes=num_classes,
        md_hidden_dim=getattr(args, "md_hidden_dim", 64),
        fusion_dim=getattr(args, "fusion_dim", 128),
        bridge_mode=getattr(args, "bridge_mode", "fused"),
    ).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=False)
    )
    model.eval()

    # ---- run inference ----
    hld_y, hld_p, _, _ = collect_probs_single_components(
        model, holdout_loader, device, aggregate_patient=True
    )

    if len(hld_y) == 0:
        return None

    hld_auc = float(roc_auc_score(
        hld_y, hld_p[:, 1] if num_classes == 2 else hld_p,
        multi_class="ovr" if num_classes > 2 else "raise",
    ))

    if num_classes == 2:
        acc_old = float(((hld_p[:, 1] >= t_acc).astype(int)    == hld_y).mean())
        acc_new = float(((hld_p[:, 1] >= t_youden).astype(int) == hld_y).mean())
        bacc_old = balanced_accuracy_score(hld_y, (hld_p[:, 1] >= t_acc).astype(int))
        bacc_new = balanced_accuracy_score(hld_y, (hld_p[:, 1] >= t_youden).astype(int))
        row_extra = {"t_old": t_acc, "t_new": t_youden}
    else:
        logits_raw = np.log(np.clip(hld_p, 1e-8, 1.0)) + bias_raw.reshape(1, -1)
        logits_bal = np.log(np.clip(hld_p, 1e-8, 1.0)) + bias_bal.reshape(1, -1)
        preds_raw  = logits_raw.argmax(1)
        preds_bal  = logits_bal.argmax(1)
        acc_old  = float((preds_raw == hld_y).mean())
        acc_new  = float((preds_bal == hld_y).mean())
        bacc_old = balanced_accuracy_score(hld_y, preds_raw)
        bacc_new = balanced_accuracy_score(hld_y, preds_bal)
        row_extra = {"bias_raw": bias_raw.tolist(), "bias_bal": bias_bal.tolist()}

    return {
        "fold_dir":    str(fold_dir),
        "fold":        fold_idx,
        "fold_seed":   fold_seed,
        "hld_auc":     hld_auc,
        "hld_acc_old": acc_old,
        "hld_acc_new": acc_new,
        "hld_bacc_old": bacc_old,
        "hld_bacc_new": bacc_new,
        "n_holdout":   len(hld_y),
        **row_extra,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir",     required=True,
                    help="e.g. analysis_data/pipeline_10x5 or analysis_data/pipeline_nocrop")
    ap.add_argument("--eval-mode",   default="binary", choices=["binary", "multiclass"])
    ap.add_argument("--fold-seed",   type=int, default=None,
                    help="Override fold seed (for single-rep runs). "
                         "For 10x5, seeds are inferred from rep dir name.")
    ap.add_argument("--backbone",    default="refugelike")
    ap.add_argument("--n-splits",    type=int, default=5)
    ap.add_argument("--holdout-per-class", type=int, default=5)
    ap.add_argument("--holdout-seed",      type=int, default=123)
    ap.add_argument("--label-col",   default="Diagnosis")
    ap.add_argument("--cat-cols",    nargs="*", default=["Gender"])
    ap.add_argument("--image-dir",   default="Papila/FundusImages")
    ap.add_argument("--clinical-dir",default="Papila/ClinicalData")
    ap.add_argument("--iop-corr-method", default="ratio")
    ap.add_argument("--batch-size",  type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--md-hidden-dim", type=int, default=128)
    ap.add_argument("--fusion-dim",    type=int, default=256)
    ap.add_argument("--bridge-mode",   default="fused")
    ap.add_argument("--device",        default="auto")
    ap.add_argument("--out",           default=None)
    args = ap.parse_args()

    device   = choose_device(args.device)
    run_dir  = Path(args.run_dir)
    out_path = Path(args.out) if args.out else \
               run_dir / f"reeval_threshold_{args.eval_mode}.csv"

    # Discover fold dirs — supports both flat (fold0..fold4) and
    # rep-based (rep00/binary/ensemble/fold0) layouts
    _BASE_SEED   = 100
    _SEED_STRIDE = 100

    fold_jobs: list[tuple[Path, int, int]] = []  # (fold_dir, fold_idx, fold_seed)

    rep_dirs = sorted(run_dir.glob("rep[0-9]*"))
    if rep_dirs:
        for rep_dir in rep_dirs:
            rep_n = int(rep_dir.name.replace("rep", ""))
            fold_seed = _BASE_SEED + rep_n * _SEED_STRIDE
            mode_dir  = rep_dir / args.eval_mode / "ensemble"
            if not mode_dir.exists():
                continue
            for fd in sorted(mode_dir.glob("fold[0-9]*"), key=lambda p: int(p.name[4:])):
                fold_jobs.append((fd, int(fd.name[4:]), fold_seed))
    else:
        # flat layout
        mode_dir  = run_dir / args.eval_mode / "ensemble"
        fold_seed = args.fold_seed if args.fold_seed is not None else 42
        for fd in sorted(mode_dir.glob("fold[0-9]*"), key=lambda p: int(p.name[4:])):
            fold_jobs.append((fd, int(fd.name[4:]), fold_seed))

    if not fold_jobs:
        sys.exit(f"No fold directories found under {run_dir}")

    print(f"Found {len(fold_jobs)} folds to re-evaluate")

    rows = []
    for i, (fold_dir, fold_idx, fold_seed) in enumerate(fold_jobs):
        print(f"\n[{i+1}/{len(fold_jobs)}] {fold_dir}  fold_seed={fold_seed}")
        row = eval_fold(fold_dir, fold_idx, fold_seed, args.eval_mode, args, device)
        if row:
            rows.append(row)
            print(f"  hld_acc(old)={row['hld_acc_old']:.3f}  "
                  f"hld_acc(new)={row['hld_acc_new']:.3f}  "
                  f"hld_bacc(old)={row['hld_bacc_old']:.3f}  "
                  f"hld_bacc(new)={row['hld_bacc_new']:.3f}  "
                  f"hld_auc={row['hld_auc']:.3f}")

    if not rows:
        print("No results.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
    is_binary = args.eval_mode == "binary"
    old_label = "acc-threshold" if is_binary else "raw-acc bias"
    new_label = "Youden-J"      if is_binary else "balanced-acc bias"

    print(f"\n{'='*60}")
    print(f"Summary ({args.eval_mode}, n={len(df)} folds)")
    print(f"{'='*60}")
    print(f"  Holdout AUC:              {df.hld_auc.mean():.4f} ± {df.hld_auc.std():.4f}")
    print(f"  Holdout acc  ({old_label:<18}): {df.hld_acc_old.mean():.4f} ± {df.hld_acc_old.std():.4f}")
    print(f"  Holdout acc  ({new_label:<18}): {df.hld_acc_new.mean():.4f} ± {df.hld_acc_new.std():.4f}")
    print(f"  Holdout bacc ({old_label:<18}): {df.hld_bacc_old.mean():.4f} ± {df.hld_bacc_old.std():.4f}")
    print(f"  Holdout bacc ({new_label:<18}): {df.hld_bacc_new.mean():.4f} ± {df.hld_bacc_new.std():.4f}")
    print(f"  Delta acc  (new − old):  {df.hld_acc_new.mean() - df.hld_acc_old.mean():+.4f}")
    print(f"  Delta bacc (new − old):  {df.hld_bacc_new.mean() - df.hld_bacc_old.mean():+.4f}")


if __name__ == "__main__":
    main()
