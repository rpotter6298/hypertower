"""V2HyperTower — central orchestrator for the V2 mode-comparison pipeline.

All model classes, metric helpers, croppers and data utilities live in their
respective category-specific modules.  This file owns only:
  - V2HyperTower (the big-picture orchestrator)
  - V2ModeComparator (thin backward-compat shim)
  - module-level ``build_parser`` alias (used by old scripts that import it
    directly; will be removed once all callers are updated)
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch

from v3.classes.croppers import build_image_preprocessor_from_args
from v3.classes.dataset import _ClinicalView  # noqa: F401  (re-exported for compat)
from v3.classes.loader_factory import (
    build_balanced_sampler,
    filter_bilateral_samples,
    filter_eye_samples,
    make_loader,
)
from v3.classes.metrics import _score_arrays, _svf, _tune_and_snap
from v3.classes.models import (
    BilateralHT,
    FusedEnsembleHT,
    SingleEyeHT,
    V2ModeComparisonOps,
    collect_probs_bilateral,
    collect_probs_bilateral_components,
    collect_probs_classic,
    collect_probs_ensemble,
    collect_probs_ensemble_pereye,
    collect_probs_eye_level,
    collect_probs_fused,
    collect_probs_single_components,
    train_bilateral_epoch,
    train_fusion_epoch,
    train_single_epoch,
)
from v3.classes.papila_builders import build_papila_data
from v3.classes.predictions import PredictionStore, head_names_for_mode
from v3.classes.profiles import build_papila_profile
from v3.classes.results import FoldArtifacts, FoldResult, _f, _nan, _sv
from v3.classes.split_manager import PatientFirstSplitManager
from v3.classes.transforms import build_eval_transform
from v3.classes.utils import (
    _drop_mixed_label_patients,
    _relabel_mixed_patients_to_max,
    choose_device,
    seed_everything,
)
from v3.classes.hypertower_logger import HypertowerLogger


# ---------------------------------------------------------------------------
# Fusion-event helper
# ---------------------------------------------------------------------------

def _fusion_events(
    y: np.ndarray,
    pf: np.ndarray,
    pi: np.ndarray,
    pm: np.ndarray,
) -> tuple[int, int]:
    """
    Count fusion corrections and errors.
    - correction: fused correct, both img and md wrong
    - error:      fused wrong, both img and md correct
    Returns (n_corrections, n_errors).
    """
    pred_f = pf.argmax(1); pred_i = pi.argmax(1); pred_m = pm.argmax(1)
    corr = int(((pred_f == y) & (pred_i != y) & (pred_m != y)).sum())
    err  = int(((pred_f != y) & (pred_i == y) & (pred_m == y)).sum())
    return corr, err


def _cm_cells(y: np.ndarray, p: np.ndarray, num_classes: int) -> dict[str, int]:
    """
    Return confusion matrix cells as a flat dict.
    Binary:     keys tn/fp/fn/tp
    Multiclass: keys cm_{i}_{j} for true class i, predicted class j
    Returns empty dict if arrays are empty or wrong shape.
    """
    if not y.size or p.ndim < 2 or p.shape[1] != num_classes:
        return {}
    pred = p.argmax(1)
    if num_classes == 2:
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tp = int(((pred == 1) & (y == 1)).sum())
        return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
    # multiclass: full NxN matrix
    out: dict[str, int] = {}
    for i in range(num_classes):
        for j in range(num_classes):
            out[f"cm_{i}_{j}"] = int(((y == i) & (pred == j)).sum())
    return out


# ---------------------------------------------------------------------------
# Per-sample prediction logging
# ---------------------------------------------------------------------------

def _save_predictions_csv(
    fold_dir: Path,
    eval_mode: str,
    y_true: np.ndarray,
    heads: dict,          # {"fused": probs_array, "img": probs_array, "md": probs_array, ...}
    suffix: str = "",     # e.g. "_pereye"
) -> None:
    """
    Save a per-sample CSV with predicted class, per-class probabilities,
    and TP/FP/TN/FN (binary) or correct flag (multiclass) for every head.
    """
    N = len(y_true)
    num_classes = next(p.shape[1] for p in heads.values() if p is not None)
    rows = []
    for i in range(N):
        true = int(y_true[i])
        row: dict = {"idx": i, "y_true": true}
        for head_name, probs in heads.items():
            if probs is None:
                continue
            pred = int(probs[i].argmax())
            row[f"pred_{head_name}"] = pred
            for c in range(num_classes):
                row[f"prob_{head_name}_c{c}"] = float(probs[i, c])
            if eval_mode == "binary":
                row[f"tp_{head_name}"] = int(pred == 1 and true == 1)
                row[f"fp_{head_name}"] = int(pred == 1 and true == 0)
                row[f"tn_{head_name}"] = int(pred == 0 and true == 0)
                row[f"fn_{head_name}"] = int(pred == 0 and true == 1)
            else:
                row[f"correct_{head_name}"] = int(pred == true)
        rows.append(row)

    if not rows:
        return
    csv_path = fold_dir / f"predictions{suffix}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# V2HyperTower
# ---------------------------------------------------------------------------

class V2HyperTower:
    """Central orchestrator.  Construct with ``V2HyperTower(args)``, call ``.run()``."""

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------

    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        ap = argparse.ArgumentParser(
            description=(
                "Three HyperTower modes: Classic (eye-level), Ensemble (patient-level avg), "
                "Bilateral (BilateralBridge with shared towers).  Pure k-fold CV."
            )
        )
        ap.add_argument("--image-dir",    default="Papila/FundusImages")
        ap.add_argument("--clinical-dir", default="Papila/ClinicalData")
        ap.add_argument("--label-col",    default="Diagnosis")
        ap.add_argument("--cat-cols",     nargs="*", default=["Gender", "Phakic/Pseudophakic"])
        ap.add_argument("--exclude-cols", nargs="*", default=[],
                        help="Feature columns to exclude entirely from the clinical feature matrix.")
        ap.add_argument("--eval-mode",    choices=["binary", "multiclass"], default="multiclass")
        ap.add_argument(
            "--tower-mode", choices=["single", "ensemble", "bilateral", "classic"],
            default="single",
            help="Train/evaluate a single tower mode.",
        )
        ap.add_argument("--n-splits",          type=int,   default=5)
        ap.add_argument("--fold-seed",         type=int,   default=42)
        ap.add_argument("--holdout-per-class", type=int,   default=5,
                        help="Patients per class reserved for holdout before train/test split (0 disables)")
        ap.add_argument("--holdout-seed",      type=int,   default=123,
                        help="Random seed for holdout sampling")
        ap.add_argument(
            "--folds", type=int, default=None,
            help="Optional cap on how many folds to run (default: all --n-splits).",
        )
        ap.add_argument("--epochs",       type=int,   default=40)
        ap.add_argument(
            "--warmup-tower-epochs", type=int, default=None,
            help="Extra tower warmup epochs (added before main epochs). Default: auto by mode.",
        )
        ap.add_argument(
            "--warmup-fused-epochs", type=int, default=None,
            help="Extra fused warmup epochs (added before main epochs). Default: auto by mode.",
        )
        ap.add_argument("--single-warmup-tower-epochs", type=int, default=None,
                        help="Single-eye model tower warmup (overrides --warmup-tower-epochs).")
        ap.add_argument("--single-warmup-fused-epochs", type=int, default=None,
                        help="Single-eye model fused warmup (overrides --warmup-fused-epochs).")
        ap.add_argument("--warmup-cd-epochs", type=int, default=0,
                        help="MD-only warmup epochs before tower warmup. Trains only cd_tower + "
                             "classifier_cd (no CNN forward pass, so 50-100 epochs is cheap).")
        ap.add_argument("--bilat-warmup-tower-epochs", type=int, default=None,
                        help="Bilateral model tower warmup (overrides --warmup-tower-epochs).")
        ap.add_argument("--bilat-warmup-fused-epochs", type=int, default=None,
                        help="Bilateral model fused warmup (overrides --warmup-fused-epochs).")
        ap.add_argument("--batch-size",   type=int,   default=8)
        ap.add_argument("--lr",           type=float, default=1e-4)
        ap.add_argument("--bcd-prob",     type=float, default=0.5,
                        help="Tower-only step probability during main phase (per model).")
        ap.add_argument("--tower-loss-mode", choices=["bcd", "all"], default="bcd",
                        help="Main-phase tower loss strategy: "
                             "'bcd' (Block Coordinate Descent — randomly train one tower or fused per step) "
                             "or 'all' (sum all three losses — fused + img + md — every step).")
        ap.add_argument("--backbone",     default="refugelike")
        ap.add_argument("--freeze-ratio", type=float, default=0.0)
        ap.add_argument("--augment",      action="store_true")
        ap.add_argument("--balanced-sampling", action="store_true",
                        help="Use WeightedRandomSampler during training to equalise class frequency (default: off).")
        ap.add_argument("--num-workers",  type=int,   default=4)
        ap.add_argument("--in-memory-cache", action="store_true", default=True,
                        help="Cache preprocessed images in RAM (default: on).")
        ap.add_argument("--no-in-memory-cache", action="store_false", dest="in_memory_cache",
                        help="Disable in-memory image cache.")
        ap.add_argument("--cache-workers", type=int, default=4,
                        help="Threads for prebuilding in-memory image cache (default: 4).")
        ap.add_argument("--device",       choices=["auto", "cpu", "cuda"], default="auto")
        ap.add_argument("--seed",         type=int,   default=1234)
        ap.add_argument("--run-name",     default=None)
        ap.add_argument("--output-root",  default="analysis_data")
        # Optional ROI cropping
        ap.add_argument("--img-crop-manifest", type=str, default=None,
                        help="Path to crop manifest CSV for ROI cropping.")
        ap.add_argument("--img-crop-gt", action="store_true",
                        help="Use ground-truth masks/contours from manifest for ROI crop.")
        ap.add_argument("--img-crop-weights", type=str, default=None,
                        help="UNet weights path for ROI cropping from predicted masks.")
        ap.add_argument("--img-crop-normalize", type=str, default="per_image",
                        choices=["per_image", "imagenet"],
                        help="UNet input normalization mode.")
        ap.add_argument("--img-crop-threshold", type=float, default=0.5,
                        help="UNet mask threshold for ROI extraction.")
        ap.add_argument("--img-crop-tta", action="store_true",
                        help="Enable flip-TTA during UNet mask inference.")
        ap.add_argument("--img-crop-scale", type=float, default=2.5,
                        help="Disc-radius multiplier for square crop.")
        ap.add_argument("--img-crop-size", type=int, default=224,
                        help="Output ROI size before tower transforms.")
        ap.add_argument("--img-crop-cache", type=str, default="cache_data/hypertower_crops",
                        help="Cache directory for cropped images and geometry sidecars.")
        ap.add_argument("--persist-img-crop-cache", action="store_true",
                        help="Keep existing cached crop .npz files instead of clearing at run start.")
        # Architecture
        ap.add_argument("--cd-hidden-dim", type=int, default=128,
                        help="ClinicalTower hidden dimension.")
        ap.add_argument("--fusion-dim",    type=int, default=256,
                        help="Bridge/BilateralBridge fusion dimension.")
        ap.add_argument("--bridge-mode", default="fused",
                        choices=["fused", "image_only", "clinical_only"],
                        help="Bridge fusion mode: fused (default), image_only, or clinical_only.")
        # Mixed patients
        ap.add_argument(
            "--exclude-mixed-patients",
            dest="exclude_mixed_patients", action="store_true",
            help="Drop patients whose two eyes have different labels before splitting.",
        )
        ap.add_argument(
            "--include-mixed-patients",
            dest="exclude_mixed_patients", action="store_false",
        )
        ap.add_argument(
            "--relabel-mixed-patients-to-max",
            dest="relabel_mixed_patients_to_max",
            action="store_true",
            help="When mixed patients are included, relabel both eyes to patient max severity.",
        )
        ap.add_argument(
            "--keep-mixed-raw-labels",
            dest="relabel_mixed_patients_to_max",
            action="store_false",
            help=argparse.SUPPRESS,
        )
        ap.set_defaults(exclude_mixed_patients=False, relabel_mixed_patients_to_max=False)
        # Tuning
        ap.add_argument(
            "--tune-binary-threshold", action="store_true",
            help="Tune per-model binary threshold on validation each epoch.",
        )
        ap.add_argument(
            "--tune-multiclass-bias", action="store_true",
            help="Tune per-model multiclass log-prob bias on validation each epoch.",
        )
        ap.add_argument("--ece-bins",     type=int, default=10)
        ap.add_argument("--log-every",    type=int, default=1)
        ap.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True,
                        help="Save best_single.pt / best_holdout_single.pt per fold (use --no-save-checkpoints to disable)")
        ap.add_argument("--use-last-epoch", action="store_true", default=False,
                        help="Score using the final epoch's model state rather than the best-AUC checkpoint.")
        # IOP feature options
        ap.add_argument(
            "--iop-corr-method",
            choices=["ratio", "ols", "lad", "multi"],
            default="ratio",
            help="Perkins→Pneumatic conversion method: ratio (default), ols, lad, or multi (+CCT).",
        )
        ap.add_argument(
            "--iop-drop-raw",
            action="store_true",
            default=False,
            help="Exclude IOP_raw from the feature matrix (keep only IOP_corr).",
        )
        ap.add_argument(
            "--fused-head", action="store_true",
            help="(ensemble mode only) After base SingleEyeHT training, freeze it and train a "
                 "small logit-level MLP fusion head on bilateral samples instead of averaging "
                 "OD/OS softmax probabilities.",
        )
        ap.add_argument(
            "--fusion-epochs", type=int, default=10,
            help="Number of epochs to train the fusion head (--fused-head, ensemble mode only).",
        )
        return ap

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, args) -> None:
        self.args = args
        self.device = choose_device(args.device)
        seed_everything(args.seed)

        print(f"Device: {self.device}", flush=True)
        print("Loading PAPILA data...", flush=True)
        self.data = build_papila_data(
            image_dir=args.image_dir,
            clinical_dir=args.clinical_dir,
            label_col=args.label_col,
            cat_cols=list(args.cat_cols),
            n_splits=args.n_splits,
            random_seed=args.fold_seed,
            iop_corr_method=getattr(args, "iop_corr_method", "ratio"),
            iop_drop_raw=getattr(args, "iop_drop_raw", False),
            exclude_cols=list(getattr(args, "exclude_cols", []) or []),
        )
        print(f"Loaded: {len(self.data.df)} rows  feature_dim={self.data.feature_dim}", flush=True)
        self.image_preprocessor = build_image_preprocessor_from_args(args)
        self.profile_eye = build_papila_profile(
            patient_col="Patient ID", label_col=args.label_col, sample_mode="eye"
        )
        self.profile_patient = build_papila_profile(
            patient_col="Patient ID", label_col=args.label_col, sample_mode="patient"
        )

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> Path:
        """Execute the full fold loop for one eval_mode × tower_mode combination."""
        args = self.args
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_name = args.run_name or f"hypertower_modes_{ts}"
        out_dir = Path(args.output_root) / run_name
        out_dir.mkdir(parents=True, exist_ok=True)

        mode = args.eval_mode
        tower_mode = "single" if args.tower_mode == "classic" else args.tower_mode
        df_mode = self.data.df.copy()

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
            if args.relabel_mixed_patients_to_max:
                before_rows = len(df_mode)
                df_mode, changed_rows, still_mixed = _relabel_mixed_patients_to_max(
                    df_mode, patient_col="Patient ID", label_col=args.label_col
                )
                print(
                    f"[{mode}] relabeled mixed patients to max severity "
                    f"(changed={changed_rows}, rows={before_rows}→{len(df_mode)}, "
                    f"remaining_mixed={len(still_mixed)}).",
                    flush=True,
                )
            else:
                print(f"[{mode}] keeping mixed-label patients with raw per-eye labels.", flush=True)

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
        requested_folds = args.n_splits if args.folds is None else int(args.folds)
        n_folds = min(requested_folds, len(plans))

        tm_dir = out_dir / mode / tower_mode
        tm_dir.mkdir(parents=True, exist_ok=True)
        fold_results: list[FoldResult] = []

        # Override profiles with df_mode slice.
        profile_eye = build_papila_profile(
            patient_col="Patient ID", label_col=args.label_col, sample_mode="eye"
        )
        profile_patient = build_papila_profile(
            patient_col="Patient ID", label_col=args.label_col, sample_mode="patient"
        )

        # ---- PredictionStore — build once before fold loop ---------------
        fused_head = getattr(args, "fused_head", False)
        _head_names = head_names_for_mode(tower_mode, fused_head=fused_head)
        fusion_epochs = int(getattr(args, "fusion_epochs", 10)) if fused_head else 0
        _global_warmup_tower = getattr(args, "warmup_tower_epochs", None)
        _global_warmup_fused = getattr(args, "warmup_fused_epochs", None)
        _warmup_tower = (
            int(args.single_warmup_tower_epochs)
            if getattr(args, "single_warmup_tower_epochs", None) is not None
            else int(_global_warmup_tower) if _global_warmup_tower is not None else 2
        )
        _warmup_fused = (
            int(args.single_warmup_fused_epochs)
            if getattr(args, "single_warmup_fused_epochs", None) is not None
            else int(_global_warmup_fused) if _global_warmup_fused is not None else 2
        )
        _warmup_cd    = int(getattr(args, "warmup_cd_epochs", 0))
        _total_epochs = _warmup_cd + _warmup_tower + _warmup_fused + int(args.epochs) + fusion_epochs

        # sample IDs depend on mode: single uses eye IDs, others use patient IDs
        if tower_mode in ("single", "classic"):
            _sample_ids = [
                f"{row['Patient ID']}{row['eyeID']}"
                for _, row in df_mode.iterrows()
            ]
            _y_true = df_mode[args.label_col].tolist()
        else:
            # one row per patient (deduplicate — take first occurrence per patient)
            _pat_df = df_mode.drop_duplicates(subset="Patient ID")
            _sample_ids = _pat_df["Patient ID"].astype(str).tolist()
            _y_true = _pat_df[args.label_col].tolist()

        pred_store = PredictionStore(
            sample_ids=_sample_ids,
            y_true=_y_true,
            head_names=_head_names,
            n_folds=n_folds,
            n_epochs=_total_epochs,
            n_classes=num_classes,
        )

        image_cache: dict | None = {} if getattr(args, "in_memory_cache", False) else None

        for fold in range(n_folds):
            seed_everything(args.seed + fold * 100)
            fold_dir = tm_dir / f"fold{fold}"
            fold_dir.mkdir(exist_ok=True)

            print(f"\n[{mode}:{tower_mode}] fold {fold+1}/{n_folds}", flush=True)
            result, artifacts = self._run_fold(
                fold=fold,
                split=plans[fold],
                mode=mode,
                data=self.data,
                num_classes=num_classes,
                profile_eye=profile_eye,
                profile_patient=profile_patient,
                fold_dir=fold_dir,
                tower_mode=tower_mode,
                pred_store=pred_store,
                image_cache=image_cache,
            )
            fold_results.append(result)
            if artifacts.y_true_ensemble is not None:
                np.save(fold_dir / "y_true.npy", artifacts.y_true_ensemble)
            if artifacts.probs_ensemble is not None:
                np.save(fold_dir / "probs_fused.npy", artifacts.probs_ensemble)
            if artifacts.probs_ensemble_img is not None:
                np.save(fold_dir / "probs_img.npy", artifacts.probs_ensemble_img)
            if artifacts.probs_ensemble_md is not None:
                np.save(fold_dir / "probs_cd.npy", artifacts.probs_ensemble_md)
            if artifacts.y_true_classic is not None:
                np.save(fold_dir / "y_true.npy", artifacts.y_true_classic)
            if artifacts.probs_classic is not None:
                np.save(fold_dir / "probs_classic.npy", artifacts.probs_classic)
            if artifacts.probs_classic_img is not None:
                np.save(fold_dir / "probs_classic_img.npy", artifacts.probs_classic_img)
            if artifacts.probs_classic_md is not None:
                np.save(fold_dir / "probs_classic_md.npy", artifacts.probs_classic_md)
            if artifacts.y_true_ensemble_pereye is not None:
                np.save(fold_dir / "y_true_pereye.npy", artifacts.y_true_ensemble_pereye)
            if artifacts.probs_ensemble_pereye is not None:
                np.save(fold_dir / "probs_fused_pereye.npy", artifacts.probs_ensemble_pereye)
            if artifacts.probs_ensemble_img_pereye is not None:
                np.save(fold_dir / "probs_img_pereye.npy", artifacts.probs_ensemble_img_pereye)
            if artifacts.probs_ensemble_md_pereye is not None:
                np.save(fold_dir / "probs_cd_pereye.npy", artifacts.probs_ensemble_md_pereye)
            if artifacts.logits_ensemble is not None:
                np.save(fold_dir / "logits_fused.npy", artifacts.logits_ensemble)
            if artifacts.logits_ensemble_img is not None:
                np.save(fold_dir / "logits_img.npy", artifacts.logits_ensemble_img)
            if artifacts.logits_ensemble_md is not None:
                np.save(fold_dir / "logits_cd.npy", artifacts.logits_ensemble_md)
            if artifacts.logits_classic is not None:
                np.save(fold_dir / "logits_classic.npy", artifacts.logits_classic)
            if artifacts.logits_classic_img is not None:
                np.save(fold_dir / "logits_classic_img.npy", artifacts.logits_classic_img)
            if artifacts.logits_classic_md is not None:
                np.save(fold_dir / "logits_classic_md.npy", artifacts.logits_classic_md)
            if artifacts.logits_ensemble_pereye is not None:
                np.save(fold_dir / "logits_fused_pereye.npy", artifacts.logits_ensemble_pereye)
            if artifacts.logits_ensemble_img_pereye is not None:
                np.save(fold_dir / "logits_img_pereye.npy", artifacts.logits_ensemble_img_pereye)
            if artifacts.logits_ensemble_md_pereye is not None:
                np.save(fold_dir / "logits_cd_pereye.npy", artifacts.logits_ensemble_md_pereye)
            if artifacts.probs_bilat is not None:
                np.save(fold_dir / "probs_bilat.npy", artifacts.probs_bilat)
            if artifacts.probs_fused is not None:
                np.save(fold_dir / "probs_fused_head.npy", artifacts.probs_fused)
            # y_true is shared across all heads for the same fold
            if artifacts.y_true_bilat is not None and artifacts.y_true_ensemble is None:
                np.save(fold_dir / "y_true.npy", artifacts.y_true_bilat)
            # per-sample prediction CSVs
            if artifacts.y_true_ensemble is not None:
                _save_predictions_csv(
                    fold_dir, mode, artifacts.y_true_ensemble,
                    {"fused": artifacts.probs_ensemble,
                     "img":   artifacts.probs_ensemble_img,
                     "md":    artifacts.probs_ensemble_md},
                )
            if artifacts.y_true_ensemble_pereye is not None:
                _save_predictions_csv(
                    fold_dir, mode, artifacts.y_true_ensemble_pereye,
                    {"fused": artifacts.probs_ensemble_pereye,
                     "img":   artifacts.probs_ensemble_img_pereye,
                     "md":    artifacts.probs_ensemble_md_pereye},
                    suffix="_pereye",
                )
            if artifacts.y_true_classic is not None:
                _save_predictions_csv(
                    fold_dir, mode, artifacts.y_true_classic,
                    {"fused": artifacts.probs_classic,
                     "img":   artifacts.probs_classic_img,
                     "md":    artifacts.probs_classic_md},
                    suffix="_classic",
                )

        fold_csv = tm_dir / "fold_results.csv"
        csv_fields = list(FoldResult.__dataclass_fields__.keys())
        with fold_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=csv_fields)
            w.writeheader()
            for r in fold_results:
                w.writerow({k: getattr(r, k) for k in csv_fields})

        summary = self._summary(fold_results)
        self._print_summary(f"{mode}:{tower_mode}", summary, tower_mode=tower_mode)
        metric_key = {
            "single": "classic_val_auc",
            "ensemble": "ensemble_val_auc",
            "bilateral": "bilat_val_auc",
        }[tower_mode]
        fold_metrics = []
        best_vals = []
        for r in fold_results:
            best_val = getattr(r, metric_key)
            fold_metrics.append({
                "fold": r.fold,
                "best_metric_value": _f(best_val),
                "best_epoch": (r.best_epoch_bilat if tower_mode == "bilateral" else r.best_epoch_single),
                "monitor": metric_key,
            })
            if not np.isnan(float(best_val)):
                best_vals.append(float(best_val))

        ts_now = time.strftime("%Y%m%d_%H%M%S")
        mode_summary = {
            "run_id": run_name,
            "backbone": args.backbone,
            "epochs": args.epochs,
            "warmup_cd_epochs": getattr(args, "warmup_cd_epochs", 0),
            "warmup_tower_epochs": args.warmup_tower_epochs,
            "warmup_fused_epochs": args.warmup_fused_epochs,
            "single_warmup_tower_epochs": args.single_warmup_tower_epochs,
            "single_warmup_fused_epochs": args.single_warmup_fused_epochs,
            "bilat_warmup_tower_epochs": args.bilat_warmup_tower_epochs,
            "bilat_warmup_fused_epochs": args.bilat_warmup_fused_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "eval_mode": mode,
            "tower_mode": tower_mode,
            "n_splits": n_folds,
            "best_metric": metric_key,
            "best_metric_mode": "max",
            "best_metric_mean": (float(np.mean(best_vals)) if best_vals else None),
            "best_metric_std": (float(np.std(best_vals)) if best_vals else None),
            "fold_metrics": fold_metrics,
            "mode_summary": summary,
        }
        (tm_dir / "summary.json").write_text(json.dumps(mode_summary, indent=2), encoding="utf-8")
        pred_store.save(tm_dir / "predictions.npz")

        root_summary_path = out_dir / "summary.json"
        if root_summary_path.exists():
            try:
                payload = json.loads(root_summary_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
        else:
            payload = {}
        payload.setdefault("run_name", run_name)
        payload.setdefault("timestamp", ts_now)
        payload["config"] = vars(args)
        payload.setdefault("summaries", {})
        payload["summaries"][f"{mode}:{tower_mode}"] = summary
        root_summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        print(f"\nOutputs written to: {out_dir}")
        return out_dir

    # ------------------------------------------------------------------
    # Fold runner
    # ------------------------------------------------------------------

    def _run_fold(
        self,
        fold: int,
        split,
        mode: str,
        data,
        num_classes: int,
        profile_eye,
        profile_patient,
        fold_dir: Path,
        tower_mode: str,
        pred_store: "PredictionStore | None" = None,
        image_cache: "dict | None" = None,
    ) -> tuple[FoldResult, FoldArtifacts]:
        args = self.args
        device = self.device
        image_preprocessor = self.image_preprocessor
        nan = _nan()
        tower_mode = "single" if tower_mode == "classic" else tower_mode
        run_single = tower_mode in ("single", "ensemble")
        run_bilat  = tower_mode == "bilateral"
        run_fused  = (tower_mode == "ensemble") and bool(getattr(args, "fused_head", False))

        global_warmup_tower = getattr(args, "warmup_tower_epochs", None)
        global_warmup_fused = getattr(args, "warmup_fused_epochs", None)
        single_warmup_tower = (
            int(args.single_warmup_tower_epochs)
            if getattr(args, "single_warmup_tower_epochs", None) is not None
            else int(global_warmup_tower) if global_warmup_tower is not None else 2
        )
        single_warmup_fused = (
            int(args.single_warmup_fused_epochs)
            if getattr(args, "single_warmup_fused_epochs", None) is not None
            else int(global_warmup_fused) if global_warmup_fused is not None else 2
        )
        bilat_warmup_tower = (
            int(args.bilat_warmup_tower_epochs)
            if getattr(args, "bilat_warmup_tower_epochs", None) is not None
            else int(global_warmup_tower) if global_warmup_tower is not None else 4
        )
        bilat_warmup_fused = (
            int(args.bilat_warmup_fused_epochs)
            if getattr(args, "bilat_warmup_fused_epochs", None) is not None
            else int(global_warmup_fused) if global_warmup_fused is not None else 3
        )
        single_warmup_cd = int(getattr(args, "warmup_cd_epochs", 0)) if run_single else 0
        if not run_single:
            single_warmup_tower = 0
            single_warmup_fused = 0
        if not run_bilat:
            bilat_warmup_tower = 0
            bilat_warmup_fused = 0
        main_epochs = int(args.epochs)
        total_single_epochs = (single_warmup_cd + single_warmup_tower + single_warmup_fused + main_epochs) if run_single else 0
        total_bilat_epochs = (bilat_warmup_tower + bilat_warmup_fused + main_epochs) if run_bilat else 0
        total_epochs = max(total_single_epochs, total_bilat_epochs)

        # ---- samples ---------------------------------------------------
        eye_train   = filter_eye_samples(profile_eye.build_samples(df=split.train, clinical=data))
        bilat_train = filter_bilateral_samples(profile_patient.build_samples(df=split.train, clinical=data))
        bilat_val   = filter_bilateral_samples(profile_patient.build_samples(df=split.val, clinical=data))

        # Register split labels in the prediction store
        if pred_store is not None:
            if tower_mode in ("single", "classic"):
                # eye-level IDs: "{patient_id}{eyeID}"
                train_sids = [f"{s['id_1']}{s.get('eye_id_1','')}" for s in eye_train]
                val_sids   = [f"{s['id_1']}{s.get('eye_id_1','')}" for s in bilat_val]
            else:
                train_sids = [str(s["id_1"]) for s in bilat_train]
                val_sids   = [str(s["id_1"]) for s in bilat_val]
            pred_store.set_split(fold, train_sids, "train")
            pred_store.set_split(fold, val_sids,   "val")

        if len(bilat_val) == 0:
            print(f"  [fold {fold+1}] WARNING: no bilateral val samples; skipping fold.", flush=True)
            empty = FoldResult(
                mode=mode, fold=fold,
                best_epoch_single=0, best_epoch_bilat=0,
                classic_val_auc=nan,  classic_val_acc=nan,  classic_val_kappa=nan,
                classic_val_mcc=nan,  classic_val_f1=nan,   classic_val_recall=None,
                classic_val_ece=nan,  classic_val_threshold=nan, classic_val_bias=None,
                classic_val_n=0,
                ensemble_val_auc=nan, ensemble_val_acc=nan, ensemble_val_kappa=nan,
                ensemble_val_mcc=nan, ensemble_val_f1=nan,  ensemble_val_recall=None,
                ensemble_val_ece=nan, ensemble_val_threshold=nan, ensemble_val_bias=None,
                ensemble_val_n=0,
                bilat_val_auc=nan,  bilat_val_acc=nan,  bilat_val_kappa=nan,
                bilat_val_mcc=nan,  bilat_val_f1=nan,   bilat_val_recall=None,
                bilat_val_ece=nan,  bilat_val_threshold=nan, bilat_val_bias=None,
                bilat_val_n=0,
                classic_holdout_auc=nan,  classic_holdout_acc=nan,
                ensemble_holdout_auc=nan, ensemble_holdout_acc=nan,
                bilat_holdout_auc=nan,    bilat_holdout_acc=nan,
                holdout_n=0,
                single_train_n=len(eye_train), bilat_train_n=len(bilat_train),
            )
            return empty, FoldArtifacts(
                y_true_classic=None, probs_classic=None,
                y_true_ensemble=None, probs_ensemble=None,
                y_true_bilat=None, probs_bilat=None,
            )

        # ---- models ----------------------------------------------------
        single = None
        bilateral = None
        if run_single:
            single = SingleEyeHT(
                backbone=args.backbone, freeze_ratio=args.freeze_ratio,
                augment=args.augment, clinical_data=data,
                num_classes=num_classes,
                cd_hidden_dim=args.cd_hidden_dim, fusion_dim=args.fusion_dim,
                bridge_mode=getattr(args, "bridge_mode", "fused"),
            ).to(device)
        if run_bilat:
            bilateral = BilateralHT(
                backbone=args.backbone, freeze_ratio=args.freeze_ratio,
                augment=args.augment, clinical_data=data,
                num_classes=num_classes,
                cd_hidden_dim=args.cd_hidden_dim, fusion_dim=args.fusion_dim,
            ).to(device)

        slots_eye     = profile_eye.slot_descriptors()
        slots_patient = profile_patient.slot_descriptors()
        loader_kw     = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                             image_cache=image_cache)

        # ---- loaders ---------------------------------------------------
        use_balanced = bool(getattr(args, "balanced_sampling", False))
        train_single_loader = None
        train_eval_loader   = None   # non-shuffled, no sampler — for per-epoch train logging
        train_bilat_loader  = None
        cd_only_loader      = None   # image-free loader for md_warmup phase
        if run_single:
            single_sampler = build_balanced_sampler(eye_train) if use_balanced else None
            train_single_loader = make_loader(
                eye_train, slots_eye,
                image_transform=single.transform,
                image_preprocessor=image_preprocessor,
                shuffle=True,
                sampler=single_sampler,
                **loader_kw,
            )
            train_eval_loader = make_loader(
                eye_train, slots_eye,
                image_transform=build_eval_transform(args.backbone),
                image_preprocessor=image_preprocessor,
                shuffle=False,
                **loader_kw,
            )
            if single_warmup_cd > 0:
                # MD-only loader: drop image_1 so PIL never opens files during md_warmup.
                # Always use balanced sampling for md_warmup — MD features alone are weaker
                # than images and collapse to majority class without class balancing.
                slots_cd_only = {k: v for k, v in slots_eye.items() if k != "image_1"}
                md_warmup_sampler = single_sampler if single_sampler is not None else build_balanced_sampler(eye_train)
                cd_only_loader = make_loader(
                    eye_train, slots_cd_only,
                    image_transform=None,
                    image_preprocessor=None,
                    shuffle=True,
                    sampler=md_warmup_sampler,
                    **loader_kw,
                )
        if run_bilat:
            bilat_sampler = build_balanced_sampler(bilat_train) if use_balanced else None
            train_bilat_loader = make_loader(
                bilat_train, slots_patient,
                image_transform=bilateral.transform,
                image_preprocessor=image_preprocessor,
                shuffle=True,
                sampler=bilat_sampler,
                **loader_kw,
            )
        elif run_fused:
            # Fused head trains on bilateral samples using the single model's transform.
            fused_sampler = build_balanced_sampler(bilat_train) if use_balanced else None
            train_bilat_loader = make_loader(
                bilat_train, slots_patient,
                image_transform=single.transform,
                image_preprocessor=image_preprocessor,
                shuffle=True,
                sampler=fused_sampler,
                **loader_kw,
            )
        eval_transform = build_eval_transform(args.backbone)
        val_loader = make_loader(
            bilat_val, slots_patient,
            image_transform=eval_transform,
            image_preprocessor=image_preprocessor,
            shuffle=False,
            **loader_kw,
        )

        # ---- holdout loader (optional) ---------------------------------
        holdout_bilat: list = []
        holdout_loader = None
        if split.holdout is not None and not split.holdout.empty:
            holdout_bilat = filter_bilateral_samples(
                profile_patient.build_samples(df=split.holdout, clinical=data)
            )
            if holdout_bilat:
                holdout_loader = make_loader(
                    holdout_bilat, slots_patient,
                    image_transform=eval_transform,
                    image_preprocessor=image_preprocessor,
                    shuffle=False,
                    **loader_kw,
                )
                print(f"  [fold {fold+1}] holdout_n={len(holdout_bilat)} (bilateral patients)", flush=True)
                if pred_store is not None:
                    pred_store.set_split(fold, [str(s["id_1"]) for s in holdout_bilat], "holdout")

        # ---- prebuild in-memory image cache (fold 0 only; shared dict fills for later folds) ----
        if image_cache is not None:
            cache_workers = int(getattr(args, "cache_workers", 4))
            _loaders_to_warm = [
                train_single_loader, train_bilat_loader, val_loader, holdout_loader,
            ]
            for _ldr in _loaders_to_warm:
                if _ldr is not None:
                    _ldr.dataset.prebuild_image_cache(cache_workers=cache_workers)

        opt_single   = torch.optim.Adam(single.parameters(), lr=args.lr)   if run_single else None
        opt_bilateral = torch.optim.Adam(bilateral.parameters(), lr=args.lr) if run_bilat  else None

        # ---- epoch log -------------------------------------------------
        epoch_fields = [
            "fold", "epoch",
            "phase_single", "phase_bilat",
            "main_epoch_single", "main_epoch_bilat",
            "single_active", "bilat_active",
            "single_train_loss", "single_train_acc",
            # val — fused head (existing)
            "classic_val_auc",      "classic_val_acc",      "classic_val_n",
            "ensemble_val_auc",     "ensemble_val_acc",     "ensemble_val_n",
            "bilat_train_loss",     "bilat_train_acc",
            "bilat_val_auc",        "bilat_val_acc",        "bilat_val_n",
            # val — img/md heads + fusion events
            "classic_val_auc_img",  "classic_val_acc_img",
            "classic_val_auc_cd",   "classic_val_acc_cd",
            "classic_val_fe_corr",  "classic_val_fe_err",
            "ensemble_val_auc_img", "ensemble_val_acc_img",
            "ensemble_val_auc_cd",  "ensemble_val_acc_cd",
            "ensemble_val_fe_corr", "ensemble_val_fe_err",
            "bilat_val_auc_img",    "bilat_val_acc_img",
            "bilat_val_auc_cd",     "bilat_val_acc_cd",
            "bilat_val_fe_corr",    "bilat_val_fe_err",
            # holdout — fused head (existing)
            "classic_holdout_auc",  "classic_holdout_acc",
            "ensemble_holdout_auc", "ensemble_holdout_acc",
            "bilat_holdout_auc",    "bilat_holdout_acc",
            # holdout — img/md heads + fusion events
            "classic_holdout_auc_img",  "classic_holdout_acc_img",
            "classic_holdout_auc_cd",   "classic_holdout_acc_cd",
            "classic_holdout_fe_corr",  "classic_holdout_fe_err",
            "ensemble_holdout_auc_img", "ensemble_holdout_acc_img",
            "ensemble_holdout_auc_cd",  "ensemble_holdout_acc_cd",
            "ensemble_holdout_fe_corr", "ensemble_holdout_fe_err",
            # train-set eval pass (eval mode, all 3 heads)
            "train_auc_fused",  "train_acc_fused",
            "train_auc_img",    "train_acc_img",
            "train_auc_cd",     "train_acc_cd",
            "train_fe_corr",    "train_fe_err",
            "train_n",
            "is_best_single", "is_best_bilat",
            "is_best_holdout_single", "is_best_holdout_bilat",
        ]
        # CM columns — named by num_classes so binary and multiclass both work
        if num_classes == 2:
            _cm_keys = ["tn", "fp", "fn", "tp"]
        else:
            _cm_keys = [f"cm_{i}_{j}" for i in range(num_classes) for j in range(num_classes)]
        for _split in ("classic_val", "ensemble_val", "classic_holdout", "ensemble_holdout", "train"):
            for _head in ("fused", "img", "md"):
                for _k in _cm_keys:
                    epoch_fields.append(f"{_split}_{_head}_{_k}")
        fold_logger = HypertowerLogger(run_dir=fold_dir)

        # per-epoch accumulation for npy tensors
        _epoch_train_pf:  list[np.ndarray] = []
        _epoch_train_pi:  list[np.ndarray] = []
        _epoch_train_pm:  list[np.ndarray] = []
        _epoch_train_ids: list[np.ndarray] = []
        _epoch_train_y:   list[np.ndarray] = []
        # per-eye val accumulators (ensemble mode: OD and OS separate)
        _epoch_val_pf_od: list[np.ndarray] = []
        _epoch_val_pi_od: list[np.ndarray] = []
        _epoch_val_pm_od: list[np.ndarray] = []
        _epoch_val_pf_os: list[np.ndarray] = []
        _epoch_val_pi_os: list[np.ndarray] = []
        _epoch_val_pm_os: list[np.ndarray] = []
        _epoch_val_y:     list[np.ndarray] = []
        _epoch_val_ids:   list[np.ndarray] = []

        # ---- best-epoch trackers ---------------------------------------
        best_single_auc = -1.0
        best_bilat_auc  = -1.0
        best_epoch_single = 0
        best_epoch_bilat  = 0
        best_single_state: Optional[dict] = None
        best_bilat_state:  Optional[dict] = None
        snap_classic:       dict = {}
        snap_ensemble:      dict = {}
        snap_bilat:         dict = {}
        snap_holdout_single: dict = {}
        snap_holdout_bilat:  dict = {}
        snap_fused:         dict = {}
        snap_holdout_fused: dict = {}
        best_holdout_single_auc   = -1.0
        best_holdout_bilat_auc    = -1.0
        best_epoch_holdout_single = 0
        best_epoch_holdout_bilat  = 0
        best_holdout_single_state: Optional[dict] = None
        best_holdout_bilat_state:  Optional[dict] = None

        if run_single:
            print(
                f"  [fold {fold+1}]  single_train_n={len(eye_train)} (eye-level)  "
                f"val_n={len(bilat_val)}  "
                f"single_warmup=md{single_warmup_cd}+twr{single_warmup_tower}+fus{single_warmup_fused} total={total_single_epochs}",
                flush=True,
            )
        else:
            print(
                f"  [fold {fold+1}]  bilat_train_n={len(bilat_train)} (bilateral)  "
                f"val_n={len(bilat_val)}  "
                f"bilat_warmup={bilat_warmup_tower}+{bilat_warmup_fused} total={total_bilat_epochs}",
                flush=True,
            )

        # ---- epoch loop ------------------------------------------------
        _prev_phase_single = "inactive"  # used to detect md_warmup → next phase transition
        for epoch in range(total_epochs):
            _epoch_t0 = time.time()
            if not run_single:
                phase_single, main_epoch_single, single_active = "inactive", 0, False
            elif epoch < single_warmup_cd:
                phase_single, main_epoch_single, single_active = "cd_warmup", 0, True
            elif epoch < (single_warmup_cd + single_warmup_tower):
                phase_single, main_epoch_single, single_active = "tower_warmup", 0, True
            elif epoch < (single_warmup_cd + single_warmup_tower + single_warmup_fused):
                phase_single, main_epoch_single, single_active = "fused_warmup", 0, True
            elif epoch < total_single_epochs:
                phase_single, main_epoch_single, single_active = (
                    "main",
                    epoch - single_warmup_cd - single_warmup_tower - single_warmup_fused + 1,
                    True,
                )
            else:
                phase_single, main_epoch_single, single_active = "done", main_epochs, False

            if not run_bilat:
                phase_bilat, main_epoch_bilat, bilat_active = "inactive", 0, False
            elif epoch < bilat_warmup_tower:
                phase_bilat, main_epoch_bilat, bilat_active = "tower_warmup", 0, True
            elif epoch < (bilat_warmup_tower + bilat_warmup_fused):
                phase_bilat, main_epoch_bilat, bilat_active = "fused_warmup", 0, True
            elif epoch < total_bilat_epochs:
                phase_bilat, main_epoch_bilat, bilat_active = (
                    "main",
                    epoch - bilat_warmup_tower - bilat_warmup_fused + 1,
                    True,
                )
            else:
                phase_bilat, main_epoch_bilat, bilat_active = "done", main_epochs, False

            if run_single and single_active:
                _active_loader = cd_only_loader if phase_single == "cd_warmup" else train_single_loader
                sl_loss, sl_acc = train_single_epoch(
                    single, _active_loader, opt_single, device,
                    phase=phase_single, bcd_prob=float(args.bcd_prob),
                    tower_loss_mode=args.tower_loss_mode,
                )
            else:
                sl_loss, sl_acc = nan, nan

            if run_bilat and bilat_active:
                bl_loss, bl_acc = train_bilateral_epoch(
                    bilateral, train_bilat_loader, opt_bilateral, device,
                    phase=phase_bilat, bcd_prob=float(args.bcd_prob),
                    tower_loss_mode=args.tower_loss_mode,
                )
            else:
                bl_loss, bl_acc = nan, nan

            _skip_val_eval = (phase_single == "cd_warmup")

            if run_single and tower_mode == "single" and not _skip_val_eval:
                y_cl, p_cl, p_cl_img, p_cl_cd = collect_probs_single_components(
                    single, val_loader, device, aggregate_patient=False
                )
                cl_acc, cl_auc, cl_n = _score_arrays(y_cl, p_cl, num_classes)
                cl_acc_img = float((p_cl_img.argmax(1) == y_cl).mean()) if y_cl.size else nan
                cl_acc_cd  = float((p_cl_cd.argmax(1) == y_cl).mean()) if y_cl.size else nan
                _, cl_auc_img, _ = _score_arrays(y_cl, p_cl_img, num_classes)
                _, cl_auc_cd, _  = _score_arrays(y_cl, p_cl_cd,  num_classes)
                y_en = np.array([], dtype=np.int64)
                p_en = p_en_img = p_en_cd = np.zeros((0, num_classes), dtype=np.float32)
                en_acc = en_auc = nan
                en_n = 0
                en_acc_img = en_acc_cd = en_auc_img = en_auc_cd = nan
            elif run_single and tower_mode == "ensemble" and not _skip_val_eval:
                (y_en,
                 _p_en_f_od, _p_en_i_od, _p_en_m_od,
                 _p_en_f_os, _p_en_i_os, _p_en_m_os,
                 _en_pat_ids) = collect_probs_ensemble_pereye(
                    single, val_loader, device, return_ids=True
                )
                # patient-level averages (used for metrics, same as before)
                p_en     = 0.5 * (_p_en_f_od + _p_en_f_os)
                p_en_img = 0.5 * (_p_en_i_od + _p_en_i_os)
                p_en_cd  = 0.5 * (_p_en_m_od + _p_en_m_os)
                en_acc, en_auc, en_n = _score_arrays(y_en, p_en, num_classes)
                en_acc_img = float((p_en_img.argmax(1) == y_en).mean()) if y_en.size else nan
                en_acc_cd  = float((p_en_cd.argmax(1) == y_en).mean()) if y_en.size else nan
                _, en_auc_img, _ = _score_arrays(y_en, p_en_img, num_classes)
                _, en_auc_cd, _  = _score_arrays(y_en, p_en_cd,  num_classes)
                y_cl = np.array([], dtype=np.int64)
                p_cl = p_cl_img = p_cl_cd = np.zeros((0, num_classes), dtype=np.float32)
                cl_acc = cl_auc = nan
                cl_n = 0
                cl_acc_img = cl_acc_cd = cl_auc_img = cl_auc_cd = nan
            else:
                y_cl = y_en = np.array([], dtype=np.int64)
                p_cl = p_cl_img = p_cl_cd = np.zeros((0, num_classes), dtype=np.float32)
                p_en = p_en_img = p_en_cd = np.zeros((0, num_classes), dtype=np.float32)
                cl_acc = cl_auc = en_acc = en_auc = nan
                cl_n = en_n = 0
                cl_acc_img = cl_acc_cd = en_acc_img = en_acc_cd = nan
                cl_auc_img = cl_auc_cd = en_auc_img = en_auc_cd = nan

            if run_bilat and not _skip_val_eval:
                y_bi, p_bi, p_bi_img, p_bi_cd = collect_probs_bilateral_components(
                    bilateral, val_loader, device
                )
                bi_acc, bi_auc, bi_n = _score_arrays(y_bi, p_bi, num_classes)
                bi_acc_img = float((p_bi_img.argmax(1) == y_bi).mean()) if y_bi.size else nan
                bi_acc_cd  = float((p_bi_cd.argmax(1) == y_bi).mean()) if y_bi.size else nan
                _, bi_auc_img, _ = _score_arrays(y_bi, p_bi_img, num_classes)
                _, bi_auc_cd, _  = _score_arrays(y_bi, p_bi_cd,  num_classes)
            else:
                y_bi = np.array([], dtype=np.int64)
                p_bi = np.zeros((0, 0), dtype=np.float32)
                bi_acc = bi_auc = nan
                bi_n = 0
                bi_acc_img = bi_acc_cd = bi_auc_img = bi_auc_cd = nan

            # --- holdout evaluation ------------------------------------
            # defaults (overwritten below when holdout_loader is not None)
            _z2 = np.zeros((0, num_classes), dtype=np.float32)
            _e2 = np.array([], dtype=np.int64)
            y_cl_h = y_en_h = _e2
            p_cl_h = p_cl_h_img = p_cl_h_cd = _z2
            p_en_h = p_en_h_img = p_en_h_cd = _z2

            if holdout_loader is not None and not _skip_val_eval:
                if run_single and tower_mode == "single":
                    y_cl_h, p_cl_h, p_cl_h_img, p_cl_h_cd = collect_probs_single_components(
                        single, holdout_loader, device, aggregate_patient=False
                    )
                    _, cl_auc_h, _ = _score_arrays(y_cl_h, p_cl_h, num_classes)
                    cl_acc_h = float((p_cl_h.argmax(1) == y_cl_h).mean()) if y_cl_h.size else nan
                    _, cl_auc_h_img, _ = _score_arrays(y_cl_h, p_cl_h_img, num_classes)
                    cl_acc_h_img = float((p_cl_h_img.argmax(1) == y_cl_h).mean()) if y_cl_h.size else nan
                    _, cl_auc_h_md, _  = _score_arrays(y_cl_h, p_cl_h_cd,  num_classes)
                    cl_acc_h_md  = float((p_cl_h_cd.argmax(1) == y_cl_h).mean()) if y_cl_h.size else nan
                    cl_fe_h_corr, cl_fe_h_err = _fusion_events(y_cl_h, p_cl_h, p_cl_h_img, p_cl_h_cd)
                    en_auc_h = en_acc_h = nan
                    en_auc_h_img = en_acc_h_img = en_auc_h_md = en_acc_h_md = nan
                    en_fe_h_corr = en_fe_h_err = 0
                elif run_single and tower_mode == "ensemble":
                    y_en_h, p_en_h, p_en_h_img, p_en_h_cd = collect_probs_single_components(
                        single, holdout_loader, device, aggregate_patient=True
                    )
                    _, en_auc_h, _ = _score_arrays(y_en_h, p_en_h, num_classes)
                    en_acc_h = float((p_en_h.argmax(1) == y_en_h).mean()) if y_en_h.size else nan
                    _, en_auc_h_img, _ = _score_arrays(y_en_h, p_en_h_img, num_classes)
                    en_acc_h_img = float((p_en_h_img.argmax(1) == y_en_h).mean()) if y_en_h.size else nan
                    _, en_auc_h_md, _  = _score_arrays(y_en_h, p_en_h_cd,  num_classes)
                    en_acc_h_md  = float((p_en_h_cd.argmax(1) == y_en_h).mean()) if y_en_h.size else nan
                    en_fe_h_corr, en_fe_h_err = _fusion_events(y_en_h, p_en_h, p_en_h_img, p_en_h_cd)
                    cl_auc_h = cl_acc_h = nan
                    cl_auc_h_img = cl_acc_h_img = cl_auc_h_md = cl_acc_h_md = nan
                    cl_fe_h_corr = cl_fe_h_err = 0
                else:
                    cl_auc_h = cl_acc_h = en_auc_h = en_acc_h = nan
                    cl_auc_h_img = cl_acc_h_img = cl_auc_h_md = cl_acc_h_md = nan
                    en_auc_h_img = en_acc_h_img = en_auc_h_md = en_acc_h_md = nan
                    cl_fe_h_corr = cl_fe_h_err = en_fe_h_corr = en_fe_h_err = 0
                if run_bilat:
                    y_bi_h, p_bi_h, _, _ = collect_probs_bilateral_components(
                        bilateral, holdout_loader, device
                    )
                    _, bi_auc_h, _ = _score_arrays(y_bi_h, p_bi_h, num_classes)
                    bi_acc_h = float((p_bi_h.argmax(1) == y_bi_h).mean()) if y_bi_h.size else nan
                else:
                    bi_auc_h = bi_acc_h = nan
            else:
                cl_auc_h = cl_acc_h = en_auc_h = en_acc_h = bi_auc_h = bi_acc_h = nan
                cl_auc_h_img = cl_acc_h_img = cl_auc_h_md = cl_acc_h_md = nan
                en_auc_h_img = en_acc_h_img = en_auc_h_md = en_acc_h_md = nan
                cl_fe_h_corr = cl_fe_h_err = en_fe_h_corr = en_fe_h_err = 0

            # --- fusion-event helpers for val sets ----------------------
            cl_fe_corr, cl_fe_err = _fusion_events(y_cl, p_cl, p_cl_img, p_cl_cd) if y_cl.size else (0, 0)
            en_fe_corr, en_fe_err = _fusion_events(y_en, p_en, p_en_img, p_en_cd) if y_en.size else (0, 0)
            bi_fe_corr, bi_fe_err = (0, 0)  # bilateral components not separated the same way

            # --- train eval pass (eval mode, all 3 heads) ----------------
            tr_auc_f = tr_acc_f = tr_auc_i = tr_acc_i = tr_auc_m = tr_acc_m = nan
            tr_fe_corr = tr_fe_err = tr_n = 0
            y_tr = np.array([], dtype=np.int64)
            p_tr_f = p_tr_i = p_tr_m = np.zeros((0, num_classes), dtype=np.float32)
            if run_single and train_eval_loader is not None and not _skip_val_eval:
                y_tr, p_tr_f, p_tr_i, p_tr_m, tr_ids = collect_probs_eye_level(
                    single, train_eval_loader, device, return_ids=True
                )
                if y_tr.size:
                    _, tr_auc_f, _ = _score_arrays(y_tr, p_tr_f, num_classes)
                    tr_acc_f = float((p_tr_f.argmax(1) == y_tr).mean())
                    _, tr_auc_i, _ = _score_arrays(y_tr, p_tr_i, num_classes)
                    tr_acc_i = float((p_tr_i.argmax(1) == y_tr).mean())
                    _, tr_auc_m, _ = _score_arrays(y_tr, p_tr_m, num_classes)
                    tr_acc_m = float((p_tr_m.argmax(1) == y_tr).mean())
                    tr_fe_corr, tr_fe_err = _fusion_events(y_tr, p_tr_f, p_tr_i, p_tr_m)
                    tr_n = int(y_tr.size)
                    # accumulate for npy tensors
                    _epoch_train_pf.append(p_tr_f)
                    _epoch_train_pi.append(p_tr_i)
                    _epoch_train_pm.append(p_tr_m)
                    _epoch_train_ids.append(tr_ids)
                    _epoch_train_y.append(y_tr)
                    # record into PredictionStore
                    if pred_store is not None:
                        if tower_mode in ("single", "classic"):
                            pred_store.record(fold, epoch, tr_ids, "fused", p_tr_f)
                            pred_store.record(fold, epoch, tr_ids, "img",   p_tr_i)
                            pred_store.record(fold, epoch, tr_ids, "md",    p_tr_m)
                        else:  # ensemble: separate OD and OS by eye suffix
                            od_mask = np.array([str(i).endswith("OD") for i in tr_ids])
                            os_mask = ~od_mask
                            od_pids = [str(i)[:-2] for i in tr_ids[od_mask]]
                            os_pids = [str(i)[:-2] for i in tr_ids[os_mask]]
                            pred_store.record(fold, epoch, od_pids, "od_fused", p_tr_f[od_mask])
                            pred_store.record(fold, epoch, od_pids, "od_img",   p_tr_i[od_mask])
                            pred_store.record(fold, epoch, od_pids, "od_md",    p_tr_m[od_mask])
                            pred_store.record(fold, epoch, os_pids, "os_fused", p_tr_f[os_mask])
                            pred_store.record(fold, epoch, os_pids, "os_img",   p_tr_i[os_mask])
                            pred_store.record(fold, epoch, os_pids, "os_md",    p_tr_m[os_mask])

            # accumulate val for npy tensors
            if run_single and tower_mode == "ensemble" and y_en.size:
                _epoch_val_pf_od.append(_p_en_f_od)
                _epoch_val_pi_od.append(_p_en_i_od)
                _epoch_val_pm_od.append(_p_en_m_od)
                _epoch_val_pf_os.append(_p_en_f_os)
                _epoch_val_pi_os.append(_p_en_i_os)
                _epoch_val_pm_os.append(_p_en_m_os)
                _epoch_val_y.append(y_en)
                _epoch_val_ids.append(_en_pat_ids)
            elif run_single and tower_mode == "single" and y_cl.size:
                # single mode: no per-eye split, reuse same array for both slots
                _epoch_val_pf_od.append(p_cl)
                _epoch_val_pi_od.append(p_cl_img)
                _epoch_val_pm_od.append(p_cl_cd)
                _epoch_val_pf_os.append(p_cl)
                _epoch_val_pi_os.append(p_cl_img)
                _epoch_val_pm_os.append(p_cl_cd)
                _epoch_val_y.append(y_cl)

            # record val into PredictionStore
            if pred_store is not None:
                if run_single and tower_mode == "ensemble" and y_en.size:
                    pred_store.record(fold, epoch, _en_pat_ids, "od_fused", _p_en_f_od)
                    pred_store.record(fold, epoch, _en_pat_ids, "od_img",   _p_en_i_od)
                    pred_store.record(fold, epoch, _en_pat_ids, "od_md",    _p_en_m_od)
                    pred_store.record(fold, epoch, _en_pat_ids, "os_fused", _p_en_f_os)
                    pred_store.record(fold, epoch, _en_pat_ids, "os_img",   _p_en_i_os)
                    pred_store.record(fold, epoch, _en_pat_ids, "os_md",    _p_en_m_os)
                elif run_single and tower_mode == "single" and y_cl.size:
                    # val in single mode: collect_probs_single_components(aggregate_patient=False)
                    # returns interleaved [all_OD, all_OS] per batch — IDs not tracked here yet
                    pass  # single-mode val IDs not currently available; train IDs are sufficient

            # Best-epoch checks (restricted to main phase).
            target_single_auc         = cl_auc if tower_mode == "single" else en_auc
            target_holdout_single_auc = cl_auc_h if tower_mode == "single" else en_auc_h
            single_ckpt_eligible = run_single and (phase_single == "main")
            is_best_single = (
                single_ckpt_eligible
                and (not np.isnan(target_single_auc))
                and (target_single_auc > best_single_auc)
            )
            if is_best_single:
                best_single_auc   = target_single_auc
                best_epoch_single = epoch + 1
                best_single_state = copy.deepcopy(single.state_dict())
                if tower_mode == "single":
                    snap_cl, _, _, _ = _tune_and_snap(y_cl, p_cl, cl_acc, num_classes, args, args.ece_bins)
                    snap_classic = snap_cl
                else:
                    snap_en, _, _, _ = _tune_and_snap(y_en, p_en, en_acc, num_classes, args, args.ece_bins)
                    snap_ensemble = snap_en
                snap_holdout_single = {
                    "auc": float(target_holdout_single_auc),
                    "acc": float(cl_acc_h if tower_mode == "single" else en_acc_h),
                }

            is_best_holdout_single = (
                holdout_loader is not None
                and single_ckpt_eligible
                and (not np.isnan(target_holdout_single_auc))
                and (target_holdout_single_auc > best_holdout_single_auc)
            )
            if is_best_holdout_single:
                best_holdout_single_auc   = target_holdout_single_auc
                best_epoch_holdout_single = epoch + 1
                best_holdout_single_state = copy.deepcopy(single.state_dict())

            bilat_ckpt_eligible = run_bilat and (phase_bilat == "main")
            is_best_bilat = (
                bilat_ckpt_eligible
                and (not np.isnan(bi_auc))
                and (bi_auc > best_bilat_auc)
            )
            if is_best_bilat:
                best_bilat_auc   = bi_auc
                best_epoch_bilat = epoch + 1
                best_bilat_state = copy.deepcopy(bilateral.state_dict())
                snap_bi, _, _, _ = _tune_and_snap(y_bi, p_bi, bi_acc, num_classes, args, args.ece_bins)
                snap_bilat = snap_bi
                snap_holdout_bilat = {"auc": float(bi_auc_h), "acc": float(bi_acc_h)}

            is_best_holdout_bilat = (
                holdout_loader is not None
                and bilat_ckpt_eligible
                and (not np.isnan(bi_auc_h))
                and (bi_auc_h > best_holdout_bilat_auc)
            )
            if is_best_holdout_bilat:
                best_holdout_bilat_auc   = bi_auc_h
                best_epoch_holdout_bilat = epoch + 1
                best_holdout_bilat_state = copy.deepcopy(bilateral.state_dict())

            # --- confusion matrix cells per split × head -------------------
            def _prefixed_cm(prefix: str, y: np.ndarray, pf: np.ndarray,
                              pi: np.ndarray, pm: np.ndarray) -> dict:
                out: dict = {}
                for head, p in (("fused", pf), ("img", pi), ("md", pm)):
                    for k, v in _cm_cells(y, p, num_classes).items():
                        out[f"{prefix}_{head}_{k}"] = v
                return out

            cm_row: dict = {}
            cm_row.update(_prefixed_cm("classic_val",        y_cl,   p_cl,     p_cl_img,   p_cl_cd))
            cm_row.update(_prefixed_cm("ensemble_val",       y_en,   p_en,     p_en_img,   p_en_cd))
            cm_row.update(_prefixed_cm("classic_holdout",    y_cl_h, p_cl_h,   p_cl_h_img, p_cl_h_cd))
            cm_row.update(_prefixed_cm("ensemble_holdout",   y_en_h, p_en_h,   p_en_h_img, p_en_h_cd))
            cm_row.update(_prefixed_cm("train",              y_tr,   p_tr_f,   p_tr_i,     p_tr_m))

            fold_logger.write_epoch_row({
                "fold": fold, "epoch": epoch + 1,
                "phase_single": phase_single, "phase_bilat": phase_bilat,
                "main_epoch_single": main_epoch_single, "main_epoch_bilat": main_epoch_bilat,
                "single_active": int(single_active), "bilat_active": int(bilat_active),
                "single_train_loss": _f(sl_loss), "single_train_acc": _f(sl_acc),
                # val — fused
                "classic_val_auc":  _f(cl_auc), "classic_val_acc":  _f(cl_acc), "classic_val_n":  cl_n,
                "ensemble_val_auc": _f(en_auc), "ensemble_val_acc": _f(en_acc), "ensemble_val_n": en_n,
                "bilat_train_loss": _f(bl_loss), "bilat_train_acc": _f(bl_acc),
                "bilat_val_auc":    _f(bi_auc), "bilat_val_acc":    _f(bi_acc), "bilat_val_n":    bi_n,
                # val — img/md + fusion events
                "classic_val_auc_img":  _f(cl_auc_img),  "classic_val_acc_img":  _f(cl_acc_img),
                "classic_val_auc_cd":   _f(cl_auc_cd),   "classic_val_acc_cd":   _f(cl_acc_cd),
                "classic_val_fe_corr":  cl_fe_corr,       "classic_val_fe_err":   cl_fe_err,
                "ensemble_val_auc_img": _f(en_auc_img),  "ensemble_val_acc_img": _f(en_acc_img),
                "ensemble_val_auc_cd":  _f(en_auc_cd),   "ensemble_val_acc_cd":  _f(en_acc_cd),
                "ensemble_val_fe_corr": en_fe_corr,       "ensemble_val_fe_err":  en_fe_err,
                "bilat_val_auc_img":    _f(bi_auc_img),  "bilat_val_acc_img":    _f(bi_acc_img),
                "bilat_val_auc_cd":     _f(bi_auc_cd),   "bilat_val_acc_cd":     _f(bi_acc_cd),
                "bilat_val_fe_corr":    bi_fe_corr,       "bilat_val_fe_err":     bi_fe_err,
                # holdout — fused
                "classic_holdout_auc":  _f(cl_auc_h), "classic_holdout_acc":  _f(cl_acc_h),
                "ensemble_holdout_auc": _f(en_auc_h), "ensemble_holdout_acc": _f(en_acc_h),
                "bilat_holdout_auc":    _f(bi_auc_h), "bilat_holdout_acc":    _f(bi_acc_h),
                # holdout — img/md + fusion events
                "classic_holdout_auc_img":  _f(cl_auc_h_img),  "classic_holdout_acc_img":  _f(cl_acc_h_img),
                "classic_holdout_auc_cd":   _f(cl_auc_h_md),   "classic_holdout_acc_cd":   _f(cl_acc_h_md),
                "classic_holdout_fe_corr":  cl_fe_h_corr,       "classic_holdout_fe_err":   cl_fe_h_err,
                "ensemble_holdout_auc_img": _f(en_auc_h_img),  "ensemble_holdout_acc_img": _f(en_acc_h_img),
                "ensemble_holdout_auc_cd":  _f(en_auc_h_md),   "ensemble_holdout_acc_cd":  _f(en_acc_h_md),
                "ensemble_holdout_fe_corr": en_fe_h_corr,       "ensemble_holdout_fe_err":  en_fe_h_err,
                # train eval pass
                "train_auc_fused": _f(tr_auc_f), "train_acc_fused": _f(tr_acc_f),
                "train_auc_img":   _f(tr_auc_i), "train_acc_img":   _f(tr_acc_i),
                "train_auc_cd":    _f(tr_auc_m), "train_acc_cd":    _f(tr_acc_m),
                "train_fe_corr":   tr_fe_corr,    "train_fe_err":    tr_fe_err,
                "train_n":         tr_n,
                "is_best_single":         int(is_best_single),
                "is_best_bilat":          int(is_best_bilat),
                "is_best_holdout_single": int(is_best_holdout_single),
                "is_best_holdout_bilat":  int(is_best_holdout_bilat),
                **cm_row,
            }, optional_cols=epoch_fields)

            # ---- md_warmup progress bar (replaces per-epoch print) --------
            if phase_single == "cd_warmup":
                _bar_w = 30
                _filled = int(_bar_w * (epoch + 1) / single_warmup_cd)
                _bar = "#" * _filled + "-" * (_bar_w - _filled)
                _bar_msg = (
                    f"  [fold {fold+1}] md_warmup [{_bar}] "
                    f"{epoch + 1}/{single_warmup_cd}  loss={sl_loss:.4f}"
                )
                print(f"\r{_bar_msg}", end="", flush=True)
                fold_logger.info(_bar_msg)
                _prev_phase_single = phase_single
                continue  # skip normal log block entirely

            if _prev_phase_single == "cd_warmup":
                print()  # seal the progress bar line

            if args.log_every > 0 and (epoch + 1) % args.log_every == 0:
                _epoch_secs = time.time() - _epoch_t0
                hld_auc = target_holdout_single_auc if run_single else bi_auc_h
                hld_suffix = f"  hld_auc={hld_auc:.4f}" if holdout_loader is not None else ""

                # Human-readable phase progress for console logs.
                if phase_single == "tower_warmup":
                    single_phase_epoch = epoch - single_warmup_cd + 1
                    single_phase_total = single_warmup_tower
                elif phase_single == "fused_warmup":
                    single_phase_epoch = epoch - single_warmup_cd - single_warmup_tower + 1
                    single_phase_total = single_warmup_fused
                else:
                    single_phase_epoch = main_epoch_single
                    single_phase_total = main_epochs

                if phase_bilat == "tower_warmup":
                    bilat_phase_epoch = epoch + 1
                    bilat_phase_total = bilat_warmup_tower
                elif phase_bilat == "fused_warmup":
                    bilat_phase_epoch = epoch - bilat_warmup_tower + 1
                    bilat_phase_total = bilat_warmup_fused
                else:
                    bilat_phase_epoch = main_epoch_bilat
                    bilat_phase_total = main_epochs

                if run_single:
                    if tower_mode == "single":
                        msg = (
                            f"    ep {epoch+1:>3}/{total_epochs} ({_epoch_secs:.1f}s) "
                            f"[single:{phase_single} {single_phase_epoch}/{single_phase_total}] "
                            f"fused(acc={cl_acc:.4f},auc={cl_auc:.4f}) "
                            f"img(acc={cl_acc_img:.4f},auc={cl_auc_img:.4f}) "
                            f"md(acc={cl_acc_cd:.4f},auc={cl_auc_cd:.4f})  "
                            f"(best_fused={best_single_auc:.4f} @ep{best_epoch_single})"
                            f"{hld_suffix}"
                        )
                    else:
                        msg = (
                            f"    ep {epoch+1:>3}/{total_epochs} ({_epoch_secs:.1f}s) "
                            f"[single:{phase_single} {single_phase_epoch}/{single_phase_total}] "
                            f"fused(acc={en_acc:.4f},auc={en_auc:.4f}) "
                            f"img(acc={en_acc_img:.4f},auc={en_auc_img:.4f}) "
                            f"md(acc={en_acc_cd:.4f},auc={en_auc_cd:.4f})  "
                            f"(best_fused={best_single_auc:.4f} @ep{best_epoch_single})"
                            f"{hld_suffix}"
                        )
                else:
                    msg = (
                        f"    ep {epoch+1:>3}/{total_epochs} ({_epoch_secs:.1f}s) "
                        f"[bilat:{phase_bilat} {bilat_phase_epoch}/{bilat_phase_total}] "
                        f"fused(acc={bi_acc:.4f},auc={bi_auc:.4f}) "
                        f"img(acc={bi_acc_img:.4f},auc={bi_auc_img:.4f}) "
                        f"md(acc={bi_acc_cd:.4f},auc={bi_auc_cd:.4f})  "
                        f"(best_bilat={best_bilat_auc:.4f} @ep{best_epoch_bilat})"
                        f"{hld_suffix}"
                    )
                print(msg, flush=True)
                fold_logger.info(msg)

            _prev_phase_single = phase_single

        fold_logger.close()

        # Override: use final epoch state instead of best-AUC checkpoint
        if getattr(args, "use_last_epoch", False):
            best_single_state = copy.deepcopy(single.state_dict())
            if run_bilat:
                best_bilat_state = copy.deepcopy(bilat.state_dict())

        if args.save_checkpoints:
            if best_single_state is not None:
                torch.save(best_single_state, fold_dir / "best_single.pt")
            if best_bilat_state is not None:
                torch.save(best_bilat_state, fold_dir / "best_bilateral.pt")
            if best_holdout_single_state is not None:
                torch.save(best_holdout_single_state, fold_dir / "best_holdout_single.pt")
            if best_holdout_bilat_state is not None:
                torch.save(best_holdout_bilat_state, fold_dir / "best_holdout_bilateral.pt")

        # ---- Save per-epoch per-patient npy tensors -------------------------
        if _epoch_train_pf:
            # Use the order from the first epoch (consistent since loader is non-shuffled)
            ids_ref = _epoch_train_ids[0]
            y_ref   = _epoch_train_y[0]
            np.save(fold_dir / "train_patient_ids.npy", ids_ref)
            np.save(fold_dir / "train_y_true.npy",      y_ref)
            np.save(fold_dir / "train_probs_fused.npy", np.stack(_epoch_train_pf))  # (n_ep, n_pts, n_cls)
            np.save(fold_dir / "train_probs_img.npy",   np.stack(_epoch_train_pi))
            np.save(fold_dir / "train_probs_cd.npy",    np.stack(_epoch_train_pm))
        if _epoch_val_pf_od:
            np.save(fold_dir / "val_y_true_epochs.npy",         np.stack(_epoch_val_y))     # (n_ep, N)
            np.save(fold_dir / "val_probs_fused_od_epochs.npy", np.stack(_epoch_val_pf_od)) # (n_ep, N, C)
            np.save(fold_dir / "val_probs_img_od_epochs.npy",   np.stack(_epoch_val_pi_od))
            np.save(fold_dir / "val_probs_cd_od_epochs.npy",    np.stack(_epoch_val_pm_od))
            np.save(fold_dir / "val_probs_fused_os_epochs.npy", np.stack(_epoch_val_pf_os))
            np.save(fold_dir / "val_probs_img_os_epochs.npy",   np.stack(_epoch_val_pi_os))
            np.save(fold_dir / "val_probs_cd_os_epochs.npy",    np.stack(_epoch_val_pm_os))
            if _epoch_val_ids:  # only ensemble mode populates this
                np.save(fold_dir / "val_patient_ids.npy", _epoch_val_ids[0])

        # ---- Phase 2: fused head training (ensemble + --fused-head only) ----
        best_fused_auc        = -1.0
        best_fused_state: Optional[dict] = None
        best_holdout_fused_auc = -1.0

        if run_fused and best_single_state is not None:
            # Revert base to its best val checkpoint, then freeze it.
            single.load_state_dict(best_single_state)
            for p in single.parameters():
                p.requires_grad_(False)

            fused     = FusedEnsembleHT(single, num_classes).to(device)
            opt_fused = torch.optim.Adam(fused.eye_scorer.parameters(), lr=args.lr)
            fusion_epochs = int(getattr(args, "fusion_epochs", 10))

            print(
                f"  [fold {fold+1}] Phase 2: training fusion head  "
                f"bilat_train_n={len(bilat_train)}  fusion_epochs={fusion_epochs}",
                flush=True,
            )

            _val_pids_for_store = [str(s["id_1"]) for s in bilat_val]
            for fep in range(fusion_epochs):
                fu_loss, fu_acc = train_fusion_epoch(fused, train_bilat_loader, opt_fused, device)
                y_fu, p_fu = collect_probs_fused(fused, val_loader, device)
                fu_auc = _score_arrays(y_fu, p_fu, num_classes)[1]
                if pred_store is not None and y_fu.size:
                    _store_ep = total_single_epochs + fep
                    pred_store.record(fold, _store_ep, _val_pids_for_store, "bilat_fused", p_fu)

                # Holdout eval (if available)
                fu_hld_auc = nan
                if holdout_loader is not None:
                    y_fu_h, p_fu_h = collect_probs_fused(fused, holdout_loader, device)
                    fu_hld_auc = _score_arrays(y_fu_h, p_fu_h, num_classes)[1]

                is_best_fused = not np.isnan(fu_auc) and fu_auc > best_fused_auc
                if is_best_fused:
                    best_fused_auc   = fu_auc
                    best_fused_state = copy.deepcopy(fused.state_dict())
                    fu_acc_val = _score_arrays(y_fu, p_fu, num_classes)[0]
                    snap_fused, _, _, _ = _tune_and_snap(y_fu, p_fu, fu_acc_val, num_classes, args, args.ece_bins)

                is_best_hld_fused = not np.isnan(fu_hld_auc) and fu_hld_auc > best_holdout_fused_auc
                if is_best_hld_fused:
                    best_holdout_fused_auc = fu_hld_auc
                    snap_holdout_fused = {"auc": fu_hld_auc, "acc": _score_arrays(y_fu_h, p_fu_h, num_classes)[0]}

                if (fep + 1) % max(1, getattr(args, "log_every", 1)) == 0:
                    print(
                        f"  [fold {fold+1}] fusion ep{fep+1:>3}  "
                        f"loss={fu_loss:.4f}  train_acc={fu_acc:.4f}  "
                        f"val_auc={fu_auc:.4f}  hld_auc={fu_hld_auc:.4f}"
                        f"{'  *' if is_best_fused else ''}",
                        flush=True,
                    )

            if best_fused_state is not None:
                if args.save_checkpoints:
                    torch.save(best_fused_state, fold_dir / "best_fused.pt")
                print(
                    f"  [fold {fold+1}] BEST  "
                    f"fused_head(acc={snap_fused.get('acc', nan):.4f},"
                    f"auc={snap_fused.get('auc', nan):.4f}) "
                    f"kappa={snap_fused.get('kappa', nan):.4f} "
                    f"F1={snap_fused.get('macro_f1', nan):.4f} "
                    f"ECE={snap_fused.get('ece', nan):.4f}  "
                    f"holdout_auc={snap_holdout_fused.get('auc', nan):.4f}",
                    flush=True,
                )

        if run_single:
            if tower_mode == "single":
                print(
                    f"  [fold {fold+1}] BEST  "
                    f"fused(acc={snap_classic.get('acc', nan):.4f},auc={snap_classic.get('auc', nan):.4f}) "
                    f"kappa={snap_classic.get('kappa', nan):.4f} "
                    f"F1={snap_classic.get('macro_f1', nan):.4f} "
                    f"ECE={snap_classic.get('ece', nan):.4f} @ep{best_epoch_single}",
                    flush=True,
                )
            else:
                print(
                    f"  [fold {fold+1}] BEST  "
                    f"ensemble(acc={snap_ensemble.get('acc', nan):.4f},auc={snap_ensemble.get('auc', nan):.4f}) "
                    f"kappa={snap_ensemble.get('kappa', nan):.4f} "
                    f"F1={snap_ensemble.get('macro_f1', nan):.4f} "
                    f"ECE={snap_ensemble.get('ece', nan):.4f} @ep{best_epoch_single}",
                    flush=True,
                )
        else:
            print(
                f"  [fold {fold+1}] BEST  "
                f"fused(acc={snap_bilat.get('acc', nan):.4f},auc={snap_bilat.get('auc', nan):.4f}) "
                f"kappa={snap_bilat.get('kappa', nan):.4f} "
                f"F1={snap_bilat.get('macro_f1', nan):.4f} "
                f"ECE={snap_bilat.get('ece', nan):.4f} @ep{best_epoch_bilat}",
                flush=True,
            )

        # Export best-epoch prediction artifacts.
        if run_single and best_single_state is not None:
            single.load_state_dict(best_single_state)
        if run_bilat and best_bilat_state is not None:
            bilateral.load_state_dict(best_bilat_state)
        y_en_pe_best = p_en_pe_best = p_en_pe_best_img = p_en_pe_best_md = None
        l_en_best = l_en_best_img = l_en_best_md = None
        l_cl_best = l_cl_best_img = l_cl_best_md = None
        l_en_pe_best = l_en_pe_best_img = l_en_pe_best_md = None
        if run_single and tower_mode == "single":
            y_cl_best, p_cl_best, p_cl_best_img, p_cl_best_md, \
                l_cl_best, l_cl_best_img, l_cl_best_md = collect_probs_single_components(
                single, val_loader, device, aggregate_patient=False, return_logits=True
            )
            y_en_best = p_en_best = p_en_best_img = p_en_best_md = None
        elif run_single and tower_mode == "ensemble":
            y_en_best, p_en_best, p_en_best_img, p_en_best_md, \
                l_en_best, l_en_best_img, l_en_best_md = collect_probs_single_components(
                single, val_loader, device, aggregate_patient=True, return_logits=True
            )
            y_en_pe_best, p_en_pe_best, p_en_pe_best_img, p_en_pe_best_md, \
                l_en_pe_best, l_en_pe_best_img, l_en_pe_best_md = collect_probs_single_components(
                single, val_loader, device, aggregate_patient=False, return_logits=True
            )
            y_cl_best = p_cl_best = p_cl_best_img = p_cl_best_md = None
        else:
            y_cl_best = y_en_best = None
            p_cl_best = p_en_best = p_en_best_img = p_en_best_md = None
            p_cl_best_img = p_cl_best_md = None
        if run_bilat:
            y_bi_best, p_bi_best = collect_probs_bilateral(bilateral, val_loader, device)
        else:
            y_bi_best = p_bi_best = None

        y_fu_best = p_fu_best = None
        if run_fused and best_fused_state is not None:
            fused.load_state_dict(best_fused_state)
            y_fu_best, p_fu_best = collect_probs_fused(fused, val_loader, device)

        return FoldResult(
            mode=mode, fold=fold,
            best_epoch_single=best_epoch_single, best_epoch_bilat=best_epoch_bilat,
            classic_val_auc=snap_classic.get("auc", nan),
            classic_val_acc=snap_classic.get("acc", nan),
            classic_val_kappa=snap_classic.get("kappa", nan),
            classic_val_mcc=snap_classic.get("mcc", nan),
            classic_val_f1=snap_classic.get("macro_f1", nan),
            classic_val_recall=_sv(snap_classic.get("per_class_recall")),
            classic_val_ece=snap_classic.get("ece", nan),
            classic_val_threshold=snap_classic.get("threshold", nan),
            classic_val_bias=_svf(snap_classic.get("bias")),
            classic_val_n=snap_classic.get("n", 0),
            ensemble_val_auc=snap_ensemble.get("auc", nan),
            ensemble_val_acc=snap_ensemble.get("acc", nan),
            ensemble_val_kappa=snap_ensemble.get("kappa", nan),
            ensemble_val_mcc=snap_ensemble.get("mcc", nan),
            ensemble_val_f1=snap_ensemble.get("macro_f1", nan),
            ensemble_val_recall=_sv(snap_ensemble.get("per_class_recall")),
            ensemble_val_ece=snap_ensemble.get("ece", nan),
            ensemble_val_threshold=snap_ensemble.get("threshold", nan),
            ensemble_val_bias=_svf(snap_ensemble.get("bias")),
            ensemble_val_n=snap_ensemble.get("n", 0),
            bilat_val_auc=snap_bilat.get("auc", nan),
            bilat_val_acc=snap_bilat.get("acc", nan),
            bilat_val_kappa=snap_bilat.get("kappa", nan),
            bilat_val_mcc=snap_bilat.get("mcc", nan),
            bilat_val_f1=snap_bilat.get("macro_f1", nan),
            bilat_val_recall=_sv(snap_bilat.get("per_class_recall")),
            bilat_val_ece=snap_bilat.get("ece", nan),
            bilat_val_threshold=snap_bilat.get("threshold", nan),
            bilat_val_bias=_svf(snap_bilat.get("bias")),
            bilat_val_n=snap_bilat.get("n", 0),
            classic_holdout_auc=snap_holdout_single.get("auc", nan)  if tower_mode == "single"   else nan,
            classic_holdout_acc=snap_holdout_single.get("acc", nan)  if tower_mode == "single"   else nan,
            ensemble_holdout_auc=snap_holdout_single.get("auc", nan) if tower_mode == "ensemble" else nan,
            ensemble_holdout_acc=snap_holdout_single.get("acc", nan) if tower_mode == "ensemble" else nan,
            bilat_holdout_auc=snap_holdout_bilat.get("auc", nan),
            bilat_holdout_acc=snap_holdout_bilat.get("acc", nan),
            holdout_n=len(holdout_bilat),
            single_train_n=len(eye_train),
            bilat_train_n=len(bilat_train),
            fused_val_auc=snap_fused.get("auc", nan),
            fused_val_acc=snap_fused.get("acc", nan),
            fused_val_kappa=snap_fused.get("kappa", nan),
            fused_val_mcc=snap_fused.get("mcc", nan),
            fused_val_f1=snap_fused.get("macro_f1", nan),
            fused_val_recall=_sv(snap_fused.get("per_class_recall")),
            fused_val_ece=snap_fused.get("ece", nan),
            fused_val_threshold=snap_fused.get("threshold", nan),
            fused_val_bias=_svf(snap_fused.get("bias")),
            fused_val_n=snap_fused.get("n", 0),
            fused_holdout_auc=snap_holdout_fused.get("auc", nan),
            fused_holdout_acc=snap_holdout_fused.get("acc", nan),
        ), FoldArtifacts(
            y_true_classic=y_cl_best,  probs_classic=p_cl_best,
            y_true_ensemble=y_en_best, probs_ensemble=p_en_best,
            y_true_bilat=y_bi_best,    probs_bilat=p_bi_best,
            y_true_fused=y_fu_best,    probs_fused=p_fu_best,
            probs_ensemble_img=p_en_best_img,
            probs_ensemble_md=p_en_best_md,
            probs_classic_img=p_cl_best_img,
            probs_classic_md=p_cl_best_md,
            y_true_ensemble_pereye=y_en_pe_best,
            probs_ensemble_pereye=p_en_pe_best,
            probs_ensemble_img_pereye=p_en_pe_best_img,
            probs_ensemble_md_pereye=p_en_pe_best_md,
            logits_ensemble=l_en_best,
            logits_ensemble_img=l_en_best_img,
            logits_ensemble_md=l_en_best_md,
            logits_classic=l_cl_best,
            logits_classic_img=l_cl_best_img,
            logits_classic_md=l_cl_best_md,
            logits_ensemble_pereye=l_en_pe_best,
            logits_ensemble_img_pereye=l_en_pe_best_img,
            logits_ensemble_md_pereye=l_en_pe_best_md,
        )

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _summary(results: list[FoldResult]) -> dict:
        def _ms(vals):
            v = np.array(
                [x for x in vals if x is not None and not np.isnan(float(x))], dtype=float
            )
            return (float(np.mean(v)) if v.size else None, float(np.std(v)) if v.size else None)

        out = {}
        for label, prefix in [
            ("classic_best_val",  "classic_val"),
            ("ensemble_best_val", "ensemble_val"),
            ("bilat_best_val",    "bilat_val"),
            ("fused_best_val",    "fused_val"),
        ]:
            sub = {}
            for m in ["auc", "acc", "kappa", "mcc", "f1", "ece", "threshold"]:
                vals = [getattr(r, f"{prefix}_{m}") for r in results]
                mean, std = _ms(vals)
                sub[f"{m}_mean"] = mean
                if m in ("auc", "f1", "kappa"):
                    sub[f"{m}_std"] = std
            out[label] = sub

        for label, prefix in [
            ("classic_holdout",  "classic_holdout"),
            ("ensemble_holdout", "ensemble_holdout"),
            ("bilat_holdout",    "bilat_holdout"),
            ("fused_holdout",    "fused_holdout"),
        ]:
            sub = {}
            for m in ["auc", "acc"]:
                vals = [getattr(r, f"{prefix}_{m}") for r in results]
                mean, std = _ms(vals)
                sub[f"{m}_mean"] = mean
                if m == "auc":
                    sub[f"{m}_std"] = std
            out[label] = sub

        for delta_label, prefix_a, prefix_b in [
            ("delta_ensemble_vs_classic", "classic_val",  "ensemble_val"),
            ("delta_bilat_vs_ensemble",   "ensemble_val", "bilat_val"),
            ("delta_fused_vs_ensemble",   "ensemble_val", "fused_val"),
        ]:
            delta = {}
            for m in ["auc", "f1", "kappa"]:
                pairs = [
                    getattr(r, f"{prefix_b}_{m}") - getattr(r, f"{prefix_a}_{m}")
                    for r in results
                    if not np.isnan(float(getattr(r, f"{prefix_a}_{m}")))
                    and not np.isnan(float(getattr(r, f"{prefix_b}_{m}")))
                ]
                delta[f"{m}_mean"] = float(np.mean(pairs)) if pairs else None
                delta[f"{m}_std"]  = float(np.std(pairs))  if pairs else None
            out[delta_label] = delta

        out["n_folds_completed"] = len(results)
        out["single_train_mode"] = "eye-level (all OD+OS samples)"
        out["bilat_train_mode"]  = "patient-level (bilateral only)"
        out["eval_note"] = (
            "classic=eye-level SingleEyeHT; "
            "ensemble=patient-level SingleEyeHT (OD+OS averaged); "
            "bilateral=patient-level BilateralHT; "
            "fused=ensemble base + learned logit-level fusion head"
        )
        return out

    @staticmethod
    def _print_summary(mode: str, s: dict, tower_mode: str | None = None) -> None:
        def f(v):
            return "nan" if v is None else f"{v:.4f}"

        cv = s["classic_best_val"]
        ev = s["ensemble_best_val"]
        bv = s["bilat_best_val"]
        fv = s["fused_best_val"]
        d1 = s["delta_ensemble_vs_classic"]
        d2 = s["delta_bilat_vs_ensemble"]
        d3 = s["delta_fused_vs_ensemble"]
        has_fused = fv["auc_mean"] is not None

        print(f"\n=== Summary [{mode}] — best-epoch val ===")
        print(f"  {'':26s}  {'AUC':>8}  {'ACC':>8}  {'Kappa':>8}  {'F1-mac':>8}  {'ECE':>8}")
        if tower_mode == "single":
            rows = [("single   (eye-lvl  eval)", cv)]
        elif tower_mode == "ensemble":
            rows = [("ensemble (pat-lvl  eval)", ev)]
            if has_fused:
                rows.append(("fused_head(pat-lvl eval)", fv))
        elif tower_mode == "bilateral":
            rows = [("bilateral (bilat   eval)", bv)]
        else:
            rows = [
                ("classic  (eye-lvl  eval)", cv),
                ("ensemble (pat-lvl  eval)", ev),
                ("bilateral (bilat   eval)", bv),
            ]
            if has_fused:
                rows.append(("fused_head(pat-lvl eval)", fv))
        for label, d in rows:
            print(
                f"  {label:26s}  "
                f"{f(d['auc_mean']):>8}  {f(d['acc_mean']):>8}  "
                f"{f(d['kappa_mean']):>8}  {f(d['f1_mean']):>8}  {f(d['ece_mean']):>8}"
            )
        if tower_mode is None:
            print(
                f"  {'Δ ensemble−classic':26s}  "
                f"{f(d1['auc_mean']):>8}  {'':>8}  "
                f"{f(d1['kappa_mean']):>8}  {f(d1['f1_mean']):>8}"
            )
            print(
                f"  {'Δ bilateral−ensemble':26s}  "
                f"{f(d2['auc_mean']):>8}  {'':>8}  "
                f"{f(d2['kappa_mean']):>8}  {f(d2['f1_mean']):>8}"
            )
        if has_fused and tower_mode in ("ensemble", None):
            print(
                f"  {'Δ fused−ensemble':26s}  "
                f"{f(d3['auc_mean']):>8}  {'':>8}  "
                f"{f(d3['kappa_mean']):>8}  {f(d3['f1_mean']):>8}"
            )


# ---------------------------------------------------------------------------
# V2ModeComparator — thin backward-compat shim
# ---------------------------------------------------------------------------

class V2ModeComparator:
    """Backward-compat shim used by run_multifold_v2_modes.py."""

    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        return V2HyperTower.build_parser()

    @staticmethod
    def run(cli_args=None) -> Path:
        args = V2HyperTower.build_parser().parse_args(cli_args)
        return V2HyperTower(args).run()


# ---------------------------------------------------------------------------
# Module-level aliases (kept for backward compat; use V2HyperTower directly)
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    return V2HyperTower.build_parser()


def run_mode(args) -> Path:
    return V2HyperTower(args).run()
