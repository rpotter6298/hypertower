"""V3HyperTower — central orchestrator for the V3 pipeline.

Key differences from V2:
  - Proper outer/inner k-fold CV: test = current fold, val = next fold, train = rest.
    No pre-carved holdout — every patient appears in test exactly once.
  - No checkpoint saving (.pt files). Model states are kept in memory only.
  - Test set evaluated each main-phase epoch (logged to epoch CSV only, never used for model selection).
    Final test metrics reported in summary use the last-epoch model state.
  - Binary-focused defaults (multiclass still supported via --eval-mode multiclass).
  - tune_binary_threshold uses Youden's J by default (class-distribution independent).
  - tune_multiclass_bias maximises balanced accuracy (class-distribution independent).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch

from v3.classes.croppers import (
    ManifestImageCropper,
    UNetImageCropper,
    build_image_preprocessor_from_args,
)
from v3.classes.image_loader import CachedImageLoader
from v3.classes.loader_factory import (
    build_balanced_sampler,
    filter_bilateral_samples,
    filter_eye_samples,
    make_loader,
)
from v3.classes.metrics import _score_arrays, _svf, _tune_and_snap
from v3.classes.bridges import Bridge
from v3.classes.towerbase import train_towers_epoch, collect_probs_towers
from v3.classes.image_towers import ImageTower
from v3.classes.clinical_towers import ClinicalDataTower
from v3.classes.geometry_towers import GeometryTower
from v3.classes.hypertower_models import (
    BilateralHT,
    EmbeddingMLPEnsembleHT,
    FusedEnsembleHT,
    LogitMLPEnsembleHT,
    SiameseHT,
    SingleEyeHT,
    V2ModeComparisonOps,
    collect_probs_bilateral,
    collect_probs_siamese,
    collect_probs_bilateral_components,
    collect_probs_classic,
    collect_probs_ensemble,
    collect_probs_ensemble_pereye,
    collect_probs_eye_level,
    collect_probs_fused,
    collect_probs_single_components,
    train_bilateral_epoch,
    train_siamese_epoch,
    train_fusion_epoch,
    train_single_epoch,
)
from v3.classes.papila_builders import build_papila_data
from v3.classes.predictions import PredictionStore, head_names_for_mode
from v3.classes.profiles import build_papila_profile
from v3.classes.results import FoldArtifacts, FoldResult, _f, _nan, _sv
from v3.classes.split_manager import EyeLevelSplitManager, PatientFirstSplitManager
from v3.classes.transforms import build_eval_transform
from v3.classes.utils import (
    _drop_mixed_label_patients,
    _relabel_mixed_patients_to_max,
    choose_device,
    seed_everything,
)
from v3.classes.hypertower_logger import HypertowerLogger


# ---------------------------------------------------------------------------
# Helpers (unchanged from V2)
# ---------------------------------------------------------------------------

def _fusion_events(y, pf, pi, pm):
    pred_f = pf.argmax(1); pred_i = pi.argmax(1); pred_m = pm.argmax(1)
    corr = int(((pred_f == y) & (pred_i != y) & (pred_m != y)).sum())
    err  = int(((pred_f != y) & (pred_i == y) & (pred_m == y)).sum())
    return corr, err


def _cm_cells(y, p, num_classes):
    if not y.size or p.ndim < 2 or p.shape[1] != num_classes:
        return {}
    pred = p.argmax(1)
    if num_classes == 2:
        return {
            "tn": int(((pred==0)&(y==0)).sum()), "fp": int(((pred==1)&(y==0)).sum()),
            "fn": int(((pred==0)&(y==1)).sum()), "tp": int(((pred==1)&(y==1)).sum()),
        }
    out = {}
    for i in range(num_classes):
        for j in range(num_classes):
            out[f"cm_{i}_{j}"] = int(((y==i)&(pred==j)).sum())
    return out


def _save_predictions_csv(fold_dir, eval_mode, y_true, heads, suffix=""):
    N = len(y_true)
    num_classes = next(p.shape[1] for p in heads.values() if p is not None)
    rows = []
    for i in range(N):
        true = int(y_true[i])
        row = {"idx": i, "y_true": true}
        for head_name, probs in heads.items():
            if probs is None:
                continue
            pred = int(probs[i].argmax())
            row[f"pred_{head_name}"] = pred
            for c in range(num_classes):
                row[f"prob_{head_name}_c{c}"] = float(probs[i, c])
            if eval_mode == "binary":
                row[f"tp_{head_name}"] = int(pred==1 and true==1)
                row[f"fp_{head_name}"] = int(pred==1 and true==0)
                row[f"tn_{head_name}"] = int(pred==0 and true==0)
                row[f"fn_{head_name}"] = int(pred==0 and true==1)
            else:
                row[f"correct_{head_name}"] = int(pred==true)
        rows.append(row)
    if not rows:
        return
    path = fold_dir / f"predictions{suffix}.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# V3HyperTower
# ---------------------------------------------------------------------------

class V3HyperTower:
    """V3 orchestrator. Construct with ``V3HyperTower(args)``, call ``.run()``."""

    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        ap = argparse.ArgumentParser(
            description=(
                "V3 HyperTower — outer/inner k-fold CV (test=current fold). "
                "No pre-carved holdout. No checkpoint saving."
            )
        )
        ap.add_argument("--image-dir",    default="Papila/FundusImages")
        ap.add_argument("--clinical-dir", default="Papila/ClinicalData")
        ap.add_argument("--label-col",    default="Diagnosis")
        ap.add_argument("--cat-cols",     nargs="*", default=["Gender", "Phakic/Pseudophakic"])
        ap.add_argument("--exclude-cols", nargs="*", default=[])
        ap.add_argument("--eval-mode",    choices=["binary", "multiclass"], default="binary")
        ap.add_argument(
            "--hypertower-mode", choices=["single", "ensemble", "bilateral", "siamese", "classic"],
            default="ensemble",
        )
        ap.add_argument("--n-splits",  type=int, default=5)
        ap.add_argument("--fold-seed", type=int, default=42)
        ap.add_argument("--leaky-cv",  action="store_true",
                        help="Split at eye level (leaky: same patient can span folds). "
                             "Used to demonstrate data-leakage effect.")
        ap.add_argument(
            "--folds", type=int, default=None,
            help="Optional cap on number of folds to run.",
        )
        ap.add_argument("--epochs",               type=int,   default=40)
        ap.add_argument("--warmup-tower-epochs",  type=int,   default=None)
        ap.add_argument("--warmup-fused-epochs",  type=int,   default=None)
        ap.add_argument("--single-warmup-tower-epochs", type=int, default=3)
        ap.add_argument("--single-warmup-fused-epochs", type=int, default=3)
        ap.add_argument("--warmup-cd-epochs",     type=int,   default=40)
        ap.add_argument("--bilat-warmup-tower-epochs", type=int, default=None)
        ap.add_argument("--bilat-warmup-fused-epochs", type=int, default=None)
        ap.add_argument("--batch-size",           type=int,   default=8)
        ap.add_argument("--lr",                   type=float, default=1e-4)
        ap.add_argument("--bcd-prob",             type=float, default=0.5)
        ap.add_argument("--tower-loss-mode", choices=["bcd", "all"], default="bcd")
        ap.add_argument("--backbone",             default="refugelike")
        ap.add_argument("--freeze-ratio",         type=float, default=0.0)
        ap.add_argument("--augment",              action="store_true")
        ap.add_argument("--balanced-sampling",    action="store_true")
        ap.add_argument("--num-workers",          type=int,   default=8)
        ap.add_argument("--in-memory-cache",      action="store_true", default=True)
        ap.add_argument("--no-in-memory-cache",   action="store_false", dest="in_memory_cache")
        ap.add_argument("--cache-workers",        type=int,   default=4)
        ap.add_argument("--device",  choices=["auto", "cpu", "cuda"], default="auto")
        ap.add_argument("--seed",    type=int,   default=1234)
        ap.add_argument("--run-name", default=None)
        ap.add_argument("--output-root", default="analysis_data")
        # ROI cropping
        ap.add_argument("--img-crop-manifest",   type=str,   default=None)
        ap.add_argument("--img-crop-gt",         action="store_true")
        ap.add_argument("--img-crop-weights",    type=str,   default=None)
        ap.add_argument("--img-crop-normalize",  type=str,   default="per_image",
                        choices=["per_image", "imagenet"])
        ap.add_argument("--img-crop-threshold",  type=float, default=0.5)
        ap.add_argument("--img-crop-tta",        action="store_true")
        ap.add_argument("--img-crop-scale",      type=float, default=2.5)
        ap.add_argument("--img-crop-size",       type=int,   default=224)
        ap.add_argument("--img-crop-cache",      type=str,   default="cache_data/hypertower_crops")
        ap.add_argument("--persist-img-crop-cache", action="store_true")
        # Architecture
        ap.add_argument("--cd-hidden-dim",   type=int,   default=128)
        ap.add_argument("--fusion-dim",      type=int,   default=256)
        ap.add_argument("--bridge-mode",     default="fused",
                        choices=["fused", "image_only", "clinical_only"])
        ap.add_argument("--bridge-dropout",  type=float, default=0.5,
                        help="Dropout in bridge classifier_fused (default: 0.5)")
        ap.add_argument("--cd-dropout",      type=float, default=0.1,
                        help="Dropout in clinical tower MLP (default: 0.1)")
        ap.add_argument("--se-img-tower",    action="store_true",
                        help="Enable SE gate on image tower output features")
        ap.add_argument("--se-cd-tower",     action="store_true",
                        help="Enable SE gate on clinical tower output features")
        ap.add_argument("--se-bridge",       action="store_true",
                        help="Enable SE gate on fused vector inside the bridge")
        # Mixed patients
        ap.add_argument("--exclude-mixed-patients",      dest="exclude_mixed_patients",
                        action="store_true")
        ap.add_argument("--include-mixed-patients",      dest="exclude_mixed_patients",
                        action="store_false")
        ap.add_argument("--relabel-mixed-patients-to-max", dest="relabel_mixed_patients_to_max",
                        action="store_true")
        ap.add_argument("--keep-mixed-raw-labels",       dest="relabel_mixed_patients_to_max",
                        action="store_false", help=argparse.SUPPRESS)
        ap.set_defaults(exclude_mixed_patients=False, relabel_mixed_patients_to_max=False)
        # Tuning
        ap.add_argument("--tune-binary-threshold", action="store_true")
        ap.add_argument("--tune-multiclass-bias",  action="store_true")
        ap.add_argument("--ece-bins",   type=int, default=10)
        ap.add_argument("--log-every",  type=int, default=1)
        # IOP
        ap.add_argument("--iop-corr-method", choices=["ratio", "ols", "lad", "multi"],
                        default="ratio")
        ap.add_argument("--iop-drop-raw", action="store_true", default=False)
        # Fused head
        ap.add_argument("--fused-head",     action="store_true")
        ap.add_argument("--fusion-epochs",  type=int, default=10)
        ap.add_argument("--head-type",
                        choices=["attention", "logit_mlp", "embedding_mlp"],
                        default="attention",
                        help="Which bilateral head to train on top of frozen ensemble base")
        ap.add_argument("--save-checkpoints", action="store_true",
                        help="Save best_single.pt per fold for explainability / GradCAM")
        # Geometry features
        ap.add_argument("--geometry-dim", type=int, default=0,
                        help="Append N geometry features to clinical metadata (0=disabled, 5=all). "
                             "Requires --img-crop-manifest.")
        ap.add_argument("--geometry-source", default="gt", choices=["gt", "unet"],
                        help="Source for geometry features: gt (GT contour annotations) or "
                             "unet (U-Net segmentation). unet also requires --img-crop-weights.")
        ap.add_argument("--geometry-tower", action="store_true",
                        help="Add a dedicated GeometryTower (disc/cup seg-map CNN) fused via the "
                             "bridge alongside ImageTower and ClinicalDataTower. Requires "
                             "--img-crop-manifest.")
        ap.add_argument("--geometry-tower-backbone", default="resnet18",
                        choices=["resnet18", "resnet50", "efficientnet_b0"],
                        help="SegCNN backbone for GeometryTower (default: resnet18).")
        ap.add_argument("--geometry-tower-in-channels", type=int, default=3, choices=[1, 3],
                        help="1 = single label map; 3 = one-hot disc/rim/cup (default: 3).")
        ap.add_argument("--geometry-tower-frozen", action="store_true",
                        help="Freeze GeometryTower backbone throughout training.")
        ap.add_argument("--geometry-tower-finetune-unet-epochs", type=int, default=0,
                        help="Epochs to fine-tune the U-Net per fold before seg-map extraction "
                             "(0 = disabled; only applies when --geometry-source unet).")
        return ap

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

        # Build geometry provider if requested, extend feature_dim to include geometry.
        # Both ManifestImageCropper and UNetImageCropper already have geometry_features()
        # and precompute_geometry() — we just pick the right one and pre-compute upfront.
        self.geometry_provider = None
        geom_dim = int(getattr(args, "geometry_dim", 0))
        if geom_dim > 0:
            source = getattr(args, "geometry_source", "gt")
            manifest = getattr(args, "img_crop_manifest", None)
            if not manifest:
                raise ValueError("--geometry-dim requires --img-crop-manifest")
            all_paths = [
                self.data.get_image_path(row)
                for _, row in self.data.df.iterrows()
            ]
            if source == "gt":
                # Reuse image_preprocessor if it's already a ManifestImageCropper,
                # otherwise build a lightweight one just for geometry (no crop cache).
                if isinstance(self.image_preprocessor, ManifestImageCropper):
                    provider = self.image_preprocessor
                else:
                    provider = ManifestImageCropper(manifest_path=Path(manifest))
                print(f"[geometry] GT source — pre-computing geometry from {manifest}", flush=True)
            elif source == "unet":
                weights = getattr(args, "img_crop_weights", None)
                if not weights:
                    raise ValueError("--geometry-source unet requires --img-crop-weights")
                if isinstance(self.image_preprocessor, UNetImageCropper):
                    provider = self.image_preprocessor
                else:
                    provider = UNetImageCropper(
                        manifest_path=Path(manifest),
                        weights_path=Path(weights),
                        normalize=getattr(args, "img_crop_normalize", "per_image"),
                        threshold=getattr(args, "img_crop_threshold", 0.5),
                    )
                print(f"[geometry] UNet source — pre-computing geometry from {weights}", flush=True)
            else:
                raise ValueError(f"Unknown --geometry-source: {source!r}")
            provider.precompute_geometry(all_paths)
            self.geometry_provider = provider
            self.data.feature_dim += geom_dim
            print(f"[geometry] feature_dim extended to {self.data.feature_dim} (+{geom_dim} geometry)", flush=True)

    def run(self) -> Path:
        """Execute the full fold loop."""
        args = self.args
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_name = args.run_name or f"v3_hypertower_{ts}"
        out_dir = Path(args.output_root) / run_name
        out_dir.mkdir(parents=True, exist_ok=True)

        mode       = args.eval_mode
        tower_mode = "single" if args.hypertower_mode == "classic" else args.hypertower_mode
        df_mode    = self.data.df.copy()

        if args.exclude_mixed_patients:
            before = df_mode["Patient ID"].nunique()
            df_mode, mixed = _drop_mixed_label_patients(
                df_mode, patient_col="Patient ID", label_col=args.label_col
            )
            print(f"[{mode}] dropped {len(mixed)} mixed-label patients ({before}→{df_mode['Patient ID'].nunique()})", flush=True)
        elif args.relabel_mixed_patients_to_max:
            df_mode, changed, still_mixed = _relabel_mixed_patients_to_max(
                df_mode, patient_col="Patient ID", label_col=args.label_col
            )
            print(f"[{mode}] relabeled {changed} mixed-patient rows to max severity", flush=True)

        if mode == "binary":
            df_mode = df_mode[df_mode[args.label_col].isin([0, 1])].reset_index(drop=True)

        num_classes = 2 if mode == "binary" else int(df_mode[args.label_col].nunique())
        print(f"\n[{mode}] num_classes={num_classes}  rows={len(df_mode)}  patients={df_mode['Patient ID'].nunique()}", flush=True)

        if getattr(args, "leaky_cv", False):
            split_manager = EyeLevelSplitManager(patient_col="Patient ID", label_col=args.label_col)
            print("[CV] WARNING: leaky-cv mode — eye-level splits, same patient can span folds.", flush=True)
        else:
            split_manager = PatientFirstSplitManager(patient_col="Patient ID", label_col=args.label_col)
        split_args = SimpleNamespace(
            eval_mode=mode,
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

        profile_eye     = build_papila_profile(patient_col="Patient ID", label_col=args.label_col, sample_mode="eye")
        profile_patient = build_papila_profile(patient_col="Patient ID", label_col=args.label_col, sample_mode="patient")

        fused_head    = getattr(args, "fused_head", False)
        _head_names   = head_names_for_mode(tower_mode, fused_head=fused_head)
        fusion_epochs = int(getattr(args, "fusion_epochs", 10)) if fused_head else 0
        _warmup_tower = int(getattr(args, "single_warmup_tower_epochs", None) or getattr(args, "warmup_tower_epochs", None) or 2)
        _warmup_fused = int(getattr(args, "single_warmup_fused_epochs", None) or getattr(args, "warmup_fused_epochs", None) or 2)
        _warmup_cd    = int(getattr(args, "warmup_cd_epochs", 0))
        _total_epochs = _warmup_cd + _warmup_tower + _warmup_fused + int(args.epochs) + fusion_epochs

        if tower_mode in ("single", "classic"):
            _sample_ids = [f"{row['Patient ID']}{row['eyeID']}" for _, row in df_mode.iterrows()]
            _y_true     = df_mode[args.label_col].tolist()
        else:
            _pat_df     = df_mode.drop_duplicates(subset="Patient ID")
            _sample_ids = _pat_df["Patient ID"].astype(str).tolist()
            _y_true     = _pat_df[args.label_col].tolist()

        pred_store = PredictionStore(
            sample_ids=_sample_ids, y_true=_y_true,
            head_names=_head_names, n_folds=n_folds,
            n_epochs=_total_epochs, n_classes=num_classes,
        )

        image_cache = CachedImageLoader(
            enabled=getattr(args, "in_memory_cache", False),
            workers=int(getattr(args, "cache_workers", 4)),
        )

        for fold in range(n_folds):
            seed_everything(args.seed + fold * 100)
            fold_dir = tm_dir / f"fold{fold}"
            fold_dir.mkdir(exist_ok=True)
            print(f"\n[{mode}:{tower_mode}] fold {fold+1}/{n_folds}", flush=True)
            result, artifacts = self._run_fold(
                fold=fold, split=plans[fold], mode=mode, data=self.data,
                num_classes=num_classes, profile_eye=profile_eye,
                profile_patient=profile_patient, fold_dir=fold_dir,
                tower_mode=tower_mode, pred_store=pred_store, image_cache=image_cache,
            )
            fold_results.append(result)

            # Save val artifacts
            if artifacts.y_true_ensemble is not None:
                np.save(fold_dir / "y_true.npy",    artifacts.y_true_ensemble)
            if artifacts.probs_ensemble is not None:
                np.save(fold_dir / "probs_fused.npy", artifacts.probs_ensemble)
            if artifacts.probs_ensemble_img is not None:
                np.save(fold_dir / "probs_img.npy", artifacts.probs_ensemble_img)
            if artifacts.probs_ensemble_md is not None:
                np.save(fold_dir / "probs_cd.npy",  artifacts.probs_ensemble_md)
            if artifacts.y_true_classic is not None:
                np.save(fold_dir / "y_true.npy",     artifacts.y_true_classic)
            if artifacts.probs_classic is not None:
                np.save(fold_dir / "probs_classic.npy", artifacts.probs_classic)
            if artifacts.y_true_ensemble_pereye is not None:
                np.save(fold_dir / "y_true_pereye.npy", artifacts.y_true_ensemble_pereye)
            if artifacts.probs_ensemble_pereye is not None:
                np.save(fold_dir / "probs_fused_pereye.npy", artifacts.probs_ensemble_pereye)
            if artifacts.logits_ensemble is not None:
                np.save(fold_dir / "logits_fused.npy", artifacts.logits_ensemble)
            if artifacts.logits_ensemble_img is not None:
                np.save(fold_dir / "logits_img.npy", artifacts.logits_ensemble_img)
            if artifacts.logits_ensemble_md is not None:
                np.save(fold_dir / "logits_cd.npy",  artifacts.logits_ensemble_md)

            # Save test artifacts
            if artifacts.y_true_test is not None:
                np.save(fold_dir / "test_y_true.npy", artifacts.y_true_test)
            if artifacts.probs_test is not None:
                np.save(fold_dir / "test_probs_fused.npy", artifacts.probs_test)
            if artifacts.probs_test_img is not None:
                np.save(fold_dir / "test_probs_img.npy", artifacts.probs_test_img)
            if artifacts.probs_test_md is not None:
                np.save(fold_dir / "test_probs_cd.npy", artifacts.probs_test_md)
            if artifacts.y_true_fused_test is not None:
                np.save(fold_dir / "test_y_true_fused.npy", artifacts.y_true_fused_test)
            if artifacts.probs_fused_test is not None:
                np.save(fold_dir / "test_probs_fused_head.npy", artifacts.probs_fused_test)

        if pred_store is not None:
            pred_store.save(tm_dir / "predictions.npz")

        # Summary
        summary = self._summary(fold_results)
        self._print_summary(mode, summary, tower_mode=tower_mode)

        # Save fold results CSV
        import dataclasses
        fold_csv = tm_dir / "fold_results.csv"
        rows = [dataclasses.asdict(r) for r in fold_results]
        if rows:
            with fold_csv.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

        # Save summary JSON
        if tower_mode in ("single", "classic"):
            _test_key = "classic_test"
        elif tower_mode == "ensemble":
            _test_key = "fused_test" if fused_head else "ensemble_test"
        elif tower_mode in ("bilateral", "siamese"):
            _test_key = "bilat_test"
        else:
            _test_key = "classic_test"
        mode_summary = {_test_key: summary.get(_test_key, {})}
        if fused_head and tower_mode == "ensemble":
            mode_summary["ensemble_test"] = summary.get("ensemble_test", {})
            mode_summary["fused_best_val"] = summary.get("fused_best_val", {})
        with (tm_dir / "summary.json").open("w") as f:
            json.dump({"mode_summary": mode_summary}, f, indent=2, default=str)

        return out_dir

    def _run_fold_towers(
        self,
        *,
        fold: int,
        split,
        mode: str,
        data,
        num_classes: int,
        profile_eye,
        profile_patient,
        fold_dir: Path,
        pred_store,
        image_cache,
    ):
        """Modular TowerBase training path (used when --geometry-tower is set).

        Builds [ImageTower, ClinicalDataTower, GeometryTower], runs the full
        fold lifecycle (prepare_fold → augment_samples → loader build → epoch
        loop → eval), and returns (FoldResult, FoldArtifacts) with metrics in
        the ensemble_val_* slots.
        """
        args   = self.args
        device = self.device
        nan    = float("nan")

        # ------------------------------------------------------------------ samples
        eye_train   = filter_eye_samples(profile_eye.build_samples(df=split.train, clinical=data))
        bilat_train = filter_bilateral_samples(profile_patient.build_samples(df=split.train, clinical=data))
        bilat_val   = filter_bilateral_samples(profile_patient.build_samples(df=split.val,   clinical=data))
        bilat_test  = filter_bilateral_samples(profile_patient.build_samples(
            df=split.test, clinical=data)) if split.test is not None else []

        # Old --geometry-dim path still applies (injects geometry into clinical stream)
        if self.geometry_provider is not None:
            eye_train   = self._augment_geometry(eye_train)
            bilat_train = self._augment_geometry(bilat_train)
            bilat_val   = self._augment_geometry(bilat_val)
            bilat_test  = self._augment_geometry(bilat_test)

        if len(bilat_val) == 0:
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
                single_train_n=len(eye_train), bilat_train_n=len(bilat_train),
            )
            return empty, FoldArtifacts(
                y_true_classic=None, probs_classic=None,
                y_true_ensemble=None, probs_ensemble=None,
                y_true_bilat=None, probs_bilat=None,
            )

        # ------------------------------------------------------------------ towers
        img_tower = ImageTower(
            backbone=args.backbone,
            freeze_ratio=args.freeze_ratio,
            augment=args.augment,
            use_se=getattr(args, "se_img_tower", False),
        )
        cd_tower = ClinicalDataTower(
            clinical_data=data,
            cd_hidden_dim=args.cd_hidden_dim,
            cd_dropout=getattr(args, "cd_dropout", 0.1),
            use_se=getattr(args, "se_cd_tower", False),
        )

        geom_tower = GeometryTower(
            backbone=getattr(args, "geometry_tower_backbone", "resnet18"),
            in_channels=getattr(args, "geometry_tower_in_channels", 3),
            pretrained=not getattr(args, "no_pretrained", False),
            frozen=getattr(args, "geometry_tower_frozen", False),
            geometry_source=getattr(args, "geometry_source", "gt"),
            manifest_path=getattr(args, "img_crop_manifest", None),
            weights_path=getattr(args, "img_crop_weights", None),
            unet_normalize=getattr(args, "img_crop_normalize", "per_image"),
            unet_threshold=getattr(args, "img_crop_threshold", 0.5),
            finetune_unet_epochs=getattr(args, "geometry_tower_finetune_unet_epochs", 0),
        )

        # GeometryTower.prepare_fold must run before augment_samples (precomputes seg maps)
        geom_tower.prepare_fold(
            eye_train=eye_train, bilat_train=bilat_train,
            bilat_val=bilat_val, bilat_test=bilat_test,
            image_preprocessor=self.image_preprocessor,
            image_cache=image_cache, device=device, args=args,
        )
        # Inject seg_map_1/seg_map_2 into all sample lists before loaders are built
        for sample_list in (eye_train, bilat_train, bilat_val, bilat_test):
            geom_tower.augment_samples(sample_list)

        # Now ImageTower.prepare_fold sees augmented samples → loader includes seg maps
        img_tower.prepare_fold(
            eye_train=eye_train, bilat_train=bilat_train,
            bilat_val=bilat_val, bilat_test=bilat_test,
            image_preprocessor=self.image_preprocessor,
            image_cache=image_cache, device=device, args=args,
        )
        cd_tower.prepare_fold(
            eye_train=eye_train, bilat_train=bilat_train,
            bilat_val=bilat_val, bilat_test=bilat_test,
            image_preprocessor=self.image_preprocessor,
            image_cache=image_cache, device=device, args=args,
        )

        towers = [img_tower, cd_tower, geom_tower]

        # ------------------------------------------------------------------ bridge
        tower_dims = []
        for t in towers:
            tower_dims.extend(t.embed_dims)
        bridge = Bridge(
            tower_dims=tower_dims,
            num_classes=num_classes,
            fusion_dim=args.fusion_dim,
            mode=getattr(args, "bridge_mode", "fused"),
            dropout=getattr(args, "bridge_dropout", 0.5),
        )

        # Move all nn.Modules to device
        for t in towers:
            if isinstance(t, torch.nn.Module):
                t.to(device)
        bridge.to(device)

        # ------------------------------------------------------------------ loaders
        slots_patient = profile_patient.slot_descriptors()
        _persistent = args.num_workers > 0
        loader_kw = dict(
            batch_size=args.batch_size, num_workers=args.num_workers,
            image_cache=image_cache, persistent_workers=_persistent,
        )
        eval_transform = build_eval_transform(args.backbone)
        val_loader = make_loader(
            bilat_val, slots_patient,
            image_transform=eval_transform,
            image_preprocessor=self.image_preprocessor,
            shuffle=False, **loader_kw,
        )
        test_loader = None
        if bilat_test:
            test_loader = make_loader(
                bilat_test, slots_patient,
                image_transform=eval_transform,
                image_preprocessor=self.image_preprocessor,
                shuffle=False, **loader_kw,
            )

        train_loader = img_tower.train_loader
        val_loader.dataset.prebuild_image_cache()
        if test_loader is not None:
            test_loader.dataset.prebuild_image_cache()

        # ------------------------------------------------------------------ optimizer
        all_params = list(bridge.parameters())
        for t in towers:
            if isinstance(t, torch.nn.Module):
                all_params.extend(t.parameters())
        optimizer = torch.optim.AdamW(
            [p for p in all_params if p.requires_grad],
            lr=args.lr,
            weight_decay=getattr(args, "weight_decay", 1e-4),
        )

        # ------------------------------------------------------------------ epoch loop
        global_warmup_tower = getattr(args, "warmup_tower_epochs", None)
        global_warmup_fused = getattr(args, "warmup_fused_epochs", None)
        warmup_cd    = int(getattr(args, "warmup_cd_epochs", 0))
        warmup_tower = int(getattr(args, "single_warmup_tower_epochs", None) or global_warmup_tower or 2)
        warmup_fused = int(getattr(args, "single_warmup_fused_epochs", None) or global_warmup_fused or 2)
        main_epochs  = int(args.epochs)

        schedule = []
        if warmup_cd    > 0: schedule.append(("cd_warmup",    warmup_cd))
        if warmup_tower > 0: schedule.append(("tower_warmup", warmup_tower))
        if warmup_fused > 0: schedule.append(("fused_warmup", warmup_fused))
        schedule.append(("main", main_epochs))

        best_val_auc      = float("-inf")
        best_epoch        = 0
        best_tower_states = None
        best_bridge_state = None
        epoch_idx         = 0

        for phase, n_epochs in schedule:
            for _ in range(n_epochs):
                for t in towers:
                    if isinstance(t, torch.nn.Module):
                        t.train()
                train_towers_epoch(
                    towers, bridge, train_loader, optimizer, device,
                    phase=phase,
                    bcd_prob=getattr(args, "bcd_prob", 0.5),
                    tower_loss_mode=getattr(args, "tower_loss_mode", "bcd"),
                )
                y_v, p_v = collect_probs_towers(towers, bridge, val_loader, device,
                                                tower_mode="ensemble")
                _, val_auc, _ = _score_arrays(y_v, p_v, num_classes)
                if not np.isnan(val_auc) and val_auc > best_val_auc:
                    best_val_auc      = val_auc
                    best_epoch        = epoch_idx
                    best_tower_states = [
                        t.state_dict() if isinstance(t, torch.nn.Module) else None
                        for t in towers
                    ]
                    best_bridge_state = bridge.state_dict()
                epoch_idx += 1

        # Restore best
        if best_bridge_state is not None:
            bridge.load_state_dict(best_bridge_state)
        if best_tower_states is not None:
            for t, st in zip(towers, best_tower_states):
                if isinstance(t, torch.nn.Module) and st is not None:
                    t.load_state_dict(st)

        # ------------------------------------------------------------------ eval
        y_val, p_val = collect_probs_towers(towers, bridge, val_loader, device, tower_mode="ensemble")
        acc_val, auc_val, n_val = _score_arrays(y_val, p_val, num_classes)
        snap_val, _, thr_val, bias_val = _tune_and_snap(
            y_val, p_val, acc_val, num_classes, args, n_bins=10
        )

        y_test = p_test = None
        test_auc = test_acc = nan
        test_n = 0
        if test_loader is not None:
            y_test, p_test = collect_probs_towers(towers, bridge, test_loader, device,
                                                  tower_mode="ensemble")
            test_acc, test_auc, test_n = _score_arrays(y_test, p_test, num_classes)

        result = FoldResult(
            mode=mode, fold=fold,
            best_epoch_single=best_epoch, best_epoch_bilat=0,
            classic_val_auc=nan,  classic_val_acc=nan,  classic_val_kappa=nan,
            classic_val_mcc=nan,  classic_val_f1=nan,   classic_val_recall=None,
            classic_val_ece=nan,  classic_val_threshold=nan, classic_val_bias=None,
            classic_val_n=0,
            ensemble_val_auc=snap_val["auc"],  ensemble_val_acc=snap_val["acc"],
            ensemble_val_kappa=snap_val["kappa"], ensemble_val_mcc=snap_val["mcc"],
            ensemble_val_f1=snap_val["macro_f1"],
            ensemble_val_recall=_sv(snap_val["per_class_recall"]),
            ensemble_val_ece=snap_val["ece"],
            ensemble_val_threshold=snap_val["threshold"],
            ensemble_val_bias=_svf(bias_val),
            ensemble_val_n=snap_val["n"],
            bilat_val_auc=nan,  bilat_val_acc=nan,  bilat_val_kappa=nan,
            bilat_val_mcc=nan,  bilat_val_f1=nan,   bilat_val_recall=None,
            bilat_val_ece=nan,  bilat_val_threshold=nan, bilat_val_bias=None,
            bilat_val_n=0,
            ensemble_test_auc=test_auc, ensemble_test_acc=test_acc,
            test_n=test_n,
            single_train_n=len(eye_train), bilat_train_n=len(bilat_train),
        )
        artifacts = FoldArtifacts(
            y_true_classic=None,   probs_classic=None,
            y_true_ensemble=y_val, probs_ensemble=p_val,
            y_true_bilat=None,     probs_bilat=None,
            y_true_test=y_test,    probs_test=p_test,
        )
        return result, artifacts

    def _augment_geometry_slot(self, samples: list) -> list:
        """Add geom_1/geom_2 keys to each sample dict (geometry tower mode).

        Unlike _augment_geometry, this does NOT touch matrix_1/matrix_2 — the
        geometry vector lives in its own slot so ImageTower and ClinicalDataTower
        each receive only their own modality.
        """
        if self.geometry_provider is None:
            return samples
        geom_dim = int(getattr(self.args, "geometry_dim", 0)) or 5
        for s in samples:
            for img_slot, geom_slot in (("image_1", "geom_1"), ("image_2", "geom_2")):
                img_path = s.get(img_slot)
                if img_path is None:
                    continue
                vec = self.geometry_provider.geometry_for_image(img_path)
                if vec is not None and len(vec) >= geom_dim:
                    s[geom_slot] = vec[:geom_dim].astype(np.float32)
                else:
                    s[geom_slot] = np.zeros(geom_dim, dtype=np.float32)
        return samples

    def _augment_geometry(self, samples: list) -> list:
        """Append geometry features to matrix_1/matrix_2 in each sample dict."""
        if self.geometry_provider is None:
            return samples
        geom_dim = int(getattr(self.args, "geometry_dim", 0))
        for s in samples:
            for img_slot, mat_slot in (("image_1", "matrix_1"), ("image_2", "matrix_2")):
                img_path = s.get(img_slot)
                mat = s.get(mat_slot)
                if img_path is None or mat is None:
                    continue
                vec = self.geometry_provider.geometry_for_image(img_path)
                if vec is not None and len(vec) >= geom_dim:
                    geom = vec[:geom_dim].astype(np.float32)
                else:
                    geom = np.zeros(geom_dim, dtype=np.float32)
                s[mat_slot] = np.concatenate([np.asarray(mat, dtype=np.float32), geom])
        return samples

    def _run_fold(
        self,
        *,
        fold: int,
        split,
        mode: str,
        data,
        num_classes: int,
        profile_eye,
        profile_patient,
        fold_dir: Path,
        tower_mode: str,
        pred_store,
        image_cache,
    ):
        args = self.args

        # Modular tower path — bypasses the legacy single/bilat/siamese code entirely
        if getattr(args, "geometry_tower", False):
            return self._run_fold_towers(
                fold=fold, split=split, mode=mode, data=data,
                num_classes=num_classes, profile_eye=profile_eye,
                profile_patient=profile_patient, fold_dir=fold_dir,
                pred_store=pred_store, image_cache=image_cache,
            )

        device = self.device
        image_preprocessor = self.image_preprocessor
        nan = float("nan")

        run_single  = tower_mode in ("single", "ensemble")
        run_bilat   = tower_mode == "bilateral"
        run_siamese = tower_mode == "siamese"
        run_fused   = tower_mode == "ensemble" and getattr(args, "fused_head", False)

        # ---- warmup schedule -------------------------------------------
        global_warmup_tower = getattr(args, "warmup_tower_epochs", None)
        global_warmup_fused = getattr(args, "warmup_fused_epochs", None)
        single_warmup_tower = int(
            getattr(args, "single_warmup_tower_epochs", None) or global_warmup_tower or 2
        )
        single_warmup_fused = int(
            getattr(args, "single_warmup_fused_epochs", None) or global_warmup_fused or 2
        )
        bilat_warmup_tower = int(
            getattr(args, "bilat_warmup_tower_epochs", None) or global_warmup_tower or 3
        )
        bilat_warmup_fused = int(
            getattr(args, "bilat_warmup_fused_epochs", None) or global_warmup_fused or 3
        )
        single_warmup_cd = int(getattr(args, "warmup_cd_epochs", 0)) if run_single else 0
        if not run_single:
            single_warmup_tower = single_warmup_fused = 0
        if not run_bilat and not run_siamese:
            bilat_warmup_tower = bilat_warmup_fused = 0
        # Warmup is meaningless in single-pathway modes — skip it entirely
        _bridge_mode = getattr(args, "bridge_mode", "fused")
        if _bridge_mode in ("image_only", "clinical_only"):
            single_warmup_cd = single_warmup_tower = single_warmup_fused = 0
            bilat_warmup_tower = bilat_warmup_fused = 0
        main_epochs = int(args.epochs)
        total_single_epochs = (single_warmup_cd + single_warmup_tower + single_warmup_fused + main_epochs) if run_single else 0
        total_bilat_epochs  = (bilat_warmup_tower + bilat_warmup_fused + main_epochs) if (run_bilat or run_siamese) else 0
        total_epochs = max(total_single_epochs, total_bilat_epochs)

        # ---- samples ---------------------------------------------------
        eye_train   = filter_eye_samples(profile_eye.build_samples(df=split.train, clinical=data))
        bilat_train = filter_bilateral_samples(profile_patient.build_samples(df=split.train, clinical=data))
        bilat_val   = filter_bilateral_samples(profile_patient.build_samples(df=split.val,   clinical=data))
        bilat_test  = filter_bilateral_samples(profile_patient.build_samples(df=split.test,  clinical=data)) if split.test is not None else []

        if self.geometry_provider is not None:
            eye_train   = self._augment_geometry(eye_train)
            bilat_train = self._augment_geometry(bilat_train)
            bilat_val   = self._augment_geometry(bilat_val)
            bilat_test  = self._augment_geometry(bilat_test)

        if pred_store is not None:
            if tower_mode in ("single", "classic"):
                train_sids = [f"{s['id_1']}{s.get('eye_id_1','')}" for s in eye_train]
                val_sids   = [f"{s['id_1']}{s.get('eye_id_1','')}" for s in bilat_val]
            else:
                train_sids = [str(s["id_1"]) for s in bilat_train]
                val_sids   = [str(s["id_1"]) for s in bilat_val]
            pred_store.set_split(fold, train_sids, "train")
            pred_store.set_split(fold, val_sids,   "val")
            if bilat_test:
                test_sids = [str(s["id_1"]) for s in bilat_test]
                pred_store.set_split(fold, test_sids, "test")

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
                ensemble_test_auc=nan, ensemble_test_acc=nan,
                classic_test_auc=nan,  classic_test_acc=nan,
                bilat_test_auc=nan,    bilat_test_acc=nan,
                test_n=0,
                single_train_n=len(eye_train), bilat_train_n=len(bilat_train),
            )
            return empty, FoldArtifacts(
                y_true_classic=None, probs_classic=None,
                y_true_ensemble=None, probs_ensemble=None,
                y_true_bilat=None, probs_bilat=None,
            )

        # ---- models (CPU for now — moved to device after workers spawn) ---
        single = None
        bilateral = None
        siamese = None
        if run_single:
            single = SingleEyeHT(
                backbone=args.backbone, freeze_ratio=args.freeze_ratio,
                augment=args.augment, clinical_data=data, num_classes=num_classes,
                cd_hidden_dim=args.cd_hidden_dim, fusion_dim=args.fusion_dim,
                bridge_mode=getattr(args, "bridge_mode", "fused"),
                bridge_dropout=getattr(args, "bridge_dropout", 0.5),
                cd_dropout=getattr(args, "cd_dropout", 0.1),
                se_img_tower=getattr(args, "se_img_tower", False),
                se_cd_tower=getattr(args, "se_cd_tower", False),
                se_bridge=getattr(args, "se_bridge", False),
            )
        if run_bilat:
            bilateral = BilateralHT(
                backbone=args.backbone, freeze_ratio=args.freeze_ratio,
                augment=args.augment, clinical_data=data, num_classes=num_classes,
                cd_hidden_dim=args.cd_hidden_dim, fusion_dim=args.fusion_dim,
            )
        if run_siamese:
            siamese = SiameseHT(
                backbone=args.backbone, freeze_ratio=args.freeze_ratio,
                augment=args.augment, num_classes=num_classes,
                fusion_dim=args.fusion_dim,
            )

        slots_eye     = profile_eye.slot_descriptors()
        slots_patient = profile_patient.slot_descriptors()
        _persistent_workers = args.num_workers > 0
        loader_kw     = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                             image_cache=image_cache,
                             persistent_workers=_persistent_workers)

        # ---- loaders ---------------------------------------------------
        use_balanced = bool(getattr(args, "balanced_sampling", False))
        train_single_loader = train_eval_loader = train_bilat_loader = cd_only_loader = None

        if run_single:
            single_sampler = build_balanced_sampler(eye_train) if use_balanced else None
            train_single_loader = make_loader(
                eye_train, slots_eye, image_transform=single.transform,
                image_preprocessor=image_preprocessor, shuffle=True,
                sampler=single_sampler, **loader_kw,
            )
            train_eval_loader = make_loader(
                eye_train, slots_eye, image_transform=build_eval_transform(args.backbone),
                image_preprocessor=image_preprocessor, shuffle=False, **loader_kw,
            )
            if single_warmup_cd > 0:
                slots_cd_only = {k: v for k, v in slots_eye.items() if k != "image_1"}
                md_sampler = single_sampler if single_sampler is not None else build_balanced_sampler(eye_train)
                cd_only_loader = make_loader(
                    eye_train, slots_cd_only, image_transform=None,
                    image_preprocessor=None, shuffle=True, sampler=md_sampler, **loader_kw,
                )
        if run_bilat:
            bilat_sampler = build_balanced_sampler(bilat_train) if use_balanced else None
            train_bilat_loader = make_loader(
                bilat_train, slots_patient, image_transform=bilateral.transform,
                image_preprocessor=image_preprocessor, shuffle=True,
                sampler=bilat_sampler, **loader_kw,
            )
        elif run_siamese:
            siamese_sampler = build_balanced_sampler(bilat_train) if use_balanced else None
            train_bilat_loader = make_loader(
                bilat_train, slots_patient, image_transform=siamese.transform,
                image_preprocessor=image_preprocessor, shuffle=True,
                sampler=siamese_sampler, **loader_kw,
            )
        elif run_fused:
            fused_sampler = build_balanced_sampler(bilat_train) if use_balanced else None
            train_bilat_loader = make_loader(
                bilat_train, slots_patient, image_transform=single.transform,
                image_preprocessor=image_preprocessor, shuffle=True,
                sampler=fused_sampler, **loader_kw,
            )

        eval_transform = build_eval_transform(args.backbone)
        val_loader = make_loader(
            bilat_val, slots_patient, image_transform=eval_transform,
            image_preprocessor=image_preprocessor, shuffle=False, **loader_kw,
        )

        # ---- test loader (never touched during training) ---------------
        test_loader = None
        if bilat_test:
            test_loader = make_loader(
                bilat_test, slots_patient, image_transform=eval_transform,
                image_preprocessor=image_preprocessor, shuffle=False, **loader_kw,
            )
            print(f"  [fold {fold+1}] test_n={len(bilat_test)} (bilateral patients)", flush=True)

        # ---- prebuild image cache --------------------------------------
        for _ldr in [train_single_loader, train_bilat_loader, val_loader, test_loader]:
            if _ldr is not None:
                _ldr.dataset.prebuild_image_cache()

        # ---- spawn DataLoader workers BEFORE CUDA init -----------------
        # Workers fork here (clean process state, no CUDA context yet).
        # persistent_workers=True keeps them alive so the training loop
        # reuses them rather than re-forking after .to(device).
        if _persistent_workers:
            for _ldr in [train_single_loader, train_bilat_loader, val_loader, test_loader]:
                if _ldr is not None:
                    _ = iter(_ldr)  # triggers fork now, before CUDA

        # ---- move models to device (CUDA init happens here) ------------
        if single is not None:
            single = single.to(device)
        if bilateral is not None:
            bilateral = bilateral.to(device)
        if siamese is not None:
            siamese = siamese.to(device)

        opt_single    = torch.optim.Adam(single.parameters(),    lr=args.lr) if run_single   else None
        opt_bilateral = torch.optim.Adam(bilateral.parameters(), lr=args.lr) if run_bilat    else None
        opt_siamese   = torch.optim.Adam(siamese.parameters(),   lr=args.lr) if run_siamese  else None

        # ---- epoch log -------------------------------------------------
        epoch_fields = [
            "fold", "epoch", "phase_single", "phase_bilat",
            "main_epoch_single", "main_epoch_bilat",
            "single_active", "bilat_active",
            "single_train_loss", "single_train_acc",
            "classic_val_auc",      "classic_val_acc",      "classic_val_n",
            "ensemble_val_auc",     "ensemble_val_acc",     "ensemble_val_n",
            "bilat_train_loss",     "bilat_train_acc",
            "bilat_val_auc",        "bilat_val_acc",        "bilat_val_n",
            "classic_val_auc_img",  "classic_val_acc_img",
            "classic_val_auc_cd",   "classic_val_acc_cd",
            "classic_val_fe_corr",  "classic_val_fe_err",
            "ensemble_val_auc_img", "ensemble_val_acc_img",
            "ensemble_val_auc_cd",  "ensemble_val_acc_cd",
            "ensemble_val_fe_corr", "ensemble_val_fe_err",
            "bilat_val_auc_img",    "bilat_val_acc_img",
            "bilat_val_auc_cd",     "bilat_val_acc_cd",
            "bilat_val_fe_corr",    "bilat_val_fe_err",
            "train_auc_fused",  "train_acc_fused",
            "train_auc_img",    "train_acc_img",
            "train_auc_cd",     "train_acc_cd",
            "train_fe_corr",    "train_fe_err",
            "train_n",
            "is_best_single", "is_best_bilat",
        ]
        if num_classes == 2:
            _cm_keys = ["tn", "fp", "fn", "tp"]
        else:
            _cm_keys = [f"cm_{i}_{j}" for i in range(num_classes) for j in range(num_classes)]
        for _split in ("classic_val", "ensemble_val", "train"):
            for _head in ("fused", "img", "md"):
                for _k in _cm_keys:
                    epoch_fields.append(f"{_split}_{_head}_{_k}")
        fold_logger = HypertowerLogger(run_dir=fold_dir)

        # Per-epoch accumulators
        _epoch_train_pf: list[np.ndarray] = []
        _epoch_train_pi: list[np.ndarray] = []
        _epoch_train_pm: list[np.ndarray] = []
        _epoch_train_ids: list[np.ndarray] = []
        _epoch_train_y:   list[np.ndarray] = []
        _epoch_val_pf_od: list[np.ndarray] = []
        _epoch_val_pi_od: list[np.ndarray] = []
        _epoch_val_pm_od: list[np.ndarray] = []
        _epoch_val_pf_os: list[np.ndarray] = []
        _epoch_val_pi_os: list[np.ndarray] = []
        _epoch_val_pm_os: list[np.ndarray] = []
        _epoch_val_y:     list[np.ndarray] = []
        _epoch_val_ids:   list[np.ndarray] = []

        snap_classic:  dict = {}
        snap_ensemble: dict = {}
        snap_bilat:    dict = {}
        snap_fused:    dict = {}

        if run_single:
            print(
                f"  [fold {fold+1}]  single_train_n={len(eye_train)}  val_n={len(bilat_val)}  "
                f"test_n={len(bilat_test)}  "
                f"warmup=md{single_warmup_cd}+twr{single_warmup_tower}+fus{single_warmup_fused}  total={total_single_epochs}",
                flush=True,
            )
        else:
            print(
                f"  [fold {fold+1}]  bilat_train_n={len(bilat_train)}  val_n={len(bilat_val)}  "
                f"test_n={len(bilat_test)}  "
                f"bilat_warmup={bilat_warmup_tower}+{bilat_warmup_fused}  total={total_bilat_epochs}",
                flush=True,
            )

        _prev_phase_single = "inactive"

        # ================================================================
        # EPOCH LOOP — no test evaluation during training
        # ================================================================
        for epoch in range(total_epochs):
            _epoch_t0 = time.time()

            # Phase logic
            if not run_single:
                phase_single, main_epoch_single, single_active = "inactive", 0, False
            elif epoch < single_warmup_cd:
                phase_single, main_epoch_single, single_active = "cd_warmup", 0, True
            elif epoch < single_warmup_cd + single_warmup_tower:
                phase_single, main_epoch_single, single_active = "tower_warmup", 0, True
            elif epoch < single_warmup_cd + single_warmup_tower + single_warmup_fused:
                phase_single, main_epoch_single, single_active = "fused_warmup", 0, True
            elif epoch < total_single_epochs:
                phase_single = "main"
                main_epoch_single = epoch - single_warmup_cd - single_warmup_tower - single_warmup_fused + 1
                single_active = True
            else:
                phase_single, main_epoch_single, single_active = "done", main_epochs, False

            if not run_bilat and not run_siamese:
                phase_bilat, main_epoch_bilat, bilat_active = "inactive", 0, False
            elif epoch < bilat_warmup_tower:
                phase_bilat, main_epoch_bilat, bilat_active = "tower_warmup", 0, True
            elif epoch < bilat_warmup_tower + bilat_warmup_fused:
                phase_bilat, main_epoch_bilat, bilat_active = "fused_warmup", 0, True
            elif epoch < total_bilat_epochs:
                phase_bilat = "main"
                main_epoch_bilat = epoch - bilat_warmup_tower - bilat_warmup_fused + 1
                bilat_active = True
            else:
                phase_bilat, main_epoch_bilat, bilat_active = "done", main_epochs, False

            # Training steps
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
            elif run_siamese and bilat_active:
                bl_loss, bl_acc = train_siamese_epoch(
                    siamese, train_bilat_loader, opt_siamese, device,
                    bcd_prob=float(args.bcd_prob),
                    tower_loss_mode=args.tower_loss_mode,
                )
            else:
                bl_loss, bl_acc = nan, nan

            _skip_val_eval = (phase_single == "cd_warmup")

            # Val evaluation
            if run_single and tower_mode == "single" and not _skip_val_eval:
                y_cl, p_cl, p_cl_img, p_cl_cd = collect_probs_single_components(
                    single, val_loader, device, aggregate_patient=False
                )
                cl_acc, cl_auc, cl_n = _score_arrays(y_cl, p_cl, num_classes)
                cl_acc_img = float((p_cl_img.argmax(1)==y_cl).mean()) if y_cl.size else nan
                cl_acc_cd  = float((p_cl_cd.argmax(1) ==y_cl).mean()) if y_cl.size else nan
                _, cl_auc_img, _ = _score_arrays(y_cl, p_cl_img, num_classes)
                _, cl_auc_cd, _  = _score_arrays(y_cl, p_cl_cd,  num_classes)
                y_en = np.array([], dtype=np.int64)
                p_en = p_en_img = p_en_cd = np.zeros((0, num_classes), dtype=np.float32)
                en_acc = en_auc = nan; en_n = 0
                en_acc_img = en_acc_cd = en_auc_img = en_auc_cd = nan
            elif run_single and tower_mode == "ensemble" and not _skip_val_eval:
                (y_en, _p_en_f_od, _p_en_i_od, _p_en_m_od,
                 _p_en_f_os, _p_en_i_os, _p_en_m_os,
                 _en_pat_ids) = collect_probs_ensemble_pereye(
                    single, val_loader, device, return_ids=True
                )
                p_en     = 0.5 * (_p_en_f_od + _p_en_f_os)
                p_en_img = 0.5 * (_p_en_i_od + _p_en_i_os)
                p_en_cd  = 0.5 * (_p_en_m_od + _p_en_m_os)
                en_acc, en_auc, en_n = _score_arrays(y_en, p_en, num_classes)
                en_acc_img = float((p_en_img.argmax(1)==y_en).mean()) if y_en.size else nan
                en_acc_cd  = float((p_en_cd.argmax(1) ==y_en).mean()) if y_en.size else nan
                _, en_auc_img, _ = _score_arrays(y_en, p_en_img, num_classes)
                _, en_auc_cd, _  = _score_arrays(y_en, p_en_cd,  num_classes)
                y_cl = np.array([], dtype=np.int64)
                p_cl = p_cl_img = p_cl_cd = np.zeros((0, num_classes), dtype=np.float32)
                cl_acc = cl_auc = nan; cl_n = 0
                cl_acc_img = cl_acc_cd = cl_auc_img = cl_auc_cd = nan
            else:
                y_cl = y_en = np.array([], dtype=np.int64)
                p_cl = p_cl_img = p_cl_cd = np.zeros((0, num_classes), dtype=np.float32)
                p_en = p_en_img = p_en_cd = np.zeros((0, num_classes), dtype=np.float32)
                cl_acc = cl_auc = en_acc = en_auc = nan; cl_n = en_n = 0
                cl_acc_img = cl_acc_cd = en_acc_img = en_acc_cd = nan
                cl_auc_img = cl_auc_cd = en_auc_img = en_auc_cd = nan

            if run_bilat and not _skip_val_eval:
                y_bi, p_bi, p_bi_img, p_bi_cd = collect_probs_bilateral_components(
                    bilateral, val_loader, device
                )
                bi_acc, bi_auc, bi_n = _score_arrays(y_bi, p_bi, num_classes)
                bi_acc_img = float((p_bi_img.argmax(1)==y_bi).mean()) if y_bi.size else nan
                bi_acc_cd  = float((p_bi_cd.argmax(1) ==y_bi).mean()) if y_bi.size else nan
                _, bi_auc_img, _ = _score_arrays(y_bi, p_bi_img, num_classes)
                _, bi_auc_cd, _  = _score_arrays(y_bi, p_bi_cd,  num_classes)
            elif run_siamese and not _skip_val_eval:
                y_bi, p_bi = collect_probs_siamese(siamese, val_loader, device)
                bi_acc, bi_auc, bi_n = _score_arrays(y_bi, p_bi, num_classes)
                bi_acc_img = bi_acc_cd = bi_auc_img = bi_auc_cd = nan
            else:
                y_bi = np.array([], dtype=np.int64)
                p_bi = np.zeros((0, 0), dtype=np.float32)
                bi_acc = bi_auc = nan; bi_n = 0
                bi_acc_img = bi_acc_cd = bi_auc_img = bi_auc_cd = nan

            # Fusion events
            cl_fe_corr, cl_fe_err = _fusion_events(y_cl, p_cl, p_cl_img, p_cl_cd) if y_cl.size else (0, 0)
            en_fe_corr, en_fe_err = _fusion_events(y_en, p_en, p_en_img, p_en_cd) if y_en.size else (0, 0)
            bi_fe_corr, bi_fe_err = (0, 0)

            # Train eval pass
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
                    tr_acc_f = float((p_tr_f.argmax(1)==y_tr).mean())
                    _, tr_auc_i, _ = _score_arrays(y_tr, p_tr_i, num_classes)
                    tr_acc_i = float((p_tr_i.argmax(1)==y_tr).mean())
                    _, tr_auc_m, _ = _score_arrays(y_tr, p_tr_m, num_classes)
                    tr_acc_m = float((p_tr_m.argmax(1)==y_tr).mean())
                    tr_fe_corr, tr_fe_err = _fusion_events(y_tr, p_tr_f, p_tr_i, p_tr_m)
                    tr_n = int(y_tr.size)
                    _epoch_train_pf.append(p_tr_f)
                    _epoch_train_pi.append(p_tr_i)
                    _epoch_train_pm.append(p_tr_m)
                    _epoch_train_ids.append(tr_ids)
                    _epoch_train_y.append(y_tr)
                    if pred_store is not None:
                        if tower_mode in ("single", "classic"):
                            pred_store.record(fold, epoch, tr_ids, "fused", p_tr_f)
                            pred_store.record(fold, epoch, tr_ids, "img",   p_tr_i)
                            pred_store.record(fold, epoch, tr_ids, "md",    p_tr_m)
                        else:
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

            # Accumulate val per-epoch npy
            if run_single and tower_mode == "ensemble" and y_en.size:
                _epoch_val_pf_od.append(_p_en_f_od); _epoch_val_pi_od.append(_p_en_i_od)
                _epoch_val_pm_od.append(_p_en_m_od); _epoch_val_pf_os.append(_p_en_f_os)
                _epoch_val_pi_os.append(_p_en_i_os); _epoch_val_pm_os.append(_p_en_m_os)
                _epoch_val_y.append(y_en); _epoch_val_ids.append(_en_pat_ids)
            elif run_single and tower_mode == "single" and y_cl.size:
                _epoch_val_pf_od.append(p_cl); _epoch_val_pi_od.append(p_cl_img)
                _epoch_val_pm_od.append(p_cl_cd); _epoch_val_pf_os.append(p_cl)
                _epoch_val_pi_os.append(p_cl_img); _epoch_val_pm_os.append(p_cl_cd)
                _epoch_val_y.append(y_cl)

            # Val PredictionStore
            if pred_store is not None:
                if run_single and tower_mode == "ensemble" and y_en.size:
                    pred_store.record(fold, epoch, _en_pat_ids, "od_fused", _p_en_f_od)
                    pred_store.record(fold, epoch, _en_pat_ids, "od_img",   _p_en_i_od)
                    pred_store.record(fold, epoch, _en_pat_ids, "od_md",    _p_en_m_od)
                    pred_store.record(fold, epoch, _en_pat_ids, "os_fused", _p_en_f_os)
                    pred_store.record(fold, epoch, _en_pat_ids, "os_img",   _p_en_i_os)
                    pred_store.record(fold, epoch, _en_pat_ids, "os_md",    _p_en_m_os)

            is_best_single = False
            is_best_bilat  = False

            # Per-epoch test eval (logged only, never used for model selection)
            te_auc = te_acc = float("nan")
            if test_loader is not None and phase_single == "main":
                if run_single and tower_mode == "single":
                    _yte, _pte, _, _ = collect_probs_single_components(
                        single, test_loader, device, aggregate_patient=False)
                elif run_single and tower_mode == "ensemble":
                    _yte, _pte, _, _ = collect_probs_single_components(
                        single, test_loader, device, aggregate_patient=True)
                elif run_bilat:
                    _yte, _pte, _, _ = collect_probs_bilateral_components(
                        bilateral, test_loader, device)
                else:
                    _yte = _pte = None
                if _yte is not None and _yte.size and len(np.unique(_yte)) > 1:
                    te_auc = float(_score_arrays(_yte, _pte, num_classes)[1])
                    te_acc = float((_pte.argmax(1) == _yte).mean())

            # Confusion matrix cells
            def _prefixed_cm(prefix, y, pf, pi, pm):
                out = {}
                for head, p in (("fused", pf), ("img", pi), ("md", pm)):
                    for k, v in _cm_cells(y, p, num_classes).items():
                        out[f"{prefix}_{head}_{k}"] = v
                return out

            cm_row = {}
            cm_row.update(_prefixed_cm("classic_val",  y_cl, p_cl, p_cl_img, p_cl_cd))
            cm_row.update(_prefixed_cm("ensemble_val", y_en, p_en, p_en_img, p_en_cd))
            cm_row.update(_prefixed_cm("train",        y_tr, p_tr_f, p_tr_i, p_tr_m))

            fold_logger.write_epoch_row({
                "fold": fold, "epoch": epoch + 1,
                "phase_single": phase_single, "phase_bilat": phase_bilat,
                "main_epoch_single": main_epoch_single, "main_epoch_bilat": main_epoch_bilat,
                "single_active": int(single_active), "bilat_active": int(bilat_active),
                "single_train_loss": _f(sl_loss), "single_train_acc": _f(sl_acc),
                "classic_val_auc":  _f(cl_auc), "classic_val_acc":  _f(cl_acc), "classic_val_n":  cl_n,
                "ensemble_val_auc": _f(en_auc), "ensemble_val_acc": _f(en_acc), "ensemble_val_n": en_n,
                "bilat_train_loss": _f(bl_loss), "bilat_train_acc": _f(bl_acc),
                "bilat_val_auc":    _f(bi_auc), "bilat_val_acc":    _f(bi_acc), "bilat_val_n":    bi_n,
                "classic_val_auc_img":  _f(cl_auc_img), "classic_val_acc_img":  _f(cl_acc_img),
                "classic_val_auc_cd":   _f(cl_auc_cd),  "classic_val_acc_cd":   _f(cl_acc_cd),
                "classic_val_fe_corr":  cl_fe_corr,      "classic_val_fe_err":   cl_fe_err,
                "ensemble_val_auc_img": _f(en_auc_img), "ensemble_val_acc_img": _f(en_acc_img),
                "ensemble_val_auc_cd":  _f(en_auc_cd),  "ensemble_val_acc_cd":  _f(en_acc_cd),
                "ensemble_val_fe_corr": en_fe_corr,      "ensemble_val_fe_err":  en_fe_err,
                "bilat_val_auc_img":    _f(bi_auc_img), "bilat_val_acc_img":    _f(bi_acc_img),
                "bilat_val_auc_cd":     _f(bi_auc_cd),  "bilat_val_acc_cd":     _f(bi_acc_cd),
                "bilat_val_fe_corr":    bi_fe_corr,      "bilat_val_fe_err":     bi_fe_err,
                "train_auc_fused": _f(tr_auc_f), "train_acc_fused": _f(tr_acc_f),
                "train_auc_img":   _f(tr_auc_i), "train_acc_img":   _f(tr_acc_i),
                "train_auc_cd":    _f(tr_auc_m), "train_acc_cd":    _f(tr_acc_m),
                "train_fe_corr":   tr_fe_corr,    "train_fe_err":    tr_fe_err,
                "train_n":         tr_n,
                "is_best_single":  int(is_best_single),
                "is_best_bilat":   int(is_best_bilat),
                "test_auc": _f(te_auc), "test_acc": _f(te_acc),
                **cm_row,
            }, optional_cols=epoch_fields)

            # md_warmup progress bar
            if phase_single == "cd_warmup":
                _bar_w = 30
                _filled = int(_bar_w * (epoch + 1) / single_warmup_cd)
                _bar = "#" * _filled + "-" * (_bar_w - _filled)
                msg = f"  [fold {fold+1}] md_warmup [{_bar}] {epoch+1}/{single_warmup_cd}  loss={sl_loss:.2f}"
                print(f"\r{msg}", end="", flush=True)
                fold_logger.info(msg)
                _prev_phase_single = phase_single
                continue

            if _prev_phase_single == "cd_warmup":
                print()

            if args.log_every > 0 and (epoch + 1) % args.log_every == 0:
                _epoch_secs = time.time() - _epoch_t0
                if tower_mode == "ensemble":
                    _auc_v, _acc_v = en_auc, en_acc
                elif tower_mode in ("bilateral", "siamese"):
                    _auc_v, _acc_v = bi_auc, bi_acc
                else:
                    _auc_v, _acc_v = cl_auc, cl_acc
                print(f"  ep {epoch+1:>3}/{total_epochs} ({_epoch_secs:.1f}s)  auc={_auc_v:.2f}  acc={_acc_v:.2f}", flush=True)
                fold_logger.info(f"  ep {epoch+1:>3}/{total_epochs} ({_epoch_secs:.1f}s)  auc={_auc_v:.4f}  acc={_acc_v:.4f}")

            _prev_phase_single = phase_single

        fold_logger.close()

        # No checkpoint saving in V3.

        # Save per-epoch npy tensors
        if _epoch_train_pf:
            ids_ref = _epoch_train_ids[0]; y_ref = _epoch_train_y[0]
            np.save(fold_dir / "train_patient_ids.npy", ids_ref)
            np.save(fold_dir / "train_y_true.npy",      y_ref)
            np.save(fold_dir / "train_probs_fused.npy", np.stack(_epoch_train_pf))
            np.save(fold_dir / "train_probs_img.npy",   np.stack(_epoch_train_pi))
            np.save(fold_dir / "train_probs_cd.npy",    np.stack(_epoch_train_pm))
        if _epoch_val_pf_od:
            np.save(fold_dir / "val_y_true_epochs.npy",         np.stack(_epoch_val_y))
            np.save(fold_dir / "val_probs_fused_od_epochs.npy", np.stack(_epoch_val_pf_od))
            np.save(fold_dir / "val_probs_img_od_epochs.npy",   np.stack(_epoch_val_pi_od))
            np.save(fold_dir / "val_probs_cd_od_epochs.npy",    np.stack(_epoch_val_pm_od))
            np.save(fold_dir / "val_probs_fused_os_epochs.npy", np.stack(_epoch_val_pf_os))
            np.save(fold_dir / "val_probs_img_os_epochs.npy",   np.stack(_epoch_val_pi_os))
            np.save(fold_dir / "val_probs_cd_os_epochs.npy",    np.stack(_epoch_val_pm_os))
            if _epoch_val_ids:
                np.save(fold_dir / "val_patient_ids.npy", _epoch_val_ids[0])

        # ================================================================
        # Phase 2: fused head (ensemble only)
        # ================================================================
        snap_holdout_fused: dict = {}

        if run_fused and single is not None:
            for p in single.parameters():
                p.requires_grad_(False)
            head_type = getattr(args, "head_type", "attention")
            if head_type == "logit_mlp":
                fused = LogitMLPEnsembleHT(single, num_classes).to(device)
            elif head_type == "embedding_mlp":
                fused = EmbeddingMLPEnsembleHT(single, num_classes).to(device)
            else:
                fused = FusedEnsembleHT(single, num_classes).to(device)
            opt_fused = torch.optim.Adam(fused.head.parameters(), lr=args.lr)
            fusion_epochs = int(getattr(args, "fusion_epochs", 10))
            print(
                f"  [fold {fold+1}] Phase 2: fusion head  bilat_train_n={len(bilat_train)}  epochs={fusion_epochs}",
                flush=True,
            )
            _val_pids_for_store = [str(s["id_1"]) for s in bilat_val]
            for fep in range(fusion_epochs):
                fu_loss, fu_acc = train_fusion_epoch(fused, train_bilat_loader, opt_fused, device)
                y_fu, p_fu = collect_probs_fused(fused, val_loader, device)
                fu_auc = _score_arrays(y_fu, p_fu, num_classes)[1]
                if pred_store is not None and y_fu.size:
                    pred_store.record(fold, total_single_epochs + fep, _val_pids_for_store, "bilat_fused", p_fu)
                if y_fu.size and not np.isnan(fu_auc):
                    fu_acc_val = _score_arrays(y_fu, p_fu, num_classes)[0]
                    snap_fused, _, _, _ = _tune_and_snap(y_fu, p_fu, fu_acc_val, num_classes, args, args.ece_bins)
                if (fep + 1) % max(1, args.log_every) == 0:
                    print(
                        f"  [fold {fold+1}] fusion ep{fep+1:>3}  loss={fu_loss:.2f}  "
                        f"val_auc={fu_auc:.2f}",
                        flush=True,
                    )
            # No checkpoint saving for fused head either.

        # ================================================================
        # Post-training: evaluate ONCE on val (final-epoch state) then test set
        # ================================================================

        # Val artifacts
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
        elif run_siamese:
            y_bi_best, p_bi_best = collect_probs_siamese(siamese, val_loader, device)
        else:
            y_bi_best = p_bi_best = None

        # Optional checkpoint saving (final-epoch weights for explainability)
        if getattr(args, "save_checkpoints", False) and run_single and single is not None:
            import torch as _torch
            _torch.save(single.state_dict(), fold_dir / "best_single.pt")

        # Compute val snaps from final-epoch model state
        if run_single and tower_mode == "single" and y_cl_best is not None:
            snap_cl, _, _, _ = _tune_and_snap(y_cl_best, p_cl_best, float((p_cl_best.argmax(1) == y_cl_best).mean()), num_classes, args, args.ece_bins)
            snap_classic = snap_cl
        elif run_single and tower_mode == "ensemble" and y_en_best is not None:
            snap_en, _, _, _ = _tune_and_snap(y_en_best, p_en_best, float((p_en_best.argmax(1) == y_en_best).mean()), num_classes, args, args.ece_bins)
            snap_ensemble = snap_en
        if (run_bilat or run_siamese) and y_bi_best is not None:
            snap_bi, _, _, _ = _tune_and_snap(y_bi_best, p_bi_best, float((p_bi_best.argmax(1) == y_bi_best).mean()), num_classes, args, args.ece_bins)
            snap_bilat = snap_bi

        y_fu_best = p_fu_best = None
        if run_fused and single is not None:
            y_fu_best, p_fu_best = collect_probs_fused(fused, val_loader, device)
            if y_fu_best is not None and y_fu_best.size:
                fu_acc_best = float((p_fu_best.argmax(1) == y_fu_best).mean())
                snap_fused, _, _, _ = _tune_and_snap(
                    y_fu_best, p_fu_best, fu_acc_best, num_classes, args, args.ece_bins
                )

        # Test set evaluation (once, never seen during training)
        snap_test: dict = {}
        snap_fused_test: dict = {}
        y_test_out = p_test_out = p_test_img_out = p_test_md_out = None
        y_fused_test_out = p_fused_test_out = None

        if test_loader is not None:
            if run_single and tower_mode == "ensemble":
                y_test_out, p_test_out, p_test_img_out, p_test_md_out = collect_probs_single_components(
                    single, test_loader, device, aggregate_patient=True
                )
            elif run_single and tower_mode == "single":
                y_test_out, p_test_out, p_test_img_out, p_test_md_out = collect_probs_single_components(
                    single, test_loader, device, aggregate_patient=False
                )
            elif run_bilat:
                y_test_out, p_test_out, _, _ = collect_probs_bilateral_components(
                    bilateral, test_loader, device
                )
            elif run_siamese:
                y_test_out, p_test_out = collect_probs_siamese(siamese, test_loader, device)
            if y_test_out is not None and y_test_out.size:
                test_acc_raw = float((p_test_out.argmax(1) == y_test_out).mean())
                snap_test, _, _, _ = _tune_and_snap(
                    y_test_out, p_test_out, test_acc_raw, num_classes, args, args.ece_bins
                )
                print(
                    f"  [fold {fold+1}] TEST  "
                    f"auc={snap_test.get('auc', nan):.2f}  "
                    f"acc={snap_test.get('acc', nan):.2f}  "
                    f"kappa={snap_test.get('kappa', nan):.2f}  "
                    f"f1={snap_test.get('macro_f1', nan):.2f}  "
                    f"ece={snap_test.get('ece', nan):.2f}  "
                    f"n={snap_test.get('n', 0)}",
                    flush=True,
                )
                _save_predictions_csv(
                    fold_dir, mode, y_test_out,
                    {"fused": p_test_out, "img": p_test_img_out, "md": p_test_md_out},
                    suffix="_test",
                )
            if run_fused and single is not None:
                y_fused_test_out, p_fused_test_out = collect_probs_fused(
                    fused, test_loader, device
                )
                if y_fused_test_out is not None and y_fused_test_out.size:
                    fused_test_acc_raw = float(
                        (p_fused_test_out.argmax(1) == y_fused_test_out).mean()
                    )
                    snap_fused_test, _, _, _ = _tune_and_snap(
                        y_fused_test_out, p_fused_test_out, fused_test_acc_raw,
                        num_classes, args, args.ece_bins
                    )
                    print(
                        f"  [fold {fold+1}] FUSED_HEAD TEST  "
                        f"auc={snap_fused_test.get('auc', nan):.2f}  "
                        f"acc={snap_fused_test.get('acc', nan):.2f}  "
                        f"kappa={snap_fused_test.get('kappa', nan):.2f}  "
                        f"f1={snap_fused_test.get('macro_f1', nan):.2f}  "
                        f"ece={snap_fused_test.get('ece', nan):.2f}  "
                        f"n={snap_fused_test.get('n', 0)}",
                        flush=True,
                    )
                    _save_predictions_csv(
                        fold_dir, mode, y_fused_test_out,
                        {"fused_head": p_fused_test_out},
                        suffix="_fused_head_test",
                    )
        else:
            print(f"  [fold {fold+1}] WARNING: no test samples for this fold.", flush=True)

        test_n = snap_test.get("n", 0)

        return FoldResult(
            mode=mode, fold=fold,
            best_epoch_single=0, best_epoch_bilat=0,
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
            ensemble_test_auc=snap_test.get("auc", nan)      if tower_mode == "ensemble"  else nan,
            ensemble_test_acc=snap_test.get("acc", nan)      if tower_mode == "ensemble"  else nan,
            ensemble_test_kappa=snap_test.get("kappa", nan)  if tower_mode == "ensemble"  else nan,
            ensemble_test_f1=snap_test.get("macro_f1", nan)  if tower_mode == "ensemble"  else nan,
            ensemble_test_ece=snap_test.get("ece", nan)      if tower_mode == "ensemble"  else nan,
            classic_test_auc=snap_test.get("auc", nan)       if tower_mode == "single"    else nan,
            classic_test_acc=snap_test.get("acc", nan)       if tower_mode == "single"    else nan,
            classic_test_kappa=snap_test.get("kappa", nan)   if tower_mode == "single"    else nan,
            classic_test_f1=snap_test.get("macro_f1", nan)   if tower_mode == "single"    else nan,
            classic_test_ece=snap_test.get("ece", nan)       if tower_mode == "single"    else nan,
            bilat_test_auc=snap_test.get("auc", nan)         if tower_mode in ("bilateral", "siamese") else nan,
            bilat_test_acc=snap_test.get("acc", nan)         if tower_mode in ("bilateral", "siamese") else nan,
            bilat_test_kappa=snap_test.get("kappa", nan)     if tower_mode in ("bilateral", "siamese") else nan,
            bilat_test_f1=snap_test.get("macro_f1", nan)     if tower_mode in ("bilateral", "siamese") else nan,
            bilat_test_ece=snap_test.get("ece", nan)         if tower_mode in ("bilateral", "siamese") else nan,
            test_n=test_n,
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
            fused_test_auc=snap_fused_test.get("auc", nan),
            fused_test_acc=snap_fused_test.get("acc", nan),
            fused_test_kappa=snap_fused_test.get("kappa", nan),
            fused_test_f1=snap_fused_test.get("macro_f1", nan),
            fused_test_ece=snap_fused_test.get("ece", nan),
            fused_test_n=snap_fused_test.get("n", 0),
        ), FoldArtifacts(
            y_true_classic=y_cl_best,   probs_classic=p_cl_best,
            y_true_ensemble=y_en_best,  probs_ensemble=p_en_best,
            y_true_bilat=y_bi_best,     probs_bilat=p_bi_best,
            y_true_fused=y_fu_best,     probs_fused=p_fu_best,
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
            y_true_test=y_test_out,
            probs_test=p_test_out,
            probs_test_img=p_test_img_out,
            probs_test_md=p_test_md_out,
            y_true_fused_test=y_fused_test_out,
            probs_fused_test=p_fused_test_out,
        )

    @staticmethod
    def _summary(results: list[FoldResult]) -> dict:
        def _ms(vals):
            v = np.array([x for x in vals if x is not None and not np.isnan(float(x))], dtype=float)
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
            ("ensemble_test", "ensemble_test"),
            ("classic_test",  "classic_test"),
            ("bilat_test",    "bilat_test"),
            ("fused_test",    "fused_test"),
        ]:
            sub = {}
            for m in ["auc", "acc", "kappa", "f1", "ece"]:
                vals = [getattr(r, f"{prefix}_{m}") for r in results]
                mean, std = _ms(vals)
                sub[f"{m}_mean"] = mean
                sub[f"{m}_std"] = std
            out[label] = sub

        for delta_label, prefix_a, prefix_b in [
            ("delta_ensemble_vs_classic", "classic_val",  "ensemble_val"),
            ("delta_bilat_vs_ensemble",   "ensemble_val", "bilat_val"),
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
        return out

    @staticmethod
    def _print_summary(mode: str, s: dict, tower_mode: str | None = None) -> None:
        def f(v):
            return " nan " if v is None else f"{v:.2f}"
        def fsd(mean, std):
            if mean is None: return "   nan   "
            if std is None: return f"{mean:.2f}     "
            return f"{mean:.2f}±{std:.2f}"

        # Resolve test key
        if tower_mode in ("single", "classic"):
            test_key = "classic_test"
        elif tower_mode == "ensemble":
            test_key = "ensemble_test"
        elif tower_mode in ("bilateral", "siamese"):
            test_key = "bilat_test"
        else:
            test_key = "classic_test"

        td = s.get(test_key, {})
        print(f"\n=== Summary [{mode}] ===")
        print(f"  {'':26s}  {'AUC':>16}  {'ACC':>16}  {'Kappa':>16}  {'F1-mac':>16}  {'ECE':>16}")
        print(f"  {'Test':26s}  "
              f"{fsd(td.get('auc_mean'), td.get('auc_std')):>16}  "
              f"{fsd(td.get('acc_mean'), td.get('acc_std')):>16}  "
              f"{fsd(td.get('kappa_mean'), td.get('kappa_std')):>16}  "
              f"{fsd(td.get('f1_mean'), td.get('f1_std')):>16}  "
              f"{fsd(td.get('ece_mean'), td.get('ece_std')):>16}")
        print()
