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

from classes.v2.croppers import build_image_preprocessor_from_args
from classes.v2.dataset import _ClinicalView  # noqa: F401  (re-exported for compat)
from classes.v2.loader_factory import (
    filter_bilateral_samples,
    filter_eye_samples,
    make_loader,
)
from classes.v2.metrics import _score_arrays, _svf, _tune_and_snap
from classes.v2.models import (
    BilateralHT,
    SingleEyeHT,
    V2ModeComparisonOps,
    collect_probs_bilateral,
    collect_probs_bilateral_components,
    collect_probs_classic,
    collect_probs_ensemble,
    collect_probs_single_components,
    train_bilateral_epoch,
    train_single_epoch,
)
from classes.v2.papila_builders import build_papila_data
from classes.v2.profiles import build_papila_profile
from classes.v2.results import FoldArtifacts, FoldResult, _f, _nan, _sv
from classes.v2.split_manager import PatientFirstSplitManager
from classes.v2.transforms import build_eval_transform
from classes.v2.utils import (
    _drop_mixed_label_patients,
    _relabel_mixed_patients_to_max,
    choose_device,
    seed_everything,
)
from classes.v2.hypertower_logger import HypertowerLogger


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
        ap.add_argument("--bilat-warmup-tower-epochs", type=int, default=None,
                        help="Bilateral model tower warmup (overrides --warmup-tower-epochs).")
        ap.add_argument("--bilat-warmup-fused-epochs", type=int, default=None,
                        help="Bilateral model fused warmup (overrides --warmup-fused-epochs).")
        ap.add_argument("--batch-size",   type=int,   default=8)
        ap.add_argument("--lr",           type=float, default=1e-4)
        ap.add_argument("--bcd-prob",     type=float, default=0.5,
                        help="Tower-only step probability during main phase (per model).")
        ap.add_argument("--backbone",     default="refugelike")
        ap.add_argument("--freeze-ratio", type=float, default=0.0)
        ap.add_argument("--augment",      action="store_true")
        ap.add_argument("--num-workers",  type=int,   default=0)
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
        ap.add_argument("--md-hidden-dim", type=int, default=128,
                        help="MDTower hidden dimension.")
        ap.add_argument("--fusion-dim",    type=int, default=256,
                        help="Bridge/BilateralBridge fusion dimension.")
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
            )
            fold_results.append(result)
            if artifacts.y_true_ensemble is not None:
                np.save(fold_dir / "y_true.npy", artifacts.y_true_ensemble)
            if artifacts.probs_ensemble is not None:
                np.save(fold_dir / "probs_fused.npy", artifacts.probs_ensemble)
            if artifacts.probs_classic is not None:
                np.save(fold_dir / "probs_classic.npy", artifacts.probs_classic)
            if artifacts.probs_bilat is not None:
                np.save(fold_dir / "probs_bilat.npy", artifacts.probs_bilat)

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
    ) -> tuple[FoldResult, FoldArtifacts]:
        args = self.args
        device = self.device
        image_preprocessor = self.image_preprocessor
        nan = _nan()
        tower_mode = "single" if tower_mode == "classic" else tower_mode
        run_single = tower_mode in ("single", "ensemble")
        run_bilat = tower_mode == "bilateral"

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
        if not run_single:
            single_warmup_tower = 0
            single_warmup_fused = 0
        if not run_bilat:
            bilat_warmup_tower = 0
            bilat_warmup_fused = 0
        main_epochs = int(args.epochs)
        total_single_epochs = (single_warmup_tower + single_warmup_fused + main_epochs) if run_single else 0
        total_bilat_epochs = (bilat_warmup_tower + bilat_warmup_fused + main_epochs) if run_bilat else 0
        total_epochs = max(total_single_epochs, total_bilat_epochs)

        # ---- samples ---------------------------------------------------
        eye_train   = filter_eye_samples(profile_eye.build_samples(df=split.train, clinical=data))
        bilat_train = filter_bilateral_samples(profile_patient.build_samples(df=split.train, clinical=data))
        bilat_val   = filter_bilateral_samples(profile_patient.build_samples(df=split.val, clinical=data))

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
                md_hidden_dim=args.md_hidden_dim, fusion_dim=args.fusion_dim,
            ).to(device)
        if run_bilat:
            bilateral = BilateralHT(
                backbone=args.backbone, freeze_ratio=args.freeze_ratio,
                augment=args.augment, clinical_data=data,
                num_classes=num_classes,
                md_hidden_dim=args.md_hidden_dim, fusion_dim=args.fusion_dim,
            ).to(device)

        slots_eye     = profile_eye.slot_descriptors()
        slots_patient = profile_patient.slot_descriptors()
        loader_kw     = dict(batch_size=args.batch_size, num_workers=args.num_workers)

        # ---- loaders ---------------------------------------------------
        train_single_loader = None
        train_bilat_loader  = None
        if run_single:
            train_single_loader = make_loader(
                eye_train, slots_eye,
                image_transform=single.transform,
                image_preprocessor=image_preprocessor,
                shuffle=True,
                **loader_kw,
            )
        if run_bilat:
            train_bilat_loader = make_loader(
                bilat_train, slots_patient,
                image_transform=bilateral.transform,
                image_preprocessor=image_preprocessor,
                shuffle=True,
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

        opt_single   = torch.optim.Adam(single.parameters(), lr=args.lr)   if run_single else None
        opt_bilateral = torch.optim.Adam(bilateral.parameters(), lr=args.lr) if run_bilat  else None

        # ---- epoch log -------------------------------------------------
        epoch_fields = [
            "fold", "epoch",
            "phase_single", "phase_bilat",
            "main_epoch_single", "main_epoch_bilat",
            "single_active", "bilat_active",
            "single_train_loss", "single_train_acc",
            "classic_val_auc",      "classic_val_acc",      "classic_val_n",
            "ensemble_val_auc",     "ensemble_val_acc",     "ensemble_val_n",
            "bilat_train_loss",     "bilat_train_acc",
            "bilat_val_auc",        "bilat_val_acc",        "bilat_val_n",
            "classic_holdout_auc",  "classic_holdout_acc",
            "ensemble_holdout_auc", "ensemble_holdout_acc",
            "bilat_holdout_auc",    "bilat_holdout_acc",
            "is_best_single", "is_best_bilat",
            "is_best_holdout_single", "is_best_holdout_bilat",
        ]
        fold_logger = HypertowerLogger(run_dir=fold_dir)

        # ---- best-epoch trackers ---------------------------------------
        best_single_auc = -1.0
        best_bilat_auc  = -1.0
        best_epoch_single = 0
        best_epoch_bilat  = 0
        best_single_state: Optional[dict] = None
        best_bilat_state:  Optional[dict] = None
        snap_classic:  dict = {}
        snap_ensemble: dict = {}
        snap_bilat:    dict = {}
        snap_holdout_single: dict = {}
        snap_holdout_bilat:  dict = {}
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
                f"single_warmup={single_warmup_tower}+{single_warmup_fused} total={total_single_epochs}",
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
        for epoch in range(total_epochs):
            if not run_single:
                phase_single, main_epoch_single, single_active = "inactive", 0, False
            elif epoch < single_warmup_tower:
                phase_single, main_epoch_single, single_active = "tower_warmup", 0, True
            elif epoch < (single_warmup_tower + single_warmup_fused):
                phase_single, main_epoch_single, single_active = "fused_warmup", 0, True
            elif epoch < total_single_epochs:
                phase_single, main_epoch_single, single_active = (
                    "main",
                    epoch - single_warmup_tower - single_warmup_fused + 1,
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
                sl_loss, sl_acc = train_single_epoch(
                    single, train_single_loader, opt_single, device,
                    phase=phase_single, bcd_prob=float(args.bcd_prob),
                )
            else:
                sl_loss, sl_acc = nan, nan

            if run_bilat and bilat_active:
                bl_loss, bl_acc = train_bilateral_epoch(
                    bilateral, train_bilat_loader, opt_bilateral, device,
                    phase=phase_bilat, bcd_prob=float(args.bcd_prob),
                )
            else:
                bl_loss, bl_acc = nan, nan

            if run_single and tower_mode == "single":
                y_cl, p_cl, p_cl_img, p_cl_md = collect_probs_single_components(
                    single, val_loader, device, aggregate_patient=False
                )
                cl_acc, cl_auc, cl_n = _score_arrays(y_cl, p_cl, num_classes)
                cl_acc_img = float((p_cl_img.argmax(1) == y_cl).mean()) if y_cl.size else nan
                cl_acc_md  = float((p_cl_md.argmax(1) == y_cl).mean()) if y_cl.size else nan
                _, cl_auc_img, _ = _score_arrays(y_cl, p_cl_img, num_classes)
                _, cl_auc_md, _  = _score_arrays(y_cl, p_cl_md,  num_classes)
                y_en = np.array([], dtype=np.int64)
                p_en = np.zeros((0, 0), dtype=np.float32)
                en_acc = en_auc = nan
                en_n = 0
                en_acc_img = en_acc_md = en_auc_img = en_auc_md = nan
            elif run_single and tower_mode == "ensemble":
                y_en, p_en, p_en_img, p_en_md = collect_probs_single_components(
                    single, val_loader, device, aggregate_patient=True
                )
                en_acc, en_auc, en_n = _score_arrays(y_en, p_en, num_classes)
                en_acc_img = float((p_en_img.argmax(1) == y_en).mean()) if y_en.size else nan
                en_acc_md  = float((p_en_md.argmax(1) == y_en).mean()) if y_en.size else nan
                _, en_auc_img, _ = _score_arrays(y_en, p_en_img, num_classes)
                _, en_auc_md, _  = _score_arrays(y_en, p_en_md,  num_classes)
                y_cl = np.array([], dtype=np.int64)
                p_cl = np.zeros((0, 0), dtype=np.float32)
                cl_acc = cl_auc = nan
                cl_n = 0
                cl_acc_img = cl_acc_md = cl_auc_img = cl_auc_md = nan
            else:
                y_cl = y_en = np.array([], dtype=np.int64)
                p_cl = p_en = np.zeros((0, 0), dtype=np.float32)
                cl_acc = cl_auc = en_acc = en_auc = nan
                cl_n = en_n = 0
                cl_acc_img = cl_acc_md = en_acc_img = en_acc_md = nan
                cl_auc_img = cl_auc_md = en_auc_img = en_auc_md = nan

            if run_bilat:
                y_bi, p_bi, p_bi_img, p_bi_md = collect_probs_bilateral_components(
                    bilateral, val_loader, device
                )
                bi_acc, bi_auc, bi_n = _score_arrays(y_bi, p_bi, num_classes)
                bi_acc_img = float((p_bi_img.argmax(1) == y_bi).mean()) if y_bi.size else nan
                bi_acc_md  = float((p_bi_md.argmax(1) == y_bi).mean()) if y_bi.size else nan
                _, bi_auc_img, _ = _score_arrays(y_bi, p_bi_img, num_classes)
                _, bi_auc_md, _  = _score_arrays(y_bi, p_bi_md,  num_classes)
            else:
                y_bi = np.array([], dtype=np.int64)
                p_bi = np.zeros((0, 0), dtype=np.float32)
                bi_acc = bi_auc = nan
                bi_n = 0
                bi_acc_img = bi_acc_md = bi_auc_img = bi_auc_md = nan

            # --- holdout evaluation ------------------------------------
            if holdout_loader is not None:
                if run_single and tower_mode == "single":
                    y_cl_h, p_cl_h, _, _ = collect_probs_single_components(
                        single, holdout_loader, device, aggregate_patient=False
                    )
                    _, cl_auc_h, _ = _score_arrays(y_cl_h, p_cl_h, num_classes)
                    cl_acc_h = float((p_cl_h.argmax(1) == y_cl_h).mean()) if y_cl_h.size else nan
                    en_auc_h = en_acc_h = nan
                elif run_single and tower_mode == "ensemble":
                    y_en_h, p_en_h, _, _ = collect_probs_single_components(
                        single, holdout_loader, device, aggregate_patient=True
                    )
                    _, en_auc_h, _ = _score_arrays(y_en_h, p_en_h, num_classes)
                    en_acc_h = float((p_en_h.argmax(1) == y_en_h).mean()) if y_en_h.size else nan
                    cl_auc_h = cl_acc_h = nan
                else:
                    cl_auc_h = cl_acc_h = en_auc_h = en_acc_h = nan
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
                "classic_holdout_auc":  _f(cl_auc_h), "classic_holdout_acc":  _f(cl_acc_h),
                "ensemble_holdout_auc": _f(en_auc_h), "ensemble_holdout_acc": _f(en_acc_h),
                "bilat_holdout_auc":    _f(bi_auc_h), "bilat_holdout_acc":    _f(bi_acc_h),
                "is_best_single":         int(is_best_single),
                "is_best_bilat":          int(is_best_bilat),
                "is_best_holdout_single": int(is_best_holdout_single),
                "is_best_holdout_bilat":  int(is_best_holdout_bilat),
            }, optional_cols=epoch_fields)

            if args.log_every > 0 and (epoch + 1) % args.log_every == 0:
                hld_auc = target_holdout_single_auc if run_single else bi_auc_h
                hld_suffix = f"  hld_auc={hld_auc:.4f}" if holdout_loader is not None else ""
                if run_single:
                    if tower_mode == "single":
                        msg = (
                            f"    ep {epoch+1:>3}/{total_epochs} "
                            f"[single:{phase_single} {main_epoch_single}/{main_epochs}] "
                            f"fused(acc={cl_acc:.4f},auc={cl_auc:.4f}) "
                            f"img(acc={cl_acc_img:.4f},auc={cl_auc_img:.4f}) "
                            f"md(acc={cl_acc_md:.4f},auc={cl_auc_md:.4f})  "
                            f"(best_fused={best_single_auc:.4f} @ep{best_epoch_single})"
                            f"{hld_suffix}"
                        )
                    else:
                        msg = (
                            f"    ep {epoch+1:>3}/{total_epochs} "
                            f"[single:{phase_single} {main_epoch_single}/{main_epochs}] "
                            f"fused(acc={en_acc:.4f},auc={en_auc:.4f}) "
                            f"img(acc={en_acc_img:.4f},auc={en_auc_img:.4f}) "
                            f"md(acc={en_acc_md:.4f},auc={en_auc_md:.4f})  "
                            f"(best_fused={best_single_auc:.4f} @ep{best_epoch_single})"
                            f"{hld_suffix}"
                        )
                else:
                    msg = (
                        f"    ep {epoch+1:>3}/{total_epochs} "
                        f"[bilat:{phase_bilat} {main_epoch_bilat}/{main_epochs}] "
                        f"fused(acc={bi_acc:.4f},auc={bi_auc:.4f}) "
                        f"img(acc={bi_acc_img:.4f},auc={bi_auc_img:.4f}) "
                        f"md(acc={bi_acc_md:.4f},auc={bi_auc_md:.4f})  "
                        f"(best_bilat={best_bilat_auc:.4f} @ep{best_epoch_bilat})"
                        f"{hld_suffix}"
                    )
                print(msg, flush=True)
                fold_logger.info(msg)

        fold_logger.close()

        if args.save_checkpoints:
            if best_single_state is not None:
                torch.save(best_single_state, fold_dir / "best_single.pt")
            if best_bilat_state is not None:
                torch.save(best_bilat_state, fold_dir / "best_bilateral.pt")
            if best_holdout_single_state is not None:
                torch.save(best_holdout_single_state, fold_dir / "best_holdout_single.pt")
            if best_holdout_bilat_state is not None:
                torch.save(best_holdout_bilat_state, fold_dir / "best_holdout_bilateral.pt")

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
        if run_single and tower_mode == "single":
            y_cl_best, p_cl_best = collect_probs_classic(single, val_loader, device)
            y_en_best = p_en_best = None
        elif run_single and tower_mode == "ensemble":
            y_en_best, p_en_best = collect_probs_ensemble(single, val_loader, device)
            y_cl_best = p_cl_best = None
        else:
            y_cl_best = y_en_best = None
            p_cl_best = p_en_best = None
        if run_bilat:
            y_bi_best, p_bi_best = collect_probs_bilateral(bilateral, val_loader, device)
        else:
            y_bi_best = p_bi_best = None

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
        ), FoldArtifacts(
            y_true_classic=y_cl_best,  probs_classic=p_cl_best,
            y_true_ensemble=y_en_best, probs_ensemble=p_en_best,
            y_true_bilat=y_bi_best,    probs_bilat=p_bi_best,
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
            "bilateral=patient-level BilateralHT"
        )
        return out

    @staticmethod
    def _print_summary(mode: str, s: dict, tower_mode: str | None = None) -> None:
        def f(v):
            return "nan" if v is None else f"{v:.4f}"

        cv = s["classic_best_val"]
        ev = s["ensemble_best_val"]
        bv = s["bilat_best_val"]
        d1 = s["delta_ensemble_vs_classic"]
        d2 = s["delta_bilat_vs_ensemble"]

        print(f"\n=== Summary [{mode}] — best-epoch val ===")
        print(f"  {'':26s}  {'AUC':>8}  {'ACC':>8}  {'Kappa':>8}  {'F1-mac':>8}  {'ECE':>8}")
        if tower_mode == "single":
            rows = [("single   (eye-lvl  eval)", cv)]
        elif tower_mode == "ensemble":
            rows = [("ensemble (pat-lvl  eval)", ev)]
        elif tower_mode == "bilateral":
            rows = [("bilateral (bilat   eval)", bv)]
        else:
            rows = [
                ("classic  (eye-lvl  eval)", cv),
                ("ensemble (pat-lvl  eval)", ev),
                ("bilateral (bilat   eval)", bv),
            ]
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
