#!/usr/bin/env python3
"""Tkinter front-end for scripts/run_multifold.py."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import sys
import signal
import subprocess
import threading
import time
import shutil
import re
from pathlib import Path
from typing import Dict, Optional
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import filedialog, messagebox

from classes.hypertower import HyperTower
from classes.backbones import list_names as list_backbones
from classes import build_papila_clinical


BACKBONES = [
    "efficientnet_b0",
    "resnet50",
    "densenet121",
    "refugelike",
    "refuge_densenet",
    "refuge_efficient_b0",
    "refuge_efficient_b7",
]

FUSION_MODES = ["fused", "image_only", "metadata_only", "vote"]
EVAL_MODES = ["multiclass", "binary"]


class Multifold:
    """Core multifold runner extracted from scripts/run_multifold."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        ap = argparse.ArgumentParser(description="Run k-fold CV and emit per-fold logs, npy, and ROC plots.")
        ap.add_argument("--image-dir", default="Papila/FundusImages")
        ap.add_argument("--clinical-dir", default="Papila/ClinicalData")
        ap.add_argument("--label-col", default="Diagnosis")
        ap.add_argument("--cat-cols", nargs="*", default=["Gender", "Phakic/Pseudophakic"])

        ap.add_argument("--backbone", type=str, default="efficientnet_b0", choices=list_backbones())
        ap.add_argument("--freeze-ratio", type=float, default=0.0)
        ap.add_argument("--fusion-mode", default="fused", choices=["fused","image_only","metadata_only","vote"])
        ap.add_argument("--epochs", type=int, default=5)
        ap.add_argument("--batch-size", type=int, default=8)
        ap.add_argument("--lr", type=float, default=1e-4)
        ap.add_argument("--num-classes", type=int, default=3)
        ap.add_argument("--n-splits", type=int, default=5)
        ap.add_argument("--fold-seed", type=int, default=42,
                        help="Random seed for patient-level splits")
        ap.add_argument("--holdout-per-class", type=int, default=0,
                        help="Reserve this many samples per class for a monitoring holdout (0 disables)")
        ap.add_argument("--holdout-seed", type=int, default=123,
                        help="Random seed used when sampling the holdout subset")
        ap.add_argument("--img-crop-manifest", type=Path, default=None,
                        help="Optional manifest for UNet cropper (enables disc-centric crops)")
        ap.add_argument("--img-crop-weights", type=Path, default=None,
                        help="UNet checkpoint weights for cropping")
        ap.add_argument("--img-crop-normalize", choices=["none", "imagenet", "per_image"], default="per_image")
        ap.add_argument("--img-crop-threshold", type=float, default=0.5)
        ap.add_argument("--img-crop-scale", type=float, default=2.5)
        ap.add_argument("--img-crop-size", type=int, default=224)
        ap.add_argument("--img-crop-cache", type=Path, default=Path("analysis_data/hypertower_crops"))
        ap.add_argument("--img-crop-tta", action="store_true")
        ap.add_argument("--img-crop-gt", action="store_true",
                        help="Use ground-truth masks/contours from manifest for cropping instead of UNet")
        ap.add_argument("--no-img-augment", dest="img_augment", action="store_false",
                        help="Disable random image augmentations for the image tower")
        ap.set_defaults(img_augment=True)
        ap.add_argument("--img-geometry-features", action="store_true",
                        help="Append disc/cup geometry features to the image tower (requires cropping)")

        ap.add_argument("--shortname", default="multi")
        ap.add_argument("--plot-head", default="fused", choices=["image","fused","metadata"],
                        help="Which head to plot/aggregate.")
        ap.add_argument("--class-names", nargs="*", default=None)
        ap.add_argument("--no-se", dest="use_se", action="store_false",
                        help="Disable SE attention in the bridge (default: enabled)")
        ap.set_defaults(use_se=True)
        ap.add_argument("--se-reduction", type=int, default=16, choices=[8,16,32],
                        help="SE bottleneck: fusion_dim // reduction (default 16)")
        ap.add_argument("--se-pre-norm", dest="se_pre_norm", action="store_true",
                        help="Enable LayerNorm on branches before the multiply (default)")
        ap.add_argument("--no-se-pre-norm", dest="se_pre_norm", action="store_false",
                        help="Disable LayerNorm on branches before the multiply")
        ap.set_defaults(se_pre_norm=True)
        ap.add_argument("--se-where", choices=["bridge","tower","both","none"], default="bridge",
                        help="Where to apply SE: bridge (default), tower, both, or none")
        ap.add_argument("--se-reduction-tower", type=int, default=16, choices=[8,16,32],
                        help="SE bottleneck for tower vectors (default 16)")
        ap.add_argument("--se-pre-norm-tower", dest="se_pre_norm_tower", action="store_true",
                        help="Enable LayerNorm on tower vectors before SE (default)")
        ap.add_argument("--no-se-pre-norm-tower", dest="se_pre_norm_tower", action="store_false",
                        help="Disable LayerNorm on tower vectors before SE (default: enabled)")
        ap.set_defaults(se_pre_norm_tower=True)
        ap.add_argument("--eval_mode", choices=["multiclass","binary"], default="multiclass",
                        help="Multiclass (3 classes) or binary (Healthy vs Glaucoma; drops Suspect).")
        ap.add_argument("--warmup-tower-epochs", type=int, default=2)
        ap.add_argument("--warmup-fused-epochs", type=int, default=3)
        ap.add_argument("--gradual-thaw", action="store_true",
                        help="Enable gradual backbone thawing schedule (image/metadata towers)")
        ap.add_argument("--thaw-phase-duration", type=int, default=5,
                        help="Epochs per thaw phase (default 5)")
        ap.add_argument("--thaw-ratio", type=float, default=0.33,
                        help="Fraction of blocks to unfreeze each phase (default 0.33)")
        ap.add_argument("--thaw-target", choices=["image","metadata","both"], default="image",
                        help="Which tower(s) to apply gradual thaw to (default image)")
        ap.add_argument("--thaw-start-epoch", type=int, default=-1,
                        help="Epoch to start thawing (default: warmup_tower_epochs)")
        ap.add_argument("--initial-freeze", action="store_true",
                        help="Before thaw start, force backbone(s) fully frozen (default off)")

        ap.add_argument("--early-stop", action="store_true",
                        help="Enable early stopping")
        ap.add_argument("--early-metric", default=None,
                        help="Metric key to monitor (e.g., eval_loss, auc_fused, acc_fused).")
        ap.add_argument("--early-mode", choices=["auto","min","max"], default="auto")
        ap.add_argument("--early-monitor-holdout", action="store_true",
                        help="Monitor the holdout metric for early stopping/checkpointing (requires holdout set).")
        ap.add_argument("--early-patience", type=int, default=7)
        ap.add_argument("--early-min-delta", type=float, default=0.0)
        ap.add_argument("--checkpoint-best", action="store_true",
                        help="Save best weights to disk during training")
        ap.add_argument("--focal-gamma", type=float, default=0.0,
                        help="Focal loss exponent (0 disables focal loss)")
        ap.add_argument("--balanced-sampler", action="store_true",
                        help="Use a class-balanced bootstrapped sampler for the training loader")
        ap.add_argument("--run-id", default=None, help=argparse.SUPPRESS)
        return ap

    @staticmethod
    def _serialize_arg(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple)):
            return [Multifold._serialize_arg(v) for v in value]
        return value

    def _export_run_settings(self, run_dir: Path) -> None:
        data = {key: self._serialize_arg(value) for key, value in vars(self.args).items()}
        try:
            with open(run_dir / "cli_args.json", "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except Exception as exc:  # pragma: no cover
            print(f"[run_multifold] Failed to write cli_args.json: {exc}")

    def run(self, callback=None) -> None:
        if callback:
            sink = _StreamCallback(callback)
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                self._run_impl()
        else:
            self._run_impl()

    # ---- Helper functions copied from run_multifold ----------------
    @staticmethod
    def eval_collect_logits(ht: HyperTower):
        device = ht.device
        ht.img_tower.eval(); ht.md_tower.eval()
        if getattr(ht, "mode", "fused") != "vote":
            ht.bridge.eval()

        y_all = []
        pf, pi, pm = [], [], []
        for batch in ht.test_loader:
            if len(batch) == 4:
                imgs, metas, geometry, labels = batch
            else:
                imgs, metas, labels = batch
                geometry = None
            imgs = imgs.to(device)
            metas = metas.to(device)
            labels = labels.to(device)
            if geometry is not None and geometry.numel() > 0:
                geometry = geometry.to(device)
            else:
                geometry = None
            if ht.mode == "vote":
                img_feats = ht.img_tower(imgs, geometry)
                md_feats  = ht.md_tower(metas)
                out_img   = ht.head_img(img_feats)
                out_md    = ht.head_md(md_feats)
                out_fused = ht.vote(out_img, out_md)
            else:
                img_feats = ht.img_tower(imgs, geometry)
                md_feats  = ht.md_tower(metas)
                outputs   = ht.bridge(img_feats, md_feats)
                if isinstance(outputs, tuple):
                    out_fused, out_img, out_md = outputs
                else:
                    out_fused, out_img, out_md = outputs, None, None

            y_all.append(labels.detach().cpu().numpy())
            pf.append(F.softmax(out_fused, dim=1).detach().cpu().numpy())
            if out_img is not None:
                pi.append(F.softmax(out_img, dim=1).detach().cpu().numpy())
            if out_md is not None:
                pm.append(F.softmax(out_md, dim=1).detach().cpu().numpy())

        y_true = np.concatenate(y_all, axis=0)
        pf = np.concatenate(pf, axis=0) if pf else None
        pi = np.concatenate(pi, axis=0) if pi else None
        pm = np.concatenate(pm, axis=0) if pm else None
        return y_true, pf, pi, pm

    @staticmethod
    def auc_for(y, p):
        y = np.asarray(y)
        if p is None:
            return float("nan")
        if p.ndim == 1 or p.shape[1] == 1:
            return roc_auc_score(y, p.ravel())
        if p.shape[1] == 2:
            return roc_auc_score(y, p[:, 1])
        return roc_auc_score(y, p, multi_class="ovr", average="macro")

    @staticmethod
    def per_class_roc(y, p):
        if p is None:
            return {}
        K = p.shape[1]
        out = {}
        for k in range(K):
            y_bin = (y == k).astype(np.uint8)
            fpr, tpr, _ = roc_curve(y_bin, p[:, k])
            out[k] = (fpr, tpr, auc(fpr, tpr) if len(fpr) > 1 else np.nan)
        return out

    @staticmethod
    def plot_mean_sd(per_fold_curves, out_png, class_names=None, title="Mean OVR ROC (±1 SD)"):
        if not per_fold_curves:
            return
        fpr_grid = np.linspace(0, 1, 501)
        fig = plt.figure(figsize=(10, 8)); ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

        keys = sorted({k for d in per_fold_curves for k in d.keys()})
        if class_names is not None and len(class_names) == len(keys):
            name_map = {k: class_names[i] for i, k in enumerate(keys)}
        else:
            name_map = {k: f"class {k}" for k in keys}

        for k in keys:
            tprs, aucs = [], []
            for d in per_fold_curves:
                if k not in d:
                    continue
                fpr, tpr, a = d[k]
                tprs.append(np.interp(fpr_grid, fpr, tpr))
                aucs.append(a)
            if not tprs:
                continue
            tprs = np.vstack(tprs)
            mean = tprs.mean(axis=0); std = tprs.std(axis=0)
            auc_mean = np.nanmean(aucs); auc_std = np.nanstd(aucs)
            label = f"{name_map[k]} (AUC {auc_mean:.3f}±{auc_std:.3f})"
            ax.plot(fpr_grid, mean, linewidth=2, label=label)
            ax.fill_between(fpr_grid, np.maximum(mean - std, 0), np.minimum(mean + std, 1), alpha=0.15)

        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title(title); ax.legend(loc="lower right"); fig.tight_layout()
        fig.savefig(out_png, dpi=160); plt.close(fig)

    @staticmethod
    def plot_overlays(per_fold_curves, out_png, title="Per-fold OVR ROC overlays"):
        if not per_fold_curves:
            return
        fig = plt.figure(figsize=(10, 8)); ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        for d in per_fold_curves:
            for _, (fpr, tpr, _) in d.items():
                ax.plot(fpr, tpr, alpha=0.25, linewidth=1)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title(title); fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)

    @staticmethod
    def plot_per_class_overlays(per_fold_curves, out_dir: Path, class_names=None, head_name: str = "fused"):
        if not per_fold_curves:
            return
        keys = sorted({k for d in per_fold_curves for k in d.keys()})
        if class_names is not None and len(class_names) == len(keys):
            name_map = {k: class_names[i] for i, k in enumerate(keys)}
        else:
            name_map = {k: f"class_{k}" for k in keys}

        out_dir.mkdir(parents=True, exist_ok=True)
        for k in keys:
            per_fold = []
            for fold_idx, d in enumerate(per_fold_curves, start=1):
                if k not in d:
                    continue
                fpr, tpr, auc_val = d[k]
                per_fold.append((fold_idx, fpr, tpr, auc_val))
            if not per_fold:
                continue

            fig = plt.figure(figsize=(10, 8)); ax = fig.add_subplot(111)
            ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="grey")
            for fold_idx, fpr, tpr, auc_val in per_fold:
                label = f"Fold {fold_idx} (AUC {auc_val:.3f})"
                ax.plot(fpr, tpr, linewidth=1.5, label=label)
            ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
            ax.set_title(f"{head_name} head — {name_map[k]} ROC per fold")
            ax.legend(loc="lower right", frameon=True)
            fig.tight_layout()
            safe_name = name_map[k].replace(" ", "_")
            fig.savefig(out_dir / f"roc_{head_name}_{safe_name}_perfold.png", dpi=160)
            plt.close(fig)

    @staticmethod
    def move_if_exists(src: Path, dest: Path):
        if src.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))

    @staticmethod
    def move_dir_overwrite(src: Path, dest: Path):
        """Move directory, replacing destination if it already exists."""
        if not src.exists():
            return
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))

    @staticmethod
    def load_holdout_roc_curves(dir_path: Path, head: str):
        """
        Load the holdout ROC JSON for a given head from a directory like
        foldX_roc_curves_holdout_best, returning the per-class curve map
        expected by plotting helpers.
        """
        if not dir_path.exists() or not dir_path.is_dir():
            return {}
        head_name = {"image": "image", "metadata": "metadata"}.get(head, "fused")
        best_path = None
        best_epoch = -1
        for path in dir_path.glob(f"epoch*_holdout_{head_name}.json"):
            m = re.match(r"epoch(\\d+)_", path.stem)
            if not m:
                continue
            try:
                epoch_idx = int(m.group(1))
            except Exception:
                continue
            if epoch_idx > best_epoch:
                best_epoch = epoch_idx
                best_path = path
        if best_path is None:
            return {}
        try:
            data = json.loads(best_path.read_text())
        except Exception:
            return {}
        per_class = data.get("per_class") or {}
        curves = {}
        for k, vals in per_class.items():
            if not isinstance(vals, dict):
                continue
            fpr = vals.get("fpr"); tpr = vals.get("tpr"); auc_val = vals.get("auc")
            if fpr is None or tpr is None or auc_val is None:
                continue
            try:
                idx = int(k)
            except Exception:
                continue
            curves[idx] = (np.array(fpr, dtype=float), np.array(tpr, dtype=float), float(auc_val))
        return curves

    def _run_impl(self) -> None:
        args = self.args
        args.num_classes = 2 if args.eval_mode == "binary" else 3

        if getattr(args, "run_id", None):
            run_id = str(args.run_id)
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            run_id = f"{args.shortname}_{ts}" if args.shortname else ts
        run_dir = Path("analysis_data") / args.shortname / run_id
        (run_dir / "plots").mkdir(parents=True, exist_ok=True)
        base_models_dir = Path("models") / args.shortname / run_id
        base_models_dir.mkdir(parents=True, exist_ok=True)
        self._export_run_settings(run_dir)

        clinical = build_papila_clinical(
            args.image_dir,
            args.clinical_dir,
            args.label_col,
            args.cat_cols,
            n_splits=args.n_splits,
            random_seed=args.fold_seed,
        )

        holdout_df = None
        if args.holdout_per_class > 0:
            df_full = clinical.df.copy()
            if args.eval_mode == "binary":
                df_full = df_full[df_full[args.label_col].isin([0, 1])].reset_index(drop=True)
            rng = np.random.default_rng(args.holdout_seed)
            holdout_indices = []
            for label, group in df_full.groupby(args.label_col):
                n = min(args.holdout_per_class, len(group))
                if n <= 0:
                    continue
                selected = rng.choice(group.index.to_numpy(), size=n, replace=False)
                holdout_indices.extend(selected.tolist())
            if holdout_indices:
                holdout_indices = sorted(set(holdout_indices))
                holdout_df = df_full.loc[holdout_indices].reset_index(drop=True)
                train_df = df_full.drop(index=holdout_indices).reset_index(drop=True)
                clinical.frames = [train_df.copy()]
                clinical.df = train_df.copy()
                clinical._infer_or_validate_feature_types()
                clinical._compute_numeric_stats()
                clinical._build_cat_maps()
                clinical._compute_feature_dim()
                clinical._build_kfold_indices()
                holdout_path = run_dir / "holdout.csv"
                holdout_df.to_csv(holdout_path, index=False)
                print(f"[run_multifold] Reserved holdout set of {len(holdout_df)} samples (saved to {holdout_path})")

        fold_macro_aucs = []
        per_fold_ovr_curves_for_plot_head = []
        holdout_per_fold_ovr_curves_for_plot_head = []

        def _default_monitor():
            if args.early_metric:
                mode = getattr(args, "early_mode", "auto")
                if mode == "auto":
                    mode = "min" if "loss" in args.early_metric.lower() else "max"
                return args.early_metric, mode
            if args.fusion_mode == "image_only":
                return "auc_img", "max"
            if args.fusion_mode == "metadata_only":
                return "auc_md", "max"
            return "auc_fused", "max"

        monitor_name, monitor_mode = _default_monitor()
        fold_summaries = []
        best_metric_values = []

        for fold in range(args.n_splits):
            print(f"\n=== Fold {fold+1}/{args.n_splits} ===")
            if args.early_metric:
                early_metric = args.early_metric
            else:
                if args.fusion_mode == "image_only":
                    early_metric = "auc_img"
                elif args.fusion_mode == "metadata_only":
                    early_metric = "auc_md"
                else:
                    early_metric = "auc_fused"

            ht_args = SimpleNamespace(
                image_dir=args.image_dir,
                clinical_dir=args.clinical_dir,
                label_col=args.label_col,
                cat_cols=args.cat_cols,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                num_classes=args.num_classes,
                img_augment=args.img_augment,
                focal_gamma=args.focal_gamma,
                eval_mode=args.eval_mode,
                fold=fold,
                run_dir=str(run_dir),
                models_dir=str((base_models_dir / f"fold{fold}").resolve()),
                backbone=args.backbone,
                freeze_ratio=args.freeze_ratio,
                fusion_mode=args.fusion_mode,
                use_se=args.use_se,
                se_reduction=args.se_reduction,
                se_pre_norm=args.se_pre_norm,
                se_where=args.se_where,
                se_reduction_tower=args.se_reduction_tower,
                se_pre_norm_tower=args.se_pre_norm_tower,
                warmup_tower_epochs=args.warmup_tower_epochs,
                warmup_fused_epochs=args.warmup_fused_epochs,
                gradual_thaw=args.gradual_thaw,
                thaw_phase_duration=args.thaw_phase_duration,
                thaw_ratio=args.thaw_ratio,
                thaw_target=args.thaw_target,
                thaw_start_epoch=args.thaw_start_epoch,
                initial_freeze=args.initial_freeze,
                bcd_prob=0.5,
                bcd_p0=0.20,
                bcd_min=0.05,
                bcd_max=0.30,
                bcd_k=0.4,
                bcd_metric="auc",
                bcd_alpha_batch=0.2,
                bcd_alpha_tower=0.3,
                bcd_explore_floor=0.15,
                aux_img=0.05,
                aux_md=0.05,
                aux_detach=True,
                ema_alpha=0.9,
                entropy_ema=0.7,
                early_stop=args.early_stop,
                early_metric=early_metric,
                early_mode=args.early_mode,
                early_patience=args.early_patience,
                early_min_delta=args.early_min_delta,
                checkpoint_best=args.checkpoint_best,
                holdout_df=holdout_df,
                img_crop_manifest=args.img_crop_manifest,
                img_crop_weights=args.img_crop_weights,
                img_crop_normalize=args.img_crop_normalize,
                img_crop_threshold=args.img_crop_threshold,
                img_crop_scale=args.img_crop_scale,
                img_crop_size=args.img_crop_size,
                img_crop_cache=args.img_crop_cache,
                img_crop_tta=args.img_crop_tta,
                img_crop_gt=args.img_crop_gt,
            )

            (base_models_dir / f"fold{fold}").mkdir(parents=True, exist_ok=True)
            ht = HyperTower(clinical, ht_args)
            ht.train()

            epoch_log_path = run_dir / "epoch_log.csv"
            best_row_data = None
            best_value = None
            if epoch_log_path.exists():
                try:
                    with epoch_log_path.open("r", newline="", encoding="utf-8") as fp:
                        reader = csv.DictReader(fp)
                        for row in reader:
                            val_raw = row.get(monitor_name)
                            try:
                                val = float(val_raw)
                            except (TypeError, ValueError):
                                continue
                            if best_value is None:
                                best_value = val
                                best_row_data = dict(row)
                            else:
                                if monitor_mode == "max":
                                    if val > best_value:
                                        best_value = val
                                        best_row_data = dict(row)
                                else:
                                    if val < best_value:
                                        best_value = val
                                        best_row_data = dict(row)
                except OSError as exc:  # pragma: no cover
                    print(f"[run_multifold] Warning: failed to read {epoch_log_path} ({exc}); skipping best-metric parse.")

            def _coerce_types(row: dict | None) -> dict | None:
                if row is None:
                    return None
                out = {}
                for key, value in row.items():
                    if value is None or value == "":
                        out[key] = None
                        continue
                    try:
                        out[key] = float(value)
                        if key == "epoch":
                            out[key] = int(float(value))
                    except ValueError:
                        out[key] = value
                return out

            best_row_converted = _coerce_types(best_row_data)
            if isinstance(best_row_converted, dict) and "epoch" in best_row_converted:
                try:
                    best_epoch = int(best_row_converted["epoch"])
                except Exception:
                    best_epoch = None
            else:
                best_epoch = None

            if best_value is not None:
                best_metric_values.append(best_value)

            fold_summaries.append({
                "fold": fold,
                "best_metric_value": best_value,
                "best_epoch": best_epoch,
                "monitor": monitor_name,
                "warmup_tower_epochs": int(getattr(ht, "warmup_tower_epochs", getattr(args, "warmup_tower_epochs", 2))),
                "warmup_fused_epochs": int(getattr(ht, "warmup_fused_epochs", getattr(args, "warmup_fused_epochs", 3))),
                "main_epochs": int(getattr(ht, "epochs", args.epochs)),
                "total_epochs": int(getattr(ht, "total_epochs", args.epochs)),
                "stats": best_row_converted,
            })

            self.move_if_exists(run_dir / "train.log", run_dir / f"fold{fold}_train.log")
            self.move_if_exists(epoch_log_path, run_dir / f"fold{fold}_epoch_log.csv")
            roc_src = run_dir / "roc_curves"
            if roc_src.exists() and roc_src.is_dir():
                self.move_dir_overwrite(roc_src, run_dir / f"fold{fold}_roc_curves")
            roc_best_src = run_dir / "roc_curves_best"
            if roc_best_src.exists() and roc_best_src.is_dir():
                self.move_dir_overwrite(roc_best_src, run_dir / f"fold{fold}_roc_curves_best")
            roc_holdout_src = run_dir / "roc_curves_holdout_best"
            if roc_holdout_src.exists() and roc_holdout_src.is_dir():
                self.move_dir_overwrite(roc_holdout_src, run_dir / f"fold{fold}_roc_curves_holdout_best")
            holdout_curves = self.load_holdout_roc_curves(run_dir / f"fold{fold}_roc_curves_holdout_best", args.plot_head)
            if holdout_curves:
                holdout_per_fold_ovr_curves_for_plot_head.append(holdout_curves)

            # Reload the recorded best checkpoint so downstream metrics/plots use the same epoch as the summary.
            best_snapshot = Path(ht_args.models_dir) / "model_best.pt"
            if best_snapshot.exists():
                try:
                    try:
                        state = torch.load(best_snapshot, map_location=ht.device, weights_only=False)
                    except TypeError:
                        state = torch.load(best_snapshot, map_location=ht.device)
                    ht._restore_from_state(state)
                except Exception as exc:  # pragma: no cover
                    print(f"[run_multifold] Warning: failed to reload best checkpoint for fold {fold}: {exc}")

            train_df, test_df = clinical.get_split_dfs(fold)
            y_true, p_fused, p_img, p_md = self.eval_collect_logits(ht)
            if args.eval_mode == "binary":
                keep = np.isin(y_true, [0, 1])
                if keep.sum() == 0:
                    raise RuntimeError("No binary samples left after filtering.")
                y_true = y_true[keep]
                if p_fused is not None: p_fused = p_fused[keep]
                if p_img is not None:   p_img   = p_img[keep]
                if p_md is not None:    p_md    = p_md[keep]

                def _slice2(p):
                    if p is None:
                        return None
                    if p.ndim == 2 and p.shape[1] >= 2:
                        return p[:, :2]
                    return p

                p_fused = _slice2(p_fused)
                p_img   = _slice2(p_img)
                p_md    = _slice2(p_md)

            np.save(run_dir / f"fold{fold}_y_true.npy", y_true)
            if p_img is not None:   np.save(run_dir / f"fold{fold}_probs_img.npy", p_img)
            if p_fused is not None: np.save(run_dir / f"fold{fold}_probs_fused.npy", p_fused)
            if p_md is not None:    np.save(run_dir / f"fold{fold}_probs_md.npy", p_md)

            if args.plot_head == "image":
                p_plot = p_img
            elif args.plot_head == "metadata":
                p_plot = p_md
            else:
                p_plot = p_fused

            fold_auc = self.auc_for(y_true, p_plot) if p_plot is not None else float("nan")
            fold_macro_aucs.append(fold_auc)

            per_fold_ovr_curves_for_plot_head.append(self.per_class_roc(y_true, p_plot))

        auc_mean = float(np.nanmean(fold_macro_aucs)) if fold_macro_aucs else float("nan")
        auc_std  = float(np.nanstd(fold_macro_aucs))  if fold_macro_aucs else float("nan")

        class_names = args.class_names if args.class_names else (
            ["Healthy", "Glaucoma"] if args.eval_mode == "binary" else ["Healthy","Glaucoma","Suspect"]
        )

        self.plot_per_class_overlays(
            per_fold_ovr_curves_for_plot_head,
            out_dir=run_dir / "plots",
            class_names=class_names,
            head_name=args.plot_head,
        )
        self.plot_mean_sd(
            per_fold_ovr_curves_for_plot_head,
            out_png=run_dir / "plots" / f"roc_{args.plot_head}_mean_ovr.png",
            class_names=class_names,
            title=f"Mean OVR ROC (±1 SD) — {args.plot_head} head"
        )

        if holdout_per_fold_ovr_curves_for_plot_head:
            holdout_plots_dir = run_dir / "plots" / "holdout"
            holdout_plots_dir.mkdir(parents=True, exist_ok=True)
            self.plot_per_class_overlays(
                holdout_per_fold_ovr_curves_for_plot_head,
                out_dir=holdout_plots_dir,
                class_names=class_names,
                head_name=f"{args.plot_head}_holdout",
            )
            self.plot_mean_sd(
                holdout_per_fold_ovr_curves_for_plot_head,
                out_png=holdout_plots_dir / f"roc_{args.plot_head}_holdout_mean_ovr.png",
                class_names=class_names,
                title=f"Holdout Mean OVR ROC (±1 SD) — {args.plot_head} head",
            )

        if best_metric_values:
            best_metric_mean = float(np.mean(best_metric_values))
            best_metric_std = float(np.std(best_metric_values, ddof=0))
        else:
            best_metric_mean = None
            best_metric_std = None

        warmup_tower_used = (
            fold_summaries[0].get("warmup_tower_epochs")
            if fold_summaries else getattr(args, "warmup_tower_epochs", None)
        )
        warmup_fused_used = (
            fold_summaries[0].get("warmup_fused_epochs")
            if fold_summaries else getattr(args, "warmup_fused_epochs", None)
        )
        total_epochs_used = (
            fold_summaries[0].get("total_epochs")
            if fold_summaries else args.epochs
        )

        summary = {
            "run_id": run_id,
            "backbone": args.backbone,
            "freeze_ratio": args.freeze_ratio,
            "fusion_mode": args.fusion_mode,
            "epochs": args.epochs,
            "warmup_tower_epochs": warmup_tower_used,
            "warmup_fused_epochs": warmup_fused_used,
            "total_epochs": total_epochs_used,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "num_classes": args.num_classes,
            "eval_mode": args.eval_mode,
            "n_splits": args.n_splits,
            "focal_gamma": args.focal_gamma,
            "balanced_sampler": bool(args.balanced_sampler),
            "se": {
                "enabled": bool(args.use_se),
                "reduction": int(args.se_reduction),
                "pre_norm": bool(args.se_pre_norm),
            },
            "best_metric": monitor_name,
            "best_metric_mode": monitor_mode,
            "best_metric_mean": best_metric_mean,
            "best_metric_std": best_metric_std,
            "fold_metrics": fold_summaries,
        }

        with open(run_dir / "summary.json", "w") as jf:
            json.dump(summary, jf, indent=2)
        print(f"Summary written to {run_dir / 'summary.json'}")


class _StreamCallback(io.TextIOBase):
    def __init__(self, callback):
        self.callback = callback

    def write(self, s):
        if self.callback and s:
            self.callback(s)
        return len(s)

    def flush(self):
        pass


class MultifoldRunner:
    """Wrapper used by the GUI to execute Multifold runs."""

    def run(self, cli_args: list[str], callback=None) -> None:
        parser = Multifold.build_parser()
        if callback:
            sink = _StreamCallback(callback)
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                args = parser.parse_args(cli_args)
        else:
            args = parser.parse_args(cli_args)
        multifold = Multifold(args)
        multifold.run(callback)


class MultifoldFrontend(tk.Tk):
    """Simple Tkinter GUI for configuring and launching run_multifold.py."""

    def __init__(self) -> None:
        super().__init__()
        self.title("run_multifold.py")
        self.vars: Dict[str, tk.Variable] = {}
        self._configure_scale()
        self.geometry(self.window_geometry)
        self._build_form()
        self._build_output()
        self.status_var = tk.StringVar(value="Idle")
        status_frame = tk.Frame(self)
        status_frame.pack(fill="x", padx=6, pady=(0, 6))
        tk.Label(status_frame, text="Status:", font=self.base_font).pack(side="left")
        tk.Label(status_frame, textvariable=self.status_var, anchor="w", font=self.base_font).pack(side="left", fill="x")
        self.force_cpu = False
        self.after(0, self.on_device_change)
        self.repo_root = Path(__file__).resolve().parents[1]
        self.current_process: Optional[subprocess.Popen] = None
        self.current_thread: Optional[threading.Thread] = None
        self.current_run_dir: Optional[Path] = None

    # ---- UI helpers -------------------------------------------------
    def _configure_scale(self) -> None:
        try:
            screen_w = self.winfo_screenwidth()
            screen_h = self.winfo_screenheight()
        except Exception:
            screen_w, screen_h = 1920, 1080

        scale = screen_w / 1920.0
        scale = max(0.8, min(scale, 1.6))
        base_size = max(10, int(10 * scale))
        self.base_font = ("TkDefaultFont", base_size)
        self.bold_font = ("TkDefaultFont", max(base_size, 11), "bold")
        self.entry_font = ("TkDefaultFont", max(9, base_size))
        self.button_font = ("TkDefaultFont", max(9, base_size - 1))
        self.mono_font = ("Courier", max(9, base_size))
        self.text_height = max(18, int(24 * scale))
        width = int(720 * scale)
        height = int(860 * scale)
        self.window_geometry = f"{width}x{height}"

        try:
            self.tk.call("tk", "scaling", scale)
        except Exception:
            pass

    def _entry(self, parent: tk.Widget, label: str, default: str = "") -> tk.Entry:
        frame = tk.Frame(parent)
        frame.pack(fill="x", padx=4, pady=2)
        tk.Label(frame, text=label, width=22, anchor="w", font=self.base_font).pack(side="left")
        var = self.vars.get(label)
        if not isinstance(var, tk.StringVar):
            var = tk.StringVar(value=default)
        entry = tk.Entry(frame, textvariable=var, font=self.entry_font)
        entry.pack(side="left", fill="x", expand=True)
        self.vars[label] = var
        return entry

    def _browse_entry(self, parent: tk.Widget, label: str, default: str = "", is_dir: bool = True) -> None:
        entry = self._entry(parent, label, default)

        def choose() -> None:
            path = filedialog.askdirectory() if is_dir else filedialog.askopenfilename()
            if path:
                entry.delete(0, tk.END)
                entry.insert(0, path)

        tk.Button(entry.master, text="Browse", command=choose, font=self.button_font).pack(side="left", padx=4)

    def _checkbox(self, parent: tk.Widget, label: str, default: bool = False) -> None:
        var = self.vars.get(label)
        if not isinstance(var, tk.BooleanVar):
            var = tk.BooleanVar(value=default)
        tk.Checkbutton(parent, text=label, variable=var, font=self.base_font).pack(anchor="w", padx=6)
        self.vars[label] = var

    def _option_menu(self, parent: tk.Widget, label: str, options: list[str], default: str) -> None:
        frame = tk.Frame(parent)
        frame.pack(fill="x", padx=4, pady=2)
        tk.Label(frame, text=label, width=22, anchor="w", font=self.base_font).pack(side="left")
        var = self.vars.get(label)
        if not isinstance(var, tk.StringVar):
            var = tk.StringVar(value=default)
        menu = tk.OptionMenu(frame, var, *options)
        menu.configure(font=self.base_font)
        menu["menu"].configure(font=self.base_font)
        menu.pack(side="left", fill="x", expand=True)
        self.vars[label] = var

    def _on_eval_mode_change(self, *_args) -> None:
        mode_var = self.vars.get("Evaluation mode")
        num_var = self.vars.get("Number of classes")
        if isinstance(mode_var, tk.StringVar) and isinstance(num_var, tk.StringVar):
            num_var.set("2" if mode_var.get() == "binary" else "3")

    def _build_popup_section(self, title: str, builder, parent: Optional[tk.Widget] = None) -> None:
        container = parent if parent is not None else self
        frame = tk.Frame(container)
        frame.pack(fill="x", padx=6, pady=2)
        tk.Button(frame, text=title, command=lambda: self._open_popup(title, builder), font=self.button_font).pack(anchor="w")

    def _open_popup(self, title: str, builder) -> None:
        win = tk.Toplevel(self)
        win.title(title)
        win.transient(self)
        content = tk.Frame(win, padx=8, pady=8)
        content.pack(fill="both", expand=True)
        builder(content)
        tk.Button(content, text="Close", command=win.destroy, font=self.button_font).pack(pady=(8, 0))

    def _build_crop_settings(self, parent: tk.Widget) -> None:
        self._browse_entry(parent, "Cropping mask manifest", "manifest.csv", is_dir=False)
        self._browse_entry(parent, "Cropping weights (optional)", "", is_dir=False)
        self._option_menu(parent, "Crop normalization", ["none", "imagenet", "per_image"], "per_image")
        self._entry(parent, "Crop threshold", "0.5")
        self._entry(parent, "Crop scale", "2.5")
        self._entry(parent, "Crop size", "224")
        self._entry(parent, "Crop cache directory", "analysis_data/hypertower_crops")
        self._checkbox(parent, "Use ground truth masks", True)
        self._checkbox(parent, "Use crop TTA", False)
        self._checkbox(parent, "Append geometry features", False)

    def _build_se_settings(self, parent: tk.Widget) -> None:
        self._checkbox(parent, "Enable SE", True)
        self._entry(parent, "SE reduction (bridge)", "16")
        self._checkbox(parent, "SE pre-norm (bridge)", True)
        self._option_menu(parent, "SE location", ["bridge", "tower", "both", "none"], "bridge")
        self._entry(parent, "SE reduction (tower)", "16")
        self._checkbox(parent, "SE pre-norm (tower)", True)

    def _build_warmup_settings(self, parent: tk.Widget) -> None:
        self._entry(parent, "Tower warmup epochs", "2")
        self._entry(parent, "Fused warmup epochs", "3")
        self._checkbox(parent, "Enable gradual thaw", False)
        self._entry(parent, "Thaw phase duration", "5")
        self._entry(parent, "Thaw ratio", "0.33")
        self._option_menu(parent, "Thaw target", ["image", "metadata", "both"], "image")
        self._entry(parent, "Thaw start epoch", "-1")
        self._checkbox(parent, "Initial freeze before thaw", False)

    def _build_early_stop_settings(self, parent: tk.Widget) -> None:
        self._entry(parent, "Early metric", "")
        self._option_menu(parent, "Early mode", ["auto", "min", "max"], "auto")
        self._entry(parent, "Early patience", "7")
        self._entry(parent, "Early min delta", "0.0")
        self._checkbox(parent, "Monitor holdout for early stop", False)
        self._checkbox(parent, "Save best checkpoint", False)

    # ---- Layout -----------------------------------------------------
    def _build_form(self) -> None:
        form = tk.Frame(self)
        form.pack(fill="both", expand=False)

        self._entry(form, "Run name", "gui_run")
        self._browse_entry(form, "Fundus image directory", "Papila/FundusImages")
        self._browse_entry(form, "Clinical data directory", "Papila/ClinicalData")
        self._entry(form, "Diagnosis column", "Diagnosis")
        self._entry(form, "Categorical columns", "Gender,Phakic/Pseudophakic")

        self._option_menu(form, "Backbone architecture", BACKBONES, BACKBONES[0])
        self._option_menu(form, "Fusion mode", FUSION_MODES, "image_only")
        self._option_menu(form, "Evaluation mode", EVAL_MODES, "binary")

        self._entry(form, "Epochs", "40")
        self._entry(form, "Batch size", "8")
        self._entry(form, "Learning rate", "5e-5")
        self._entry(form, "Number of classes", "2")
        self._entry(form, "Number of folds", "5")
        self._entry(form, "Fold seed", "42")
        self._entry(form, "Holdout per class", "8")
        self._entry(form, "Holdout seed", "123")

        eval_var = self.vars.get("Evaluation mode")
        if isinstance(eval_var, tk.StringVar):
            eval_var.trace_add("write", self._on_eval_mode_change)
            self._on_eval_mode_change()

        tk.Label(form, text="Options", font=self.bold_font).pack(anchor="w", padx=6, pady=(8, 0))
        self._checkbox(form, "Disable image augmentation", False)
        self._checkbox(form, "Use balanced sampler", False)
        self._checkbox(form, "Enable early stopping", False)
        self._checkbox(form, "Use image cropping", True)
        self._checkbox(form, "Use focal loss", False)
        self._entry(form, "Focal gamma", "0.0")

        dummy = tk.Frame(self)
        self._build_crop_settings(dummy)
        self._build_se_settings(dummy)
        self._build_warmup_settings(dummy)
        self._build_early_stop_settings(dummy)
        dummy.destroy()

        self._build_popup_section("Cropping Settings", self._build_crop_settings, parent=form)
        self._build_popup_section("Squeeze-and-Excitation Settings", self._build_se_settings, parent=form)
        self._build_popup_section("Warmup / Thaw Settings", self._build_warmup_settings, parent=form)
        self._build_popup_section("Early Stop Settings", self._build_early_stop_settings, parent=form)

        device_frame = tk.Frame(form)
        device_frame.pack(fill="x", pady=4, padx=6)
        tk.Label(device_frame, text="Compute Device", font=self.bold_font).pack(anchor="w")
        self.device_var = tk.StringVar(value="gpu")
        tk.Radiobutton(device_frame, text="GPU", variable=self.device_var, value="gpu",
                       command=self.on_device_change, font=self.base_font).pack(anchor="w")
        tk.Radiobutton(device_frame, text="CPU", variable=self.device_var, value="cpu",
                       command=self.on_device_change, font=self.base_font).pack(anchor="w")

        btn_frame = tk.Frame(form)
        btn_frame.pack(fill="x", pady=8)
        self.run_button = tk.Button(btn_frame, text="Run", command=self.run_command, font=self.button_font)
        self.run_button.pack(side="left", padx=4)
        self.stop_button = tk.Button(btn_frame, text="Stop", command=self.stop_command, font=self.button_font, state="disabled")
        self.stop_button.pack(side="left", padx=4)
        tk.Button(btn_frame, text="Preview Command", command=self.preview_command, font=self.button_font).pack(side="left", padx=4)
        tk.Button(btn_frame, text="Export Settings", command=self.export_settings, font=self.button_font).pack(side="right", padx=4)
        tk.Button(btn_frame, text="Import Settings", command=self.import_settings, font=self.button_font).pack(side="right", padx=4)

    def _build_output(self) -> None:
        tk.Label(self, text="Command / Output", font=self.bold_font).pack(anchor="w", padx=6)
        self.output = tk.Text(self, height=self.text_height, font=self.mono_font)
        self.output.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self.output.configure(state="disabled")

    # ---- Run management helpers ------------------------------------
    def _compute_run_id_and_dir(self) -> tuple[str, Path]:
        shortname = ""
        short_var = self.vars.get("Run name")
        if isinstance(short_var, tk.StringVar):
            shortname = short_var.get().strip()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_id = f"{shortname}_{timestamp}" if shortname else timestamp
        base = Path("analysis_data")
        if shortname:
            base = base / shortname
        run_dir = base / run_id
        return run_id, run_dir

    def _auto_export_settings(self, run_dir: Path) -> None:
        data = {}
        for key, var in self.vars.items():
            try:
                data[key] = var.get()
            except Exception:
                pass
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            with open(run_dir / "gui_settings.json", "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            self.append_output(f"[GUI] Settings saved to {run_dir / 'gui_settings.json'}\n")
        except Exception as exc:  # pragma: no cover
            self.append_output(f"[GUI] Failed to save GUI settings: {exc}\n")

    def _compose_command(self, include_run_id: bool = False) -> tuple[list[str], Optional[str], Optional[Path]]:
        cli_args = self.build_cli_args()
        run_id = None
        run_dir = None
        if include_run_id:
            run_id, run_dir = self._compute_run_id_and_dir()
            cli_args = cli_args + ["--run-id", run_id]
        cmd = [sys.executable, "-u", "scripts/run_multifold.py", *cli_args]
        return cmd, run_id, run_dir

    def _on_process_finished(self, exit_code: Optional[int], error: Optional[Exception]) -> None:
        self.current_process = None
        self.current_thread = None
        self.current_run_dir = None
        self.stop_button.config(state="disabled")
        self.run_button.config(state="normal")
        if error is not None:
            self.append_output(f"\n[GUI] Error: {error}\n")
            self.set_status("Error")
            return
        if exit_code is None:
            self.append_output("\nProcess finished.\n")
            self.set_status("Finished")
            return
        self.append_output(f"\nProcess finished with exit code {exit_code}\n")
        self.set_status("Finished (exit 0)" if exit_code == 0 else f"Finished (exit {exit_code})")

    def _force_terminate_if_running(self) -> None:
        proc = self.current_process
        if proc is None or proc.poll() is not None:
            return
        self.append_output("[GUI] Process still running after interrupt; terminating...\n")
        try:
            proc.terminate()
        except Exception as exc:  # pragma: no cover
            self.append_output(f"[GUI] Failed to terminate process: {exc}\n")
        self.after(4000, self._kill_process)

    def _kill_process(self) -> None:
        proc = self.current_process
        if proc is None or proc.poll() is not None:
            return
        self.append_output("[GUI] Forcing process kill.\n")
        try:
            proc.kill()
        except Exception as exc:  # pragma: no cover
            self.append_output(f"[GUI] Failed to kill process: {exc}\n")

    def stop_command(self) -> None:
        proc = self.current_process
        if proc is None or proc.poll() is not None:
            self.stop_button.config(state="disabled")
            return
        self.append_output("\n[GUI] Sending interrupt signal...\n")
        self.set_status("Stopping...")
        try:
            if os.name == "nt":
                ctrl_break = getattr(signal, "CTRL_BREAK_EVENT", signal.SIGINT)
                proc.send_signal(ctrl_break)
            else:
                proc.send_signal(signal.SIGINT)
        except Exception as exc:  # pragma: no cover
            self.append_output(f"[GUI] Failed to send interrupt: {exc}\n")
        self.stop_button.config(state="disabled")
        self.after(4000, self._force_terminate_if_running)

    # ---- Output helpers ---------------------------------------------
    def append_output(self, text: str) -> None:
        self.after(0, self._append_output, text)

    def _append_output(self, text: str) -> None:
        self.output.configure(state="normal")
        self.output.insert(tk.END, text)
        self.output.see(tk.END)
        self.output.configure(state="disabled")

    def set_status(self, text: str) -> None:
        self.after(0, self.status_var.set, text)

    def preview_command(self) -> None:
        cmd_list, run_id, run_dir = self._compose_command(include_run_id=True)
        cmd = " ".join(cmd_list)
        self.output.configure(state="normal")
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, cmd + "\n")
        if run_dir is not None:
            self.output.insert(tk.END, f"# output directory: {run_dir}\n")
        if run_id is not None:
            self.output.insert(tk.END, f"# run id: {run_id}\n")
        self.output.configure(state="disabled")

    # ---- Command execution ------------------------------------------
    def run_command(self) -> None:
        if self.current_process and self.current_process.poll() is None:
            messagebox.showwarning("Run in progress", "A run is already in progress.")
            return

        cmd_list, run_id, run_dir = self._compose_command(include_run_id=True)
        self.output.configure(state="normal")
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, "Running: " + " ".join(cmd_list) + "\n\n")
        if run_dir is not None:
            self.output.insert(tk.END, f"# output directory: {run_dir}\n\n")
        self.output.configure(state="disabled")
        self.set_status("Running")
        self.run_button.config(state="disabled")
        self.stop_button.config(state="normal")

        env = os.environ.copy()
        if self.force_cpu:
            env["CUDA_VISIBLE_DEVICES"] = "-1"

        self.current_run_dir = run_dir
        if run_dir is not None:
            self._auto_export_settings(run_dir)

        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0

        def worker() -> None:
            exit_code: Optional[int] = None
            error: Optional[Exception] = None
            try:
                proc = subprocess.Popen(
                    cmd_list,
                    cwd=self.repo_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    env=env,
                    creationflags=creationflags,
                )
                self.current_process = proc
                assert proc.stdout is not None
                for line in proc.stdout:
                    if not line:
                        break
                    self.append_output(line)
                proc.stdout.close()
                exit_code = proc.wait()
            except Exception as exc:  # pragma: no cover
                error = exc
            finally:
                self.after(0, lambda: self._on_process_finished(exit_code, error))

        self.current_thread = threading.Thread(target=worker, daemon=True)
        self.current_thread.start()

    # ---- Command builder --------------------------------------------
    def build_cli_args(self) -> list[str]:
        args_list: list[str] = []

        def add(flag: str, var_name: str, allow_empty: bool = False) -> None:
            var = self.vars.get(var_name)
            if isinstance(var, tk.StringVar):
                value = var.get().strip()
                if value or allow_empty:
                    args_list.extend([flag, value])

        add("--shortname", "Run name")
        add("--image-dir", "Fundus image directory")
        add("--clinical-dir", "Clinical data directory")
        add("--label-col", "Diagnosis column")

        cats = self.vars["Categorical columns"].get().strip()
        if cats:
            args_list.extend(["--cat-cols", *[c.strip() for c in cats.split(",") if c.strip()]])

        add("--backbone", "Backbone architecture")
        add("--fusion-mode", "Fusion mode")
        add("--epochs", "Epochs")
        add("--batch-size", "Batch size")
        add("--lr", "Learning rate")
        add("--num-classes", "Number of classes")
        add("--n-splits", "Number of folds")
        add("--fold-seed", "Fold seed")
        add("--holdout-per-class", "Holdout per class")
        add("--holdout-seed", "Holdout seed")

        eval_mode = self.vars["Evaluation mode"].get()
        if eval_mode:
            args_list.extend(["--eval_mode", eval_mode])

        use_crop_var = self.vars.get("Use image cropping")
        if isinstance(use_crop_var, tk.BooleanVar) and use_crop_var.get():
            add("--img-crop-manifest", "Cropping mask manifest")
            weights = self.vars["Cropping weights (optional)"].get().strip()
            if weights:
                args_list.extend(["--img-crop-weights", weights])
            norm_var = self.vars.get("Crop normalization")
            if isinstance(norm_var, tk.StringVar):
                norm = norm_var.get().strip()
                if norm:
                    args_list.extend(["--img-crop-normalize", norm])
            add("--img-crop-threshold", "Crop threshold")
            add("--img-crop-cache", "Crop cache directory")
            add("--img-crop-scale", "Crop scale")
            add("--img-crop-size", "Crop size")

            if self.vars["Use ground truth masks"].get():
                args_list.append("--img-crop-gt")
            if self.vars["Use crop TTA"].get():
                args_list.append("--img-crop-tta")
            if self.vars["Append geometry features"].get():
                args_list.append("--img-geometry-features")

        if self.vars["Disable image augmentation"].get():
            args_list.append("--no-img-augment")
        if self.vars["Use balanced sampler"].get():
            args_list.append("--balanced-sampler")
        if self.vars["Enable early stopping"].get():
            args_list.append("--early-stop")

        if self.vars.get("Use focal loss") and self.vars["Use focal loss"].get():
            focal_gamma = self.vars["Focal gamma"].get().strip()
            if focal_gamma:
                args_list.extend(["--focal-gamma", focal_gamma])

        if not self.vars["Enable SE"].get():
            args_list.append("--no-se")
        else:
            add("--se-reduction", "SE reduction (bridge)")
            if not self.vars["SE pre-norm (bridge)"].get():
                args_list.append("--no-se-pre-norm")
            se_where = self.vars["SE location"].get().strip()
            if se_where:
                args_list.extend(["--se-where", se_where])
            add("--se-reduction-tower", "SE reduction (tower)")
            if not self.vars["SE pre-norm (tower)"].get():
                args_list.append("--no-se-pre-norm-tower")

        add("--warmup-tower-epochs", "Tower warmup epochs")
        add("--warmup-fused-epochs", "Fused warmup epochs")
        if self.vars["Enable gradual thaw"].get():
            args_list.append("--gradual-thaw")
        add("--thaw-phase-duration", "Thaw phase duration")
        add("--thaw-ratio", "Thaw ratio")
        thaw_target = self.vars["Thaw target"].get().strip()
        if thaw_target:
            args_list.extend(["--thaw-target", thaw_target])
        add("--thaw-start-epoch", "Thaw start epoch")
        if self.vars["Initial freeze before thaw"].get():
            args_list.append("--initial-freeze")

        early_metric = self.vars["Early metric"].get().strip()
        if early_metric:
            args_list.extend(["--early-metric", early_metric])
        early_mode = self.vars["Early mode"].get().strip()
        if early_mode:
            args_list.extend(["--early-mode", early_mode])
        add("--early-patience", "Early patience")
        add("--early-min-delta", "Early min delta")
        if self.vars["Monitor holdout for early stop"].get():
            args_list.append("--early-monitor-holdout")
        if self.vars["Save best checkpoint"].get():
            args_list.append("--checkpoint-best")

        return args_list

    def build_command(self) -> list[str]:
        return [sys.executable, "-u", "scripts/run_multifold.py", *self.build_cli_args()]

    def on_device_change(self) -> None:
        choice = getattr(self, "device_var", None)
        if choice is None:
            return
        choice = self.device_var.get()
        if choice == "cpu":
            self.force_cpu = True
            self.append_output("CPU selected. Forcing CPU usage.\n")
            self.set_status("CPU selected")
        else:
            if not torch.cuda.is_available():
                self.append_output("GPU selected but CUDA is not available. Falling back to CPU.\n")
                self.device_var.set("cpu")
                self.force_cpu = True
                self.set_status("GPU unavailable; CPU selected")
                return
            try:
                x = torch.rand((2048,), device="cuda")
                y = torch.rand((2048,), device="cuda")
                _ = (x * y).sum().item()
                self.append_output("GPU support confirmed.\n")
                self.force_cpu = False
                self.set_status("GPU selected")
            except Exception as exc:
                self.append_output(f"GPU self-test failed ({exc}). Falling back to CPU.\n")
                self.device_var.set("cpu")
                self.force_cpu = True
                self.set_status("GPU test failed; CPU selected")

    # ---- Settings import / export ----------------------------------
    def export_settings(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Export Settings",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        data = {}
        for key, var in self.vars.items():
            try:
                data[key] = var.get()
            except Exception:
                pass
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            self.set_status(f"Settings exported to {path}")
        except Exception as exc:  # pragma: no cover
            messagebox.showerror("Export failed", str(exc))
            self.set_status("Export failed")

    def import_settings(self) -> None:
        path = filedialog.askopenfilename(
            title="Import Settings",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            messagebox.showerror("Import failed", str(exc))
            self.set_status("Import failed")
            return

        for key, value in data.items():
            var = self.vars.get(key)
            if var is None:
                continue
            try:
                if isinstance(var, tk.BooleanVar):
                    var.set(bool(value))
                else:
                    var.set(str(value))
            except Exception:
                continue
        self.set_status(f"Settings imported from {path}")
        self.on_device_change()


def launch_frontend() -> None:
    app = MultifoldFrontend()
    app.mainloop()


if __name__ == "__main__":
    launch_frontend()
