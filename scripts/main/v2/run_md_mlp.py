#!/usr/bin/env python3
"""
V2-parity metadata-only runner.

Goal:
- Match V2HyperTower single-model metadata-only behavior as closely as possible.
- Avoid image tower/image IO overhead in forward/training.

How parity is achieved:
- Uses PatientFirstSplitManager (same split policy).
- Uses PAPILA profile builders + V2 filters:
  - eye_train      = filter_eye_samples(...)
  - bilat_val/test = filter_bilateral_samples(...)
- Uses V2 training/eval helpers directly:
  - train_single_epoch(...)
  - collect_probs_single_components(...)
- Uses bridge_mode="metadata_only".

Implementation detail:
- Batch dictionaries still include image slots to satisfy shared V2 helpers,
  but these are tiny dummy tensors and are never consumed in metadata-only mode.

Outputs:
  analysis_data/{run_name}/{eval_mode}/{tower_mode}/fold{N}/
    y_true.npy
    probs_classic.npy or probs_ensemble.npy
    y_true_holdout.npy
    probs_classic_holdout.npy or probs_ensemble_holdout.npy
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# ensure repo root is on sys.path when run directly
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from classes.v2.bridges import Bridge
from classes.v2.loader_factory import (
    build_balanced_sampler,
    filter_bilateral_samples,
    filter_eye_samples,
)
from classes.v2.metrics import _score_arrays
from classes.v2.models import collect_probs_single_components, train_single_epoch
from classes.v2.papila_builders import build_papila_data
from classes.v2.profiles import build_papila_profile
from classes.v2.split_manager import PatientFirstSplitManager
from classes.v2.towers import MDTower
from classes.v2.utils import choose_device, seed_everything


class MetadataOnlySingleHT(nn.Module):
    """SingleEyeHT-compatible shell without real image tower usage."""

    def __init__(
        self,
        *,
        clinical_data,
        num_classes: int,
        md_hidden_dim: int,
        fusion_dim: int,
        dropout: float,
        use_se: bool,
        se_reduction: int,
        se_pre_norm: bool,
    ):
        super().__init__()
        # Placeholder module to satisfy phase toggling logic.
        self.img_tower = nn.Identity()
        self.md_tower = MDTower(
            clinical_data=clinical_data,
            hidden_dim=md_hidden_dim,
            dropout=dropout,
            use_se=use_se,
            se_reduction=se_reduction,
            se_pre_norm=se_pre_norm,
        )
        # img_dim is irrelevant in metadata_only mode, but Bridge defines img head params.
        self.bridge = Bridge(
            img_dim=1,
            meta_dim=self.md_tower.out_dim,
            num_classes=num_classes,
            fusion_dim=fusion_dim,
            mode="metadata_only",
            use_se=False,
            se_reduction=16,
            se_pre_norm=True,
        )


class EyeMetaDataset(Dataset):
    """Eye-level dataset for V2 train_single_epoch input contract."""

    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            "image_1": torch.zeros(1, dtype=torch.float32),
            "matrix_1": torch.as_tensor(s["matrix_1"], dtype=torch.float32),
            "label_1": torch.tensor(int(s["label_1"]), dtype=torch.long),
        }


class BilatMetaDataset(Dataset):
    """Patient-level bilateral dataset for collect_probs_single_components."""

    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            "image_1": torch.zeros(1, dtype=torch.float32),
            "image_2": torch.zeros(1, dtype=torch.float32),
            "matrix_1": torch.as_tensor(s["matrix_1"], dtype=torch.float32),
            "matrix_2": torch.as_tensor(s["matrix_2"], dtype=torch.float32),
            "label_1": torch.tensor(int(s["label_1"]), dtype=torch.long),
        }


def _phase_for_epoch(epoch_idx: int, warm_tower: int, warm_fused: int, main_epochs: int) -> tuple[str, int]:
    if epoch_idx < warm_tower:
        return "tower_warmup", 0
    if epoch_idx < (warm_tower + warm_fused):
        return "fused_warmup", 0
    if epoch_idx < (warm_tower + warm_fused + main_epochs):
        main_ep = epoch_idx - warm_tower - warm_fused + 1
        return "main", main_ep
    return "done", main_epochs


def _evaluate_single(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    aggregate_patient: bool,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    y, p_fused, _, p_md = collect_probs_single_components(
        model, loader, device, aggregate_patient=aggregate_patient
    )
    # In metadata_only mode p_fused == p_md; keep md explicitly for clarity.
    probs = p_md if p_md.size else p_fused
    acc, auc, _ = _score_arrays(y, probs, num_classes)
    return y, probs, float(auc), float(acc)


def main() -> None:
    ap = argparse.ArgumentParser(description="V2-parity metadata-only runner.")
    ap.add_argument("--eval-mode", required=True, choices=["binary", "multiclass"])
    ap.add_argument("--tower-mode", default="single", choices=["single", "ensemble"])
    ap.add_argument("--run-name", required=True)

    ap.add_argument("--epochs", type=int, default=40, help="Main-phase epochs.")
    ap.add_argument("--warmup-tower-epochs", type=int, default=None)
    ap.add_argument("--warmup-fused-epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--bcd-prob", type=float, default=0.5)

    ap.add_argument("--md-hidden-dim", type=int, default=128)
    ap.add_argument("--fusion-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--use-se", action="store_true")
    ap.add_argument("--se-reduction", type=int, default=16)
    ap.add_argument("--se-pre-norm", action="store_true")

    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--holdout-per-class", type=int, default=5)
    ap.add_argument("--holdout-seed", type=int, default=123)
    ap.add_argument("--fold-seed", type=int, default=42)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--balanced-sampling", action=argparse.BooleanOptionalAction, default=False)

    ap.add_argument("--analysis-dir", default="analysis_data")
    ap.add_argument("--image-dir", default="Papila/FundusImages")
    ap.add_argument("--clinical-dir", default="Papila/ClinicalData")
    ap.add_argument("--label-col", default="Diagnosis")
    ap.add_argument("--patient-col", default="Patient ID")
    ap.add_argument("--cat-cols", nargs="*", default=["Gender", "Phakic/Pseudophakic"])

    args = ap.parse_args()

    seed_everything(args.seed)
    device = choose_device(None)
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
    print(f"Loaded: {len(data.df)} rows  feature_dim={data.feature_dim}", flush=True)

    num_classes = 2 if args.eval_mode == "binary" else 3
    df_mode = data.df.copy()
    if args.eval_mode == "binary":
        df_mode = df_mode[df_mode[args.label_col].isin([0, 1])].reset_index(drop=True)
    print(f"[{args.eval_mode}] rows={len(df_mode)}", flush=True)

    class _ClinicalShim:
        label_col = args.label_col

        def __init__(self, df):
            self.df = df

    split_args = SimpleNamespace(
        eval_mode=args.eval_mode,
        holdout_per_class=args.holdout_per_class,
        holdout_seed=args.holdout_seed,
        n_splits=args.n_splits,
        fold_seed=args.fold_seed,
    )
    splitter = PatientFirstSplitManager(patient_col=args.patient_col, label_col=args.label_col)
    plans = splitter.build_plans(clinical=_ClinicalShim(df_mode), args=split_args)

    profile_eye = build_papila_profile(
        patient_col=args.patient_col,
        label_col=args.label_col,
        sample_mode="eye",
    )
    profile_patient = build_papila_profile(
        patient_col=args.patient_col,
        label_col=args.label_col,
        sample_mode="patient",
    )

    warm_tower = int(args.warmup_tower_epochs) if args.warmup_tower_epochs is not None else 2
    warm_fused = int(args.warmup_fused_epochs) if args.warmup_fused_epochs is not None else 2
    total_epochs = warm_tower + warm_fused + int(args.epochs)

    out_root = Path(args.analysis_dir) / args.run_name / args.eval_mode / args.tower_mode
    out_root.mkdir(parents=True, exist_ok=True)

    fold_metrics = []
    aggregate_patient = args.tower_mode == "ensemble"

    for fold_idx, split in enumerate(plans[: args.n_splits]):
        fold_dir = out_root / f"fold{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        eye_train = filter_eye_samples(profile_eye.build_samples(df=split.train, clinical=data))
        bilat_val = filter_bilateral_samples(profile_patient.build_samples(df=split.val, clinical=data))

        holdout_bilat = []
        if split.holdout is not None and not split.holdout.empty:
            holdout_bilat = filter_bilateral_samples(
                profile_patient.build_samples(df=split.holdout, clinical=data)
            )

        if not eye_train or not bilat_val:
            print(f"[fold {fold_idx+1}] skipped (eye_train={len(eye_train)} bilat_val={len(bilat_val)})", flush=True)
            fold_metrics.append(
                {
                    "fold": fold_idx,
                    "best_epoch": None,
                    "best_phase": None,
                    "val_auc": float("nan"),
                    "val_acc": float("nan"),
                    "hld_auc": float("nan"),
                    "hld_acc": float("nan"),
                    "eye_train_n": len(eye_train),
                    "bilat_val_n": len(bilat_val),
                    "bilat_holdout_n": len(holdout_bilat),
                }
            )
            continue

        sampler = build_balanced_sampler(eye_train) if args.balanced_sampling else None
        train_loader = DataLoader(
            EyeMetaDataset(eye_train),
            batch_size=args.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
        )
        val_loader = DataLoader(BilatMetaDataset(bilat_val), batch_size=args.batch_size, shuffle=False)
        holdout_loader = (
            DataLoader(BilatMetaDataset(holdout_bilat), batch_size=args.batch_size, shuffle=False)
            if holdout_bilat
            else None
        )

        model = MetadataOnlySingleHT(
            clinical_data=data,
            num_classes=num_classes,
            md_hidden_dim=args.md_hidden_dim,
            fusion_dim=args.fusion_dim,
            dropout=args.dropout,
            use_se=bool(args.use_se),
            se_reduction=int(args.se_reduction),
            se_pre_norm=bool(args.se_pre_norm),
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_auc = -1.0
        best_epoch = 0
        best_phase = ""
        best_state = None
        epoch_log_rows = []

        print(
            f"\n[fold {fold_idx+1}/{args.n_splits}] "
            f"eye_train_n={len(eye_train)} bilat_val_n={len(bilat_val)} "
            f"holdout_n={len(holdout_bilat)} warmup={warm_tower}+{warm_fused} total={total_epochs}",
            flush=True,
        )

        for ep in range(total_epochs):
            phase, main_ep = _phase_for_epoch(ep, warm_tower, warm_fused, int(args.epochs))
            tr_loss, tr_acc = train_single_epoch(
                model,
                train_loader,
                opt,
                device,
                phase=phase,
                bcd_prob=float(args.bcd_prob),
            )

            _, p_val, val_auc, val_acc = _evaluate_single(
                model,
                val_loader,
                device,
                num_classes,
                aggregate_patient=aggregate_patient,
            )

            hld_auc_ep = float("nan")
            hld_acc_ep = float("nan")
            if holdout_loader is not None:
                _, _, hld_auc_ep, hld_acc_ep = _evaluate_single(
                    model, holdout_loader, device, num_classes,
                    aggregate_patient=aggregate_patient,
                )

            is_main = phase == "main"
            if is_main and (not np.isnan(val_auc)) and val_auc > best_auc:
                best_auc = float(val_auc)
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = ep + 1
                best_phase = phase

            epoch_log_rows.append({
                "epoch": ep + 1,
                "phase": phase,
                "train_loss": float(tr_loss),
                "train_acc": float(tr_acc),
                "val_auc": float(val_auc),
                "val_acc": float(val_acc),
                "hld_auc": float(hld_auc_ep),
                "hld_acc": float(hld_acc_ep),
            })

            if ep == 0 or (ep + 1) % 10 == 0 or (ep + 1) == total_epochs:
                print(
                    f"  ep {ep+1:>3}/{total_epochs} [{phase}:{main_ep}/{args.epochs}] "
                    f"loss={tr_loss:.4f} acc={tr_acc:.4f} "
                    f"val_auc={val_auc:.4f} val_acc={val_acc:.4f} "
                    f"best_auc={best_auc:.4f}",
                    flush=True,
                )

        import pandas as _pd
        _pd.DataFrame(epoch_log_rows).to_csv(fold_dir / "epoch_log.csv", index=False)

        if best_state is not None:
            model.load_state_dict(best_state)

        y_val, p_val, val_auc, val_acc = _evaluate_single(
            model,
            val_loader,
            device,
            num_classes,
            aggregate_patient=aggregate_patient,
        )

        if args.tower_mode == "single":
            probs_name = "probs_classic.npy"
            probs_h_name = "probs_classic_holdout.npy"
        else:
            probs_name = "probs_ensemble.npy"
            probs_h_name = "probs_ensemble_holdout.npy"

        np.save(fold_dir / "y_true.npy", y_val)
        np.save(fold_dir / probs_name, p_val)

        hld_auc = float("nan")
        hld_acc = float("nan")
        if holdout_loader is not None:
            y_h, p_h, hld_auc, hld_acc = _evaluate_single(
                model,
                holdout_loader,
                device,
                num_classes,
                aggregate_patient=aggregate_patient,
            )
            np.save(fold_dir / "y_true_holdout.npy", y_h)
            np.save(fold_dir / probs_h_name, p_h)

        print(
            f"  [fold {fold_idx+1}] best_epoch={best_epoch} best_auc={best_auc:.4f} "
            f"val_auc={val_auc:.4f} val_acc={val_acc:.4f} "
            f"hld_auc={hld_auc:.4f} hld_acc={hld_acc:.4f}",
            flush=True,
        )

        fold_metrics.append(
            {
                "fold": fold_idx,
                "best_epoch": best_epoch,
                "best_phase": best_phase,
                "val_auc": float(val_auc),
                "val_acc": float(val_acc),
                "hld_auc": float(hld_auc),
                "hld_acc": float(hld_acc),
                "eye_train_n": len(eye_train),
                "bilat_val_n": len(bilat_val),
                "bilat_holdout_n": len(holdout_bilat),
            }
        )

    val_aucs = [m["val_auc"] for m in fold_metrics if not np.isnan(m["val_auc"])]
    hld_aucs = [m["hld_auc"] for m in fold_metrics if not np.isnan(m["hld_auc"])]
    if val_aucs:
        print(f"\nMean val AUC: {np.mean(val_aucs):.4f} ± {np.std(val_aucs):.4f}", flush=True)
    if hld_aucs:
        print(f"Mean hld AUC: {np.mean(hld_aucs):.4f} ± {np.std(hld_aucs):.4f}", flush=True)

    summary = {
        "run_name": args.run_name,
        "eval_mode": args.eval_mode,
        "tower_mode": args.tower_mode,
        "bridge_mode": "metadata_only",
        "model": "MetadataOnlySingleHT",
        "epochs": int(args.epochs),
        "warmup_tower_epochs": warm_tower,
        "warmup_fused_epochs": warm_fused,
        "md_hidden_dim": int(args.md_hidden_dim),
        "fusion_dim": int(args.fusion_dim),
        "dropout": float(args.dropout),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "bcd_prob": float(args.bcd_prob),
        "balanced_sampling": bool(args.balanced_sampling),
        "feature_dim": int(data.feature_dim),
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "fold_metrics": fold_metrics,
        "val_auc_mean": float(np.mean(val_aucs)) if val_aucs else None,
        "val_auc_std": float(np.std(val_aucs)) if val_aucs else None,
        "hld_auc_mean": float(np.mean(hld_aucs)) if hld_aucs else None,
        "hld_auc_std": float(np.std(hld_aucs)) if hld_aucs else None,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nOutputs written to: {out_root}", flush=True)


if __name__ == "__main__":
    main()
