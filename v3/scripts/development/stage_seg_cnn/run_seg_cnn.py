#!/usr/bin/env python
"""
Segmentation-map CNN — glaucoma grading from disc/cup label maps.

Trains a CNN whose input is the combined optic disc / cup segmentation map
(pixel values 0=bg, 1=disc_rim, 2=cup) rather than the original fundus image.
The model must learn structural relationships like cup-to-disc ratio, rim area,
and cup eccentricity directly from the segmentation geometry.

Segmentation source (--seg-mode):
  gt    [default]  Rasterise expert contour annotations from the PAPILA manifest.
  unet             Run a trained UNetSegmenter on the raw fundus image.
                   Requires --unet-weights.

Cross-validation:
  5-fold stratified group CV, with both eyes of the same patient always in the
  same fold (preventing OD/OS leakage).  Folds are built from the PAPILA
  clinical CSV, then matched to manifest entries by patient ID + eye.

Usage examples:
  # GT masks, default settings
  python -m v3.scripts.main.phase_seg_cnn.run_seg_cnn \\
      --image-dir   Papila/FundusImages \\
      --clinical-dir Papila/ClinicalData \\
      --manifest     manifest.csv

  # U-Net predicted masks, resnet50 backbone
  python -m v3.scripts.main.phase_seg_cnn.run_seg_cnn \\
      --image-dir   Papila/FundusImages \\
      --clinical-dir Papila/ClinicalData \\
      --manifest     manifest.csv \\
      --seg-mode unet --unet-weights models/unet_segmenter/best.pt \\
      --backbone resnet50

  # Single-channel label map instead of one-hot
  python -m v3.scripts.main.phase_seg_cnn.run_seg_cnn \\
      --image-dir   Papila/FundusImages \\
      --clinical-dir Papila/ClinicalData \\
      --manifest     manifest.csv \\
      --channels 1 --no-pretrained
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from v3.classes.seg_cnn import (
    SegCNN, SegMapDataset, SegMapRecord,
    UNetFineTuneDataset, precompute_unet_seg_maps,
)


# ---------------------------------------------------------------------------
# Clinical data loader (PAPILA)
# ---------------------------------------------------------------------------

def load_papila_labels(
    clinical_dir: Path,
    label_col: str = "Diagnosis",
    drop_suspects: bool = True,
) -> pd.DataFrame:
    """
    Return a DataFrame with columns:
      patient_id (int), eye (str), label (int)

    Reads patient_data_od.xlsx + patient_data_os.xlsx from clinical_dir.
    """
    od_path = clinical_dir / "patient_data_od.xlsx"
    os_path = clinical_dir / "patient_data_os.xlsx"
    frames = []
    for path, eye in ((od_path, "OD"), (os_path, "OS")):
        if not path.exists():
            raise FileNotFoundError(f"Clinical data not found: {path}")
        df = pd.read_excel(path, header=1)
        df["eye"] = eye
        id_col = "Patient ID" if "Patient ID" in df.columns else "ID"
        df["patient_id"] = (
            df[id_col].astype(str).str.extract(r"(\d+)")[0].astype(int)
        )
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    if drop_suspects:
        df = df[df[label_col] != 2].reset_index(drop=True)

    df["label"] = df[label_col].astype(int)
    return df[["patient_id", "eye", "label"]].copy()


# ---------------------------------------------------------------------------
# Build SegMapRecords from manifest + clinical labels
# ---------------------------------------------------------------------------

def build_records(
    manifest_path: Path,
    clinical_df: pd.DataFrame,
    dataset_filter: str = "papila",
) -> List[SegMapRecord]:
    """
    Join manifest entries (image paths + annotation paths) with clinical labels.

    Returns one SegMapRecord per matched eye sample.
    """
    manifest = pd.read_csv(manifest_path)
    papila_rows = manifest[manifest["dataset"] == dataset_filter].copy()

    # Parse patient_id and eye from sample_id  e.g. "papila_RET042OD" → 42, "OD"
    def _parse(sid: str):
        sid = sid.replace(f"{dataset_filter}_RET", "")
        eye = sid[-2:].upper()        # "OD" or "OS"
        pid = int(sid[:-2])
        return pid, eye

    papila_rows[["patient_id", "eye"]] = pd.DataFrame(
        papila_rows["sample_id"].apply(_parse).tolist(),
        index=papila_rows.index,
    )

    merged = papila_rows.merge(
        clinical_df[["patient_id", "eye", "label"]],
        on=["patient_id", "eye"],
        how="inner",
    )

    records: List[SegMapRecord] = []
    for _, row in merged.iterrows():
        records.append(
            SegMapRecord(
                sample_id=row["sample_id"],
                image_path=Path(row["image_path"]),
                annotation_disc=Path(row["annotation_disc"]),
                annotation_cup=Path(row["annotation_cup"]),
                annotation_type_disc=row["annotation_type_disc"],
                annotation_type_cup=row["annotation_type_cup"],
                patient_id=int(row["patient_id"]),
                eye=str(row["eye"]),
                label=int(row["label"]),
            )
        )
    return records


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / n if n else float("nan")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Returns (auc, acc, y_true, y_prob)."""
    model.eval()
    probs_list, labels_list = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        probs_list.append(prob)
        labels_list.append(y.numpy())
    y_true = np.concatenate(labels_list)
    y_prob = np.concatenate(probs_list)
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    acc = float(accuracy_score(y_true, (y_prob >= 0.5).astype(int)))
    return auc, acc, y_true, y_prob


# ---------------------------------------------------------------------------
# U-Net fine-tuning
# ---------------------------------------------------------------------------

def finetune_unet(
    segmenter,
    records: List[SegMapRecord],
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
) -> None:
    """Fine-tune the U-Net on a fold's training records using GT annotations."""
    import copy
    ds = UNetFineTuneDataset(
        records,
        target_size=segmenter.target_size,
        normalize=segmenter.normalize,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(segmenter.model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    segmenter.model.train()
    for epoch in tqdm(range(1, epochs + 1), desc="  U-Net finetune", unit="ep", leave=False):
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(segmenter.model(images), masks)
            loss.backward()
            optimizer.step()
    segmenter.model.eval()


# ---------------------------------------------------------------------------
# Cross-validation loop
# ---------------------------------------------------------------------------

def _train_one_split(
    train_recs, val_recs, y_train, args, device, out_dir, label,
    train_seg_maps=None, val_seg_maps=None,
):
    """Train one fold/split, return (auc, acc, y_true, y_prob, fpr, tpr)."""
    train_ds = SegMapDataset(
        train_recs,
        target_size=args.img_size,
        channels=args.channels,
        augment=True,
        seg_target_size=args.seg_size,
        crop_to_disc=not args.no_crop,
        precomputed_seg_maps=train_seg_maps,
    )
    val_ds = SegMapDataset(
        val_recs,
        target_size=args.img_size,
        channels=args.channels,
        augment=False,
        seg_target_size=args.seg_size,
        crop_to_disc=not args.no_crop,
        precomputed_seg_maps=val_seg_maps,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    model = SegCNN(
        num_classes=2,
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        in_channels=args.channels,
        dropout=args.dropout,
    ).to(device)

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    class_weights = (
        torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32).to(device)
        if n_pos > 0 and n_neg > 0 else None
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    best_auc, best_state = -1.0, None
    epoch_bar = tqdm(range(1, args.epochs + 1), desc=label, unit="ep")
    for _ in epoch_bar:
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_auc, val_acc, _, _ = evaluate(model, val_loader, device)
        scheduler.step()
        epoch_bar.set_postfix(loss=f"{train_loss:.4f}", auc=f"{val_auc:.3f}", acc=f"{val_acc:.3f}")
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save({"model": best_state, "auc": best_auc}, out_dir / "best.pt")

    if best_state is not None:
        model.load_state_dict(best_state)
    final_auc, final_acc, y_true, y_prob = evaluate(model, val_loader, device)
    fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=1)
    return final_auc, final_acc, y_true, y_prob, fpr, tpr


def run_cv(
    records: List[SegMapRecord],
    args,
    device: torch.device,
    unet_segmenter=None,
    out_dir: Path = Path("analysis_data/seg_cnn"),
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)

    y = np.array([r.label for r in records])
    groups = np.array([r.patient_id for r in records])
    splits = list(StratifiedGroupKFold(n_splits=args.n_splits).split(
        np.arange(len(records)), y, groups
    ))

    # If using U-Net without fine-tuning, precompute all seg maps once upfront.
    # If fine-tuning, we must precompute per fold (after fine-tuning) so we
    # save the base weights here to restore at the start of each fold.
    base_unet_state = None
    all_seg_maps = None
    if unet_segmenter is not None:
        if args.finetune_epochs > 0:
            import copy
            base_unet_state = copy.deepcopy(unet_segmenter.model.state_dict())
        else:
            print(f"Precomputing U-Net seg maps for {len(records)} samples (once for all folds)...")
            all_seg_maps = precompute_unet_seg_maps(records, unet_segmenter, args.unet_threshold)

    fold_metrics: List[Dict] = []
    roc_curves = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        start = time.time()
        print(f"\n{'='*60}")
        print(f"  Fold {fold_idx+1}/{args.n_splits}  "
              f"(train={len(train_idx)}, val={len(val_idx)})")
        print(f"{'='*60}")

        fold_dir = out_dir / f"fold{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_recs = [records[i] for i in train_idx]
        val_recs   = [records[i] for i in val_idx]

        if unet_segmenter is not None and args.finetune_epochs > 0:
            # Restore base REFUGE weights, then fine-tune on this fold's training data only
            unet_segmenter.model.load_state_dict(copy.deepcopy(base_unet_state))
            print(f"  Fine-tuning U-Net for {args.finetune_epochs} epochs on training fold...")
            finetune_unet(
                unet_segmenter, train_recs,
                epochs=args.finetune_epochs,
                lr=args.finetune_lr,
                batch_size=args.finetune_batch_size,
                device=device,
            )
            print(f"  Generating seg maps with fine-tuned U-Net...")
            train_maps = precompute_unet_seg_maps(train_recs, unet_segmenter, args.unet_threshold)
            val_maps   = precompute_unet_seg_maps(val_recs,   unet_segmenter, args.unet_threshold)
        else:
            train_maps = [all_seg_maps[i] for i in train_idx] if all_seg_maps else None
            val_maps   = [all_seg_maps[i] for i in val_idx]   if all_seg_maps else None

        auc, acc, y_true, y_prob, fpr, tpr = _train_one_split(
            train_recs, val_recs, y[train_idx], args, device,
            fold_dir, label=f"Fold {fold_idx+1}",
            train_seg_maps=train_maps, val_seg_maps=val_maps,
        )
        elapsed = time.time() - start
        roc_curves.append((fpr, tpr, auc))
        print(f"  Fold {fold_idx+1}  AUC={auc:.4f}  ACC={acc:.4f}  ({elapsed:.0f}s)")

        pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).to_csv(
            fold_dir / "val_probs.csv", index=False
        )
        fold_metrics.append({
            "fold": fold_idx, "auc": auc, "acc": acc,
            "n_train": len(train_idx), "n_val": len(val_idx), "elapsed_s": elapsed,
        })

    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.to_csv(out_dir / "fold_metrics.csv", index=False)

    mean_auc = float(metrics_df["auc"].mean())
    std_auc  = float(metrics_df["auc"].std())
    mean_acc = float(metrics_df["acc"].mean())
    std_acc  = float(metrics_df["acc"].std())

    print(f"\n{'='*60}")
    print(f"  CV Summary ({args.n_splits} folds)")
    print(f"  AUC = {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  ACC = {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"{'='*60}\n")

    pd.DataFrame([{
        "backbone": args.backbone, "seg_mode": args.seg_mode,
        "channels": args.channels, "pretrained": not args.no_pretrained,
        "epochs": args.epochs, "lr": args.lr, "dropout": args.dropout,
        "auc_mean": mean_auc, "auc_std": std_auc,
        "acc_mean": mean_acc, "acc_std": std_acc,
        "n_folds": args.n_splits,
    }]).to_csv(out_dir / "summary.csv", index=False)

    # Mean ROC curve
    mean_fpr = np.linspace(0, 1, 200)
    tprs = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr, _ in roc_curves]
    mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
    std_tpr  = np.std(tprs, axis=0)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(mean_fpr, mean_tpr, lw=2,
            label=f"AUC = {mean_auc:.3f} ± {std_auc:.3f}")
    ax.fill_between(mean_fpr,
                    np.maximum(mean_tpr - std_tpr, 0),
                    np.minimum(mean_tpr + std_tpr, 1),
                    alpha=0.2, color="steelblue")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Seg-map CNN ({args.backbone}, {args.seg_mode})")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout(); fig.savefig(out_dir / "roc_mean.png", dpi=170); plt.close(fig)

    return metrics_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a CNN on optic disc/cup segmentation maps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Data paths ---
    p.add_argument("--image-dir",    required=True,  help="PAPILA FundusImages directory")
    p.add_argument("--clinical-dir", required=True,  help="PAPILA ClinicalData directory")
    p.add_argument("--manifest",     required=True,  help="manifest.csv with annotation paths")
    p.add_argument("--output-dir",   default="analysis_data/seg_cnn",
                   help="Where to save results")

    # --- Segmentation mode ---
    p.add_argument("--seg-mode", choices=["gt", "unet"], default="gt",
                   help="gt = expert annotations; unet = predicted masks from trained U-Net")
    p.add_argument("--unet-weights",   default=None,
                   help="Path to trained UNet weights (.pt); required for --seg-mode unet")
    p.add_argument("--unet-normalize", default="per_image",
                   choices=["none", "per_image", "imagenet"],
                   help="Normalisation used when the UNet was trained")
    p.add_argument("--unet-threshold", type=float, default=0.5,
                   help="Sigmoid threshold for binary mask from U-Net logits")
    p.add_argument("--seg-size", type=int, default=512,
                   help="Spatial size at which GT contours are rasterised / U-Net runs")

    # --- Model ---
    p.add_argument("--backbone", default="resnet18",
                   choices=["resnet18", "resnet50", "efficientnet_b0"],
                   help="CNN backbone")
    p.add_argument("--channels", type=int, default=3, choices=[1, 3],
                   help="1 = single-channel normalised label map; "
                        "3 = one-hot [bg, disc_rim, cup]")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Do not load ImageNet weights for the backbone")
    p.add_argument("--dropout", type=float, default=0.3)

    # --- Training ---
    p.add_argument("--epochs",     type=int,   default=60)
    p.add_argument("--batch-size", type=int,   default=16)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--wd",         type=float, default=1e-4,
                   help="AdamW weight decay")
    p.add_argument("--img-size",   type=int,   default=224,
                   help="CNN input spatial resolution")
    p.add_argument("--no-crop", action="store_true",
                   help="Disable disc-region cropping (keeps full-image seg map)")
    p.add_argument("--workers",    type=int,   default=4,
                   help="DataLoader num_workers")

    # --- U-Net fine-tuning (only applies with --seg-mode unet) ---
    p.add_argument("--finetune-epochs",     type=int,   default=0,
                   help="Epochs to fine-tune U-Net on each fold's training data "
                        "(0 = disabled, uses base REFUGE weights as-is)")
    p.add_argument("--finetune-lr",         type=float, default=1e-5,
                   help="Learning rate for U-Net fine-tuning")
    p.add_argument("--finetune-batch-size", type=int,   default=4,
                   help="Batch size for U-Net fine-tuning")

    # --- CV ---
    p.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")
    p.add_argument("--seed",     type=int, default=42)

    # --- Misc ---
    p.add_argument("--device", default=None,
                   help="torch device string (default: cuda if available)")
    p.add_argument("--label-col", default="Diagnosis",
                   help="Label column in PAPILA clinical xlsx")

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # ---- Load PAPILA labels ----
    clinical_df = load_papila_labels(
        Path(args.clinical_dir),
        label_col=args.label_col,
        drop_suspects=True,
    )
    print(f"Clinical labels loaded: {len(clinical_df)} eye records "
          f"(N={int((clinical_df.label==0).sum())}, G={int((clinical_df.label==1).sum())})")

    # ---- Build records ----
    records = build_records(Path(args.manifest), clinical_df)
    print(f"Matched records: {len(records)} "
          f"(N={sum(r.label==0 for r in records)}, G={sum(r.label==1 for r in records)})")

    if not records:
        print("ERROR: No records matched.  Check manifest and clinical data paths.")
        sys.exit(1)

    # ---- Load UNet if needed ----
    unet_segmenter = None
    if args.seg_mode == "unet":
        if not args.unet_weights:
            print("ERROR: --seg-mode unet requires --unet-weights")
            sys.exit(1)
        from v3.classes.unet_segmenter import UNetSegmenter  # noqa
        import tempfile, csv, os

        # Build a minimal manifest for UNetSegmenter init
        fd, tmp = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "sample_id", "dataset", "image_path",
                "annotation_disc", "annotation_cup",
                "annotation_type_disc", "annotation_type_cup", "split",
            ])
            r = records[0]
            writer.writerow([
                r.sample_id, "papila", str(r.image_path),
                str(r.annotation_disc), str(r.annotation_cup),
                r.annotation_type_disc, r.annotation_type_cup, "train",
            ])
        unet_segmenter = UNetSegmenter(
            manifest_path=Path(tmp),
            normalize=args.unet_normalize,
            target_size=args.seg_size,
        )
        os.unlink(tmp)
        state = torch.load(args.unet_weights, map_location=unet_segmenter.device)
        state_dict = state.get("model", state)
        unet_segmenter.model.load_state_dict(state_dict)
        unet_segmenter.model.to(unet_segmenter.device)
        unet_segmenter.model.eval()
        print(f"U-Net weights loaded from {args.unet_weights}")

    # ---- Run CV ----
    out_dir = Path(args.output_dir)
    run_cv(
        records=records,
        args=args,
        device=device,
        unet_segmenter=unet_segmenter,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
