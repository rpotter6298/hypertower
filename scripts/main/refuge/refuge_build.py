"""REFUGE training/evaluation helper.

Usage examples (after activating .venv_refuge):

  python refuge_build.py --train-clf
  python refuge_build.py --eval --with-ttt

The script expects the REFUGE folder and writes checkpoints under
models/refuge/segmentation and models/refuge/classifier.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
import shutil
import sys

import torch
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torch import nn
from torchvision import models

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classes.refuge_preprocessing import RefugePreprocessing, RefugeSample
from classes.refuge_segmentation import RefugeSegmentation
from classes.refuge_classification import (
    RefugeClassification,
    RefugeClassificationRecord,
    RefugeClassificationDataset,
    _default_image_transform,
    _geometry_from_mask,
    UNetGeometryProvider,
)
from classes.unet_segmenter import UNetSegmenter
from classes.papila_builders import build_papila_clinical

REFUGE_ROOT = Path("REFUGE")
SEG_CKPT = Path("models/refuge/segmentation/refuge_segmentation_best.pt")
CLF_DIR = Path("models/refuge/classifier")
UNET_WEIGHT_CANDIDATES = (
    Path("models/v2/refuge/segmentation/per_image/best.pt"),
    Path("models/v2/refuge/segmentation/best.pt"),
    Path("models/unet_segmenter/best.pt"),
)

CLASSIFIER_BACKBONES = {
    "resnet50": models.ResNet50_Weights.DEFAULT,
    "densenet121": models.DenseNet121_Weights.DEFAULT,
    "efficientnet_b0": models.EfficientNet_B0_Weights.DEFAULT,
    "efficientnet_b7": models.EfficientNet_B7_Weights.DEFAULT,
}


def build_classifier_backbone(name: str) -> nn.Module:
    name = name.lower()
    if name not in CLASSIFIER_BACKBONES:
        raise ValueError(f"Unsupported classifier backbone '{name}'")

    weights = CLASSIFIER_BACKBONES[name]

    if name == "resnet50":
        model = models.resnet50(weights=weights)
        feat_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif name == "densenet121":
        model = models.densenet121(weights=weights)
        feat_dim = model.classifier.in_features
        model.classifier = nn.Identity()
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        feat_dim = model.classifier[-1].in_features  # type: ignore[index]
        model.classifier = nn.Identity()
    elif name == "efficientnet_b7":
        model = models.efficientnet_b7(weights=weights)
        feat_dim = model.classifier[-1].in_features  # type: ignore[index]
        model.classifier = nn.Identity()
    else:  # pragma: no cover
        raise ValueError(f"Unsupported classifier backbone '{name}'")

    setattr(model, "_feature_dim", int(feat_dim))
    return model


def classifier_checkpoint_dir(backbone_name: str) -> Path:
    return CLF_DIR / backbone_name


def classifier_checkpoint_path(backbone_name: str) -> Path:
    return classifier_checkpoint_dir(backbone_name) / "refuge_classifier_best.pt"


def resolve_unet_weights(explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return explicit
    for cand in UNET_WEIGHT_CANDIDATES:
        if cand.exists():
            return cand
    return UNET_WEIGHT_CANDIDATES[0]


def ensure_preprocessing() -> RefugePreprocessing:
    if not REFUGE_ROOT.exists():
        raise FileNotFoundError(f"REFUGE directory not found at {REFUGE_ROOT}")
    return RefugePreprocessing(REFUGE_ROOT)


def load_allowed_ids(
    csv_path: Optional[Path], dice_threshold: float
) -> Optional[Set[str]]:
    if csv_path is None or not csv_path.exists():
        return None
    allowed: Set[str] = set()
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sample_id = row.get("sample_id")
            if not sample_id or sample_id == "__mean__":
                continue
            try:
                disc = float(row.get("dice_disc", "nan"))
                cup = float(row.get("dice_cup", "nan"))
            except (TypeError, ValueError):
                continue
            if disc < dice_threshold and cup < dice_threshold:
                continue
            allowed.add(sample_id)
    return allowed


def build_papila_samples(
    image_dir: Path,
    clinical_dir: Path,
    label_col: str,
    positive_labels: Sequence[str],
    allowed_ids: Optional[Set[str]],
) -> List[RefugeSample]:
    clinical = build_papila_clinical(
        image_dir=str(image_dir),
        clinical_dir=str(clinical_dir),
        label_col=label_col,
        cat_cols=[],
    )
    positives = {lbl.lower() for lbl in positive_labels}
    samples: Dict[str, RefugeSample] = {}
    for _, row in clinical.df.iterrows():
        image_path = clinical.get_image_path(row)
        sample_id = f"papila_{Path(image_path).stem}"
        if allowed_ids is not None and sample_id not in allowed_ids:
            continue
        if sample_id in samples:
            continue
        value = row.get(label_col)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        try:
            label_int = int(value)
            if label_int == 2:
                continue
            label = 1 if label_int > 0 else 0
        except (TypeError, ValueError):
            label = 1 if str(value).strip().lower() in positives else 0
        samples[sample_id] = RefugeSample(
            sample_id=sample_id,
            dataset="papila",
            split="holdout",
            image_path=Path(image_path),
            label=label,
            device=None,
            mask_path=None,
            fovea_coord=None,
        )
    return list(samples.values())


def load_contour(path: Path) -> np.ndarray:
    coords = np.loadtxt(path)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 2)
    return coords


def contour_to_mask(coords: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    if coords is None or coords.size == 0:
        return np.zeros((size[1], size[0]), dtype=np.uint8)
    img = Image.new("L", size, 0)
    draw = ImageDraw.Draw(img)
    points = [tuple(map(float, pt)) for pt in coords]
    draw.polygon(points, outline=1, fill=1)
    return np.array(img, dtype=np.uint8)


class PapilaGTGeometryProvider:
    def __init__(self, contours_dir: Path) -> None:
        self.contours_dir = contours_dir

    def _pick(self, base: str, kind: str) -> Optional[Path]:
        for exp in ("exp2", "exp1"):
            cand = self.contours_dir / f"{base}_{kind}_{exp}.txt"
            if cand.exists():
                return cand
        return None

    def __call__(self, sample: RefugeSample, scale: float):
        base = Path(sample.image_path).stem
        disc_path = self._pick(base, "disc")
        cup_path = self._pick(base, "cup")
        if disc_path is None or cup_path is None:
            raise RuntimeError(f"Missing ground-truth contours for {sample.sample_id}")

        image = Image.open(sample.image_path).convert("RGB")
        disc_coords = load_contour(disc_path)
        cup_coords = load_contour(cup_path)
        disc_mask = contour_to_mask(disc_coords, image.size)
        cup_mask = contour_to_mask(cup_coords, image.size)
        cup_mask = ((cup_mask > 0) & (disc_mask > 0)).astype(np.uint8)
        geom = _geometry_from_mask(disc_mask, scale)
        return geom, disc_mask.astype(np.uint8), cup_mask.astype(np.uint8)


def build_papila_records(
    args: argparse.Namespace,
    pre: RefugePreprocessing,
    checkpoint_path: Path,
) -> Tuple[List[RefugeClassificationRecord], Optional[RefugeClassification]]:
    allowed = load_allowed_ids(
        getattr(args, "papila_metrics", None),
        getattr(args, "papila_dice_threshold", 0.01),
    )
    samples = build_papila_samples(
        args.papila_image_dir,
        args.papila_clinical_dir,
        args.papila_label_col,
        args.papila_positive_labels,
        allowed,
    )
    if not samples:
        return [], None

    cache_dir = args.clf_cache_dir
    if cache_dir is not None and getattr(args, "papila_use_gt", False):
        cache_dir = cache_dir / "gt"

    if getattr(args, "papila_use_gt", False):
        geometry_fn = PapilaGTGeometryProvider(args.papila_contours_dir)
        provider = geometry_fn
    else:
        seg_manifest = getattr(args, "seg_manifest", None)
        seg_weights = resolve_unet_weights(getattr(args, "seg_weights", None))
        if seg_manifest is None or seg_weights is None:
            raise SystemExit(
                "Papila evaluation without GT masks requires --seg-manifest and --seg-weights"
            )
        segmenter = UNetSegmenter(
            manifest_path=seg_manifest,
            device=args.device,
            normalize=args.seg_normalize,
        )
        seg_state = torch.load(seg_weights, map_location=args.device)
        seg_state_dict = seg_state.get("model", seg_state)
        segmenter.model.load_state_dict(seg_state_dict)
        segmenter.model.to(args.device)
        provider = UNetGeometryProvider(
            segmenter=segmenter,
            threshold=args.segmenter_threshold,
            tta=args.segmenter_tta,
        )
        geometry_fn = provider

    papila_seg = RefugeSegmentation(pre)
    backbone = build_classifier_backbone(args.clf_backbone)
    papila_clf = RefugeClassification(
        pre,
        papila_seg,
        backbone=backbone,
        geometry_fn=provider,
        cache_dir=cache_dir,
    )
    papila_clf.crop_scale = args.crop_scale
    papila_clf.crop_size = args.crop_size
    papila_clf.eval_transform = _default_image_transform(args.crop_size)
    papila_clf.ttt_transform = papila_clf.eval_transform
    papila_state = torch.load(checkpoint_path, map_location=args.device)
    papila_clf.backbone.load_state_dict(papila_state["backbone"])
    papila_clf.classifier_head.load_state_dict(papila_state["classifier"])
    papila_clf.rotation_head.load_state_dict(papila_state["rotation"])
    papila_clf.backbone.to(args.device)
    papila_clf.classifier_head.to(args.device)
    papila_clf.rotation_head.to(args.device)
    if getattr(args, "clear_clf_cache", False):
        papila_clf.clear_disk_cache()

    records = papila_clf.build_records_for_samples(
        samples, crop_scale=args.crop_scale, progress_prefix="papila"
    )
    print(f"[eval] Prepared {len(records)} PAPILA records")
    return records, papila_clf


def train_unet_segmenter(args: argparse.Namespace) -> None:
    manifest_path = args.seg_manifest or Path("manifest.csv")
    mask_cache_dir = None if args.in_memory_cache else args.mask_cache_dir
    image_cache_dir = None if args.in_memory_cache else args.image_cache_dir
    if args.in_memory_cache and (args.mask_cache_dir or args.image_cache_dir):
        print("[unet-seg] in_memory_cache enabled: disk caches disabled for this run.")

    segmenter = UNetSegmenter(
        manifest_path=manifest_path,
        device=args.device,
        target_size=args.seg_image_size,
        normalize=args.seg_normalize,
        use_stronger_aug=args.seg_strong_aug,
        train_datasets=args.seg_train_datasets,
        val_datasets=args.seg_val_datasets,
        holdout_datasets=args.seg_holdout_datasets,
        mask_cache_dir=mask_cache_dir,
        image_cache_dir=image_cache_dir,
        in_memory_cache=args.in_memory_cache,
        loader_workers=args.loader_workers,
    )
    if mask_cache_dir:
        print(f"[unet-seg] mask_cache_dir={mask_cache_dir}")
    if image_cache_dir:
        print(f"[unet-seg] image_cache_dir={image_cache_dir}")
    if args.in_memory_cache:
        print("[unet-seg] prebuilding in-memory cache")
        segmenter.prebuild_in_memory_cache(
            cache_workers=max(0, int(args.cache_workers)),
            include_train=True,
            include_val=True,
            include_holdout=False,
        )

    segmenter.train(
        epochs=args.seg_epochs,
        batch_size=args.seg_batch_size,
        lr=args.seg_lr,
        weight_decay=args.seg_weight_decay,
        checkpoint_dir=args.seg_checkpoint_dir,
    )
    print(
        "[unet-seg] Training complete. Best checkpoint stored at",
        (args.seg_checkpoint_dir / "best.pt").resolve(),
    )


def _load_segmentation(
    pre: RefugePreprocessing, args: argparse.Namespace
) -> RefugeSegmentation:
    seg = RefugeSegmentation(pre)
    seg.build_datasets(
        image_size=args.seg_image_size,
        batch_size=args.seg_batch_size,
        num_workers=args.num_workers,
    )
    if not SEG_CKPT.exists():
        raise FileNotFoundError(f"Segmentation checkpoint missing: {SEG_CKPT}")
    state = torch.load(SEG_CKPT, map_location=args.device)
    seg.model.load_state_dict(state)
    seg.model.to(args.device)
    return seg


def train_classifier(args: argparse.Namespace) -> None:
    pre = ensure_preprocessing()
    seg = _load_segmentation(pre, args)
    backbone = build_classifier_backbone(args.clf_backbone)

    print(f"[classifier] Using backbone: {args.clf_backbone}")

    clf = RefugeClassification(
        pre,
        seg,
        backbone=backbone,
        cache_dir=args.clf_cache_dir,
        use_all_labeled=args.clf_use_all,
        auto_val_ratio=args.clf_auto_val_ratio,
    )
    if args.clear_clf_cache:
        clf.clear_disk_cache()
    clf.build_datasets(
        crop_scale=args.crop_scale,
        crop_size=args.crop_size,
        batch_size=args.clf_batch_size,
        num_workers=args.num_workers,
    )

    default_ckpt_path = classifier_checkpoint_path(args.clf_backbone)
    ckpt_path = args.clf_checkpoint_path or default_ckpt_path
    ckpt_dir = ckpt_path.parent
    history = clf.train(
        epochs=args.clf_epochs,
        lr=args.clf_lr,
        weight_decay=args.clf_weight_decay,
        rotation_weight=args.rotation_weight,
        checkpoint_dir=ckpt_dir,
        device=args.device,
    )
    print("Classifier training complete. Best AUC:", history.get("best_auc"))
    print(f"Checkpoint directory: {ckpt_dir}")
    saved_path = ckpt_dir / "refuge_classifier_best.pt"
    if ckpt_path != saved_path:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(saved_path, ckpt_path)
        print(f"Checkpoint copied to: {ckpt_path}")


def _load_classifier(
    pre: RefugePreprocessing, seg: RefugeSegmentation, args: argparse.Namespace
) -> Tuple[RefugeClassification, Path]:
    backbone = build_classifier_backbone(args.clf_backbone)
    clf = RefugeClassification(
        pre,
        seg,
        backbone=backbone,
        cache_dir=args.clf_cache_dir,
        use_all_labeled=args.clf_use_all,
        auto_val_ratio=args.clf_auto_val_ratio,
    )
    clf.build_datasets(
        crop_scale=args.crop_scale,
        crop_size=args.crop_size,
        batch_size=args.clf_batch_size,
        num_workers=args.num_workers,
    )
    ckpt_path = args.clf_checkpoint_path or classifier_checkpoint_path(
        args.clf_backbone
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Classifier checkpoint missing: {ckpt_path}")
    print(f"[classifier] Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=args.device)
    clf.backbone.load_state_dict(state["backbone"])
    clf.classifier_head.load_state_dict(state["classifier"])
    clf.rotation_head.load_state_dict(state["rotation"])
    clf.backbone.to(args.device)
    clf.classifier_head.to(args.device)
    clf.rotation_head.to(args.device)
    return clf, ckpt_path


def _collect_records(
    pre: RefugePreprocessing,
    seg: RefugeSegmentation,
    clf: RefugeClassification,
    dataset_name: str,
    split: str,
    scale: float,
) -> List[RefugeClassificationRecord]:
    manifest = pre.build_manifest()
    samples = [
        sample
        for sample in manifest
        if sample.dataset == dataset_name
        and sample.split == split
        and sample.label is not None
    ]
    if not samples:
        return []
    print(f"[eval] Preparing {len(samples)} samples for {dataset_name.upper()} {split}")
    return clf.build_records_for_samples(
        samples, crop_scale=scale, progress_prefix=f"{dataset_name}_{split}"
    )


def _auc_for_records(
    clf: RefugeClassification,
    records: List[RefugeClassificationRecord],
    device: str,
) -> float:
    if not records:
        return float("nan")
    dataset = RefugeClassificationDataset(
        records,
        transform=clf.eval_transform,
        polar_transform=clf.polar_transform,
        size=clf.crop_size,
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    clf.backbone.to(device).eval()
    clf.classifier_head.to(device).eval()
    preds: List[float] = []
    targets: List[int] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False, unit="batch"):
            images = batch["image"].to(device)
            polars = batch["polar"].to(device)
            extra_feats = batch["features"].to(device)
            labels = batch["label"].cpu().numpy().tolist()
            feats_img = clf.backbone(images)
            feats = feats_img
            if getattr(clf, "use_polar", False):
                feats_polar = clf.backbone(polars)
                feats = torch.cat([feats, feats_polar], dim=1)
            if getattr(clf, "extra_feature_dim", 0) > 0:
                feats = torch.cat([feats, extra_feats], dim=1)
            logits = clf.classifier_head(feats)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist()
            preds.extend(probs)
            targets.extend(labels)
    if len(set(targets)) < 2:
        return float("nan")
    return float(roc_auc_score(targets, preds))


def evaluate(args: argparse.Namespace) -> None:
    pre = ensure_preprocessing()
    seg = _load_segmentation(pre, args)
    clf, clf_ckpt = _load_classifier(pre, seg, args)
    if args.clear_clf_cache:
        clf.clear_disk_cache()

    def evaluate_subset(
        clf_obj: RefugeClassification,
        records: List[RefugeClassificationRecord],
        label: str,
    ) -> None:
        if not records:
            print(f"[eval] No samples found for {label}; skipping.")
            return

        base_state = {
            "backbone": clf_obj.backbone.state_dict(),
            "rotation": clf_obj.rotation_head.state_dict(),
        }

        auc_no_ttt = _auc_for_records(clf_obj, records, device=args.device)

        auc_ttt = float("nan")
        if args.with_ttt:
            ttt_loader = DataLoader(
                RefugeClassificationDataset(
                    records,
                    transform=clf_obj.ttt_transform,
                    polar_transform=clf_obj.polar_transform,
                    size=clf_obj.crop_size,
                ),
                batch_size=16,
                shuffle=False,
                num_workers=0,
            )
            ttt_iter = tqdm(range(args.ttt_steps), desc="TTT", unit="step")
            for _ in ttt_iter:
                clf_obj.apply_ttt(ttt_loader, device=args.device, steps=1)
            auc_ttt = _auc_for_records(clf_obj, records, device=args.device)
            clf_obj.backbone.load_state_dict(base_state["backbone"])
            clf_obj.rotation_head.load_state_dict(base_state["rotation"])

        print(
            f"{label}: AUC (no TTT) = {auc_no_ttt:.4f}"
            + (f", AUC (TTT) = {auc_ttt:.4f}" if args.with_ttt else "")
        )

    if args.eval_datasets:
        for dataset_name in dict.fromkeys(args.eval_datasets):
            if dataset_name.lower() == "papila":
                papila_records, papila_clf = build_papila_records(args, pre, clf_ckpt)
                if papila_clf is None:
                    print("[eval] Papila evaluation aborted; no samples built.")
                else:
                    evaluate_subset(papila_clf, papila_records, "PAPILA holdout")
            else:
                records = _collect_records(
                    pre,
                    seg,
                    clf,
                    dataset_name,
                    "holdout",
                    scale=args.crop_scale,
                )
                evaluate_subset(clf, records, f"{dataset_name.upper()} holdout")
        return

    # Do not mix splits: report per dataset + split
    subsets = [
        ("refuge1", "val"),
        ("refuge2", "val"),
        ("refuge2", "test"),
    ]

    for dataset_name, split in subsets:
        records = _collect_records(
            pre, seg, clf, dataset_name, split, scale=args.crop_scale
        )
        evaluate_subset(clf, records, f"{dataset_name.upper()} {split}")

        if args.dump_masks and dataset_name == "refuge1" and split == "val":
            out_dir = Path(args.dump_masks)
            out_dir.mkdir(parents=True, exist_ok=True)
            for rec in records:
                sample = rec.sample
                if sample is None:
                    continue
                pred = seg.predict_mask(sample, device=args.device).numpy()
                Image.fromarray((pred * 255).astype(np.uint8)).save(
                    out_dir / f"{sample.sample_id}_pred.png"
                )
                if sample.mask_path and sample.mask_path.exists():
                    Image.open(sample.mask_path).convert("L").save(
                        out_dir / f"{sample.sample_id}_gt.png"
                    )


def evaluate_segmentation(args: argparse.Namespace) -> None:
    manifest_path = args.seg_manifest or Path("manifest.csv")
    mask_cache_dir = None if args.in_memory_cache else args.mask_cache_dir
    image_cache_dir = None if args.in_memory_cache else args.image_cache_dir

    segmenter = UNetSegmenter(
        manifest_path=manifest_path,
        normalize=args.seg_normalize,
        device=args.device,
        mask_cache_dir=mask_cache_dir,
        image_cache_dir=image_cache_dir,
        in_memory_cache=args.in_memory_cache,
        loader_workers=args.loader_workers,
    )
    if args.seg_weights is None:
        raise SystemExit(
            "--seg-weights must be specified for --eval-seg; "
            "e.g. --seg-weights models/v2/refuge/segmentation/per_image_refuge_build/best.pt"
        )
    ckpt = args.seg_weights
    if not ckpt.exists():
        raise FileNotFoundError(f"Segmentation weights not found at {ckpt}")
    state = torch.load(ckpt, map_location=segmenter.device)
    state_dict = state.get("model", state)
    segmenter.model.load_state_dict(state_dict, strict=False)
    print(f"[seg-eval] Loaded weights from {ckpt}")

    if args.in_memory_cache:
        segmenter.prebuild_in_memory_cache(
            cache_workers=max(0, int(args.cache_workers)),
            include_train=False,
            include_val="val" in args.eval_seg_splits,
            include_holdout="holdout" in args.eval_seg_splits,
        )

    dataset_filter = args.eval_seg_datasets
    split_filter = args.eval_seg_splits
    output_dir = args.eval_seg_output or Path("analysis_data/segmenter_eval")
    metrics_path = args.eval_seg_metrics_path

    segmenter.evaluate_dataset(
        dataset_filter=dataset_filter,
        split_filter=split_filter,
        output_dir=output_dir,
        save_overlays=not args.eval_seg_no_overlays,
        metrics_path=metrics_path,
        threshold=args.eval_seg_threshold,
        tta=args.eval_seg_tta,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REFUGE pipeline helper")
    parser.add_argument(
        "--train-unet-seg",
        action="store_true",
        help="Train the UNet segmenter (replacement for scripts/run_unet_segmenter.py)",
    )
    parser.add_argument(
        "--train-clf", action="store_true", help="Train the classification model"
    )
    parser.add_argument(
        "--eval", action="store_true", help="Run evaluation on stored checkpoints"
    )
    parser.add_argument(
        "--with-ttt",
        action="store_true",
        help="Apply test-time training during evaluation",
    )
    parser.add_argument(
        "--ttt-steps", type=int, default=1, help="TTT epochs over evaluation loader"
    )
    parser.add_argument(
        "--export-backbone",
        type=Path,
        default=None,
        help="Optional path to export the trained backbone weights",
    )
    parser.add_argument(
        "--dump-masks",
        type=Path,
        default=None,
        help="Optional directory to dump predicted/GT masks during eval",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num-workers", type=int, default=4)
    # Segmentation hyperparameters
    parser.add_argument("--seg-epochs", type=int, default=40)
    parser.add_argument("--seg-lr", type=float, default=1e-3)
    parser.add_argument("--seg-weight-decay", type=float, default=1e-5)
    parser.add_argument("--seg-image-size", type=int, default=512)
    parser.add_argument("--seg-batch-size", type=int, default=4)
    parser.add_argument(
        "--seg-manifest",
        type=Path,
        default=Path("manifest.csv"),
        help="Manifest CSV for the UNet segmenter (default: manifest.csv)",
    )
    parser.add_argument(
        "--seg-weights",
        type=Path,
        default=None,
        help="Path to UNet segmenter weights (default: models/unet_segmenter/best.pt)",
    )
    parser.add_argument(
        "--seg-normalize",
        choices=["none", "imagenet", "per_image"],
        default="none",
        help="Normalization mode used when running the UNet segmenter",
    )
    parser.add_argument(
        "--seg-strong-aug",
        action="store_true",
        help="Enable stronger geometric augmentations when training the UNet segmenter",
    )
    parser.add_argument(
        "--seg-train-datasets",
        nargs="+",
        default=["refuge"],
        help="Datasets to use for UNet segmenter training (default: refuge)",
    )
    parser.add_argument(
        "--seg-val-datasets",
        nargs="+",
        default=["refuge"],
        help="Datasets eligible for validation sampling (default: refuge)",
    )
    parser.add_argument(
        "--seg-holdout-datasets",
        nargs="+",
        default=["refuge"],
        help="Datasets reserved for holdout set during UNet segmenter training (default: refuge)",
    )
    parser.add_argument(
        "--seg-checkpoint-dir",
        type=Path,
        default=Path("models/v2/refuge/segmentation/per_image"),
        help="Directory to store UNet segmenter checkpoints",
    )
    parser.add_argument(
        "--loader-workers",
        type=int,
        default=0,
        help="DataLoader workers for UNet segmenter train/eval.",
    )
    parser.add_argument(
        "--mask-cache-dir",
        type=Path,
        default=None,
        help="Optional cache dir for parsed/resized disc+cup masks.",
    )
    parser.add_argument(
        "--image-cache-dir",
        type=Path,
        default=None,
        help="Optional cache dir for resized RGB images before augmentation.",
    )
    parser.add_argument(
        "--in-memory-cache",
        action="store_true",
        help="Cache preprocessed images and masks in RAM (per DataLoader worker process).",
    )
    parser.add_argument(
        "--cache-workers",
        type=int,
        default=0,
        help="Worker threads for prebuilding in-memory cache before training/eval.",
    )
    # Classification hyperparameters
    parser.add_argument("--clf-epochs", type=int, default=30)
    parser.add_argument("--clf-lr", type=float, default=1e-4)
    parser.add_argument("--clf-weight-decay", type=float, default=1e-4)
    parser.add_argument("--clf-batch-size", type=int, default=16)
    parser.add_argument(
        "--clf-backbone",
        choices=sorted(CLASSIFIER_BACKBONES.keys()),
        default="resnet50",
        help="Backbone architecture for the REFUGE classifier",
    )
    parser.add_argument("--rotation-weight", type=float, default=0.5)
    parser.add_argument("--crop-scale", type=float, default=2.5)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument(
        "--clf-cache-dir",
        type=Path,
        default=Path("cache_data/classifier_cache"),
        help="Directory to cache classifier preprocessing artifacts",
    )
    parser.add_argument(
        "--clear-clf-cache",
        action="store_true",
        help="Delete all cached geometry/mask files before running (use when segmenter weights have changed)",
    )
    parser.add_argument(
        "--clf-use-all",
        action="store_true",
        help="Use all labelled samples (train+val) when building classifier dataset",
    )
    parser.add_argument(
        "--clf-auto-val-ratio",
        type=float,
        default=0.1,
        help="Fraction for automatic validation split when no explicit val set is used",
    )
    parser.add_argument(
        "--clf-checkpoint-path",
        type=Path,
        default=None,
        help="Optional explicit path for the classifier checkpoint (defaults to models/refuge/classifier/<backbone>/refuge_classifier_best.pt)",
    )
    parser.add_argument(
        "--eval-datasets",
        nargs="+",
        help="Datasets to evaluate during --eval (e.g. papila). Defaults to REFUGE splits.",
    )
    # Segmentation evaluation parameters
    parser.add_argument(
        "--eval-seg",
        action="store_true",
        help="Evaluate the segmentation model on specified datasets/splits",
    )
    parser.add_argument(
        "--eval-seg-datasets",
        nargs="+",
        default=["refuge"],
        help="Segmentation datasets to evaluate (default: refuge)",
    )
    parser.add_argument(
        "--eval-seg-splits",
        nargs="+",
        choices=["train", "val", "holdout"],
        default=["holdout"],
        help="Segmentation splits to evaluate (default: holdout)",
    )
    parser.add_argument(
        "--eval-seg-output",
        type=Path,
        default=Path("analysis_data/segmenter_eval"),
        help="Directory to store segmentation metrics CSVs",
    )
    parser.add_argument(
        "--eval-seg-threshold",
        type=float,
        default=0.5,
        help="Threshold for binarising predicted masks during segmentation eval",
    )
    parser.add_argument(
        "--eval-seg-metrics-path",
        type=Path,
        default=None,
        help="Optional explicit CSV path for segmentation metrics output",
    )
    parser.add_argument(
        "--eval-seg-no-overlays",
        action="store_true",
        help="Skip saving GT/pred overlay images during segmentation evaluation",
    )
    parser.add_argument(
        "--eval-seg-tta",
        action="store_true",
        help="Enable horizontal/vertical flip TTA during segmentation evaluation",
    )
    parser.add_argument(
        "--papila-metrics",
        type=Path,
        default=None,
        help="Optional CSV of Papila Dice metrics used to filter samples",
    )
    parser.add_argument(
        "--papila-dice-threshold",
        type=float,
        default=0.01,
        help="Minimum Dice required (disc or cup) when filtering Papila metrics",
    )
    parser.add_argument(
        "--papila-positive-labels",
        nargs="+",
        default=["glaucoma", "glaucoma suspect", "suspect"],
        help="Papila label values treated as positive when labels are non-numeric",
    )
    parser.add_argument(
        "--papila-image-dir",
        type=Path,
        default=Path("Papila/FundusImages"),
        help="Path to Papila fundus images",
    )
    parser.add_argument(
        "--papila-clinical-dir",
        type=Path,
        default=Path("Papila/ClinicalData"),
        help="Path to Papila clinical CSVs",
    )
    parser.add_argument(
        "--papila-label-col",
        type=str,
        default="Diagnosis",
        help="Column name containing Papila labels",
    )
    parser.add_argument(
        "--papila-use-gt",
        action="store_true",
        help="Use Papila ground-truth contours when evaluating classifiers",
    )
    parser.add_argument(
        "--papila-contours-dir",
        type=Path,
        default=Path("Papila/ExpertsSegmentations/Contours"),
        help="Directory containing Papila contour text files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not any(
        [
            args.train_unet_seg,
            args.train_clf,
            args.eval,
            args.eval_seg,
            args.export_backbone,
        ]
    ):
        raise SystemExit(
            "Specify at least one action: --train-unet-seg, --train-clf, --eval, --eval-seg, or --export-backbone"
        )

    if args.train_unet_seg:
        train_unet_segmenter(args)

    if args.train_clf:
        train_classifier(args)

    if args.eval:
        evaluate(args)

    if args.eval_seg:
        evaluate_segmentation(args)

    if args.export_backbone:
        pre = ensure_preprocessing()
        seg = _load_segmentation(pre, args)
        clf = _load_classifier(pre, seg, args)
        out_path = args.export_backbone
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(clf.extract_backbone().state_dict(), out_path)
        print(f"Backbone weights exported to {out_path}")


if __name__ == "__main__":
    main()
