"""Evaluate REFUGE-trained classifier on Papila images using UNet crops."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image, ImageDraw

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classes.refuge_preprocessing import RefugePreprocessing, RefugeSample
from classes.refuge_segmentation import RefugeSegmentation
from classes.refuge_classification import (
    RefugeClassification,
    RefugeClassificationDataset,
    RefugeClassificationRecord,
    UNetGeometryProvider,
    _default_image_transform,
    _geometry_from_mask,
)
from classes.backbones import BACKBONES, load_backbone_weights
from classes.unet_segmenter import UNetSegmenter
from classes.papila_builders import build_papila_clinical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate classifier on Papila with UNet crops")
    parser.add_argument("--filtered-metrics", type=Path, required=True, help="CSV of Papila samples with acceptable Dice")
    parser.add_argument("--segmenter-manifest", type=Path, required=True, help="Manifest used to train the UNet segmenter")
    parser.add_argument("--segmenter-weights", type=Path, required=True, help="Path to trained UNet weights (best.pt)")
    parser.add_argument("--classifier-weights", type=Path, required=False, help="Path to classifier checkpoint (refuge_classifier_best.pt)")
    parser.add_argument("--refuge-root", type=Path, default=Path("REFUGE"))
    parser.add_argument("--image-dir", type=Path, default=Path("Papila/FundusImages"))
    parser.add_argument("--clinical-dir", type=Path, default=Path("Papila/ClinicalData"))
    parser.add_argument("--label-col", type=str, default="Diagnosis", help="Column name holding Papila labels")
    parser.add_argument(
        "--positive-labels",
        nargs="*",
        default=["glaucoma", "glaucoma suspect", "suspect"],
        help="Values treated as glaucoma-positive when labels are non-numeric",
    )
    parser.add_argument("--dice-threshold", type=float, default=0.01, help="Minimum Dice (disc or cup) to keep a sample")
    parser.add_argument("--segmenter-threshold", type=float, default=0.5, help="Probability threshold for UNet geometry")
    parser.add_argument("--segmenter-normalize", choices=["none", "imagenet", "per_image"], default="per_image")
    parser.add_argument("--segmenter-tta", action="store_true", help="Enable TTA (H/V flips) when deriving geometry")
    parser.add_argument("--crop-scale", type=float, default=2.5)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV to store per-sample probabilities")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("analysis_data/classifier_cache"),
        help="Directory to reuse classifier preprocessing cache",
    )
    parser.add_argument(
        "--use-gt-masks",
        action="store_true",
        help="Use ground truth Papila contours instead of UNet predictions",
    )
    parser.add_argument(
        "--gt-contours-dir",
        type=Path,
        default=Path("Papila/ExpertsSegmentations/Contours"),
        help="Directory containing Papila contour text files",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="Optional backbone name (e.g. inception_v3, densenet121). Requires matching classifier weights.",
    )
    return parser.parse_args()


def load_allowed_ids(path: Path, dice_threshold: float) -> Set[str]:
    allowed: Set[str] = set()
    with path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            sample_id = row.get("sample_id")
            if not sample_id or sample_id == "__mean__":
                continue
            try:
                disc = float(row.get("dice_disc", "nan"))
                cup = float(row.get("dice_cup", "nan"))
            except ValueError:
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
    allowed_ids: Set[str],
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
        sample_id = f"papila_{image_path.stem}"
        if sample_id not in allowed_ids or sample_id in samples:
            continue
        value = row.get(label_col)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        label: Optional[int]
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
            split="eval",
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


def contour_to_mask(coords: np.ndarray, size: Sequence[int]) -> np.ndarray:
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


def build_backbone(name: Optional[str]) -> Optional[torch.nn.Module]:
    if not name:
        return None
    key = name.lower()
    if key not in BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'. Available: {', '.join(sorted(BACKBONES.keys()))}")
    spec = BACKBONES[key]
    model = spec.ctor(weights=spec.weights_default)
    out_dim, model = spec.strip(model)
    setattr(model, "_feature_dim", out_dim)
    if key == "refugelike":
        load_backbone_weights(key, model)
    return model


def evaluate_records(
    clf: RefugeClassification,
    records: Sequence[RefugeClassificationRecord],
    device: str,
    batch_size: int,
) -> Dict[str, float]:
    dataset = RefugeClassificationDataset(
        records,
        transform=clf.eval_transform,
        polar_transform=clf.polar_transform,
        size=clf.crop_size,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    clf.backbone.to(device).eval()
    clf.classifier_head.to(device).eval()
    preds: List[float] = []
    targets: List[int] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Papila Eval", leave=False, unit="batch"):
            images = batch["image"].to(device)
            polars = batch["polar"].to(device)
            extra_feats = batch["features"].to(device)
            labels = batch["label"].cpu().numpy().tolist()
            feats_img = clf.backbone(images)
            feats = feats_img
            if clf.use_polar:
                feats_polar = clf.backbone(polars)
                feats = torch.cat([feats, feats_polar], dim=1)
            if clf.extra_feature_dim > 0:
                feats = torch.cat([feats, extra_feats], dim=1)
            logits = clf.classifier_head(feats)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist()
            preds.extend(probs)
            targets.extend(labels)
    metrics: Dict[str, float] = {"count": float(len(targets))}
    unique_labels = set(targets)
    if len(unique_labels) >= 2:
        metrics["auc"] = float(torchmetrics_auc(targets, preds))
    else:
        metrics["auc"] = float("nan")
    preds_bin = [1 if p >= 0.5 else 0 for p in preds]
    accuracy = sum(int(p == t) for p, t in zip(preds_bin, targets)) / max(1, len(targets))
    metrics["accuracy"] = float(accuracy)
    metrics["mean_prob"] = float(np.mean(preds)) if preds else float("nan")
    metrics["labels_pos"] = float(sum(targets))
    if preds:
        metrics["probs_std"] = float(np.std(preds))
    return metrics


def torchmetrics_auc(targets: Sequence[int], preds: Sequence[float]) -> float:
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required to compute AUC") from exc

    return float(roc_auc_score(targets, preds))


def main() -> None:
    args = parse_args()
    device = args.device

    allowed_ids = load_allowed_ids(args.filtered_metrics, args.dice_threshold)
    if not allowed_ids:
        raise SystemExit("No Papila samples passed the Dice threshold.")

    papila_samples = build_papila_samples(
        args.image_dir,
        args.clinical_dir,
        args.label_col,
        args.positive_labels,
        allowed_ids,
    )
    if not papila_samples:
        raise SystemExit("No Papila samples with labels matched the filtered metrics.")

    cache_dir = args.cache_dir
    if args.use_gt_masks and cache_dir is not None:
        cache_dir = cache_dir / "gt"

    if args.use_gt_masks:
        geometry_provider = PapilaGTGeometryProvider(args.gt_contours_dir)
        segmenter = None
    else:
        segmenter = UNetSegmenter(
            manifest_path=args.segmenter_manifest,
            device=device,
            normalize=args.segmenter_normalize,
        )
        seg_state = torch.load(args.segmenter_weights, map_location=device)
        seg_state_dict = seg_state.get("model", seg_state)
        segmenter.model.load_state_dict(seg_state_dict)
        segmenter.model.to(device)
        geometry_provider = UNetGeometryProvider(
            segmenter=segmenter,
            threshold=args.segmenter_threshold,
            tta=args.segmenter_tta,
        )

    pre = RefugePreprocessing(args.refuge_root)
    dummy_seg = RefugeSegmentation(pre)
    backbone = build_backbone(args.backbone)
    clf = RefugeClassification(
        pre,
        dummy_seg,
        geometry_fn=geometry_provider,
        cache_dir=cache_dir,
        backbone=backbone,
    )
    clf.crop_scale = args.crop_scale
    clf.crop_size = args.crop_size
    clf.eval_transform = _default_image_transform(args.crop_size)
    clf.ttt_transform = clf.eval_transform

    if args.classifier_weights is not None:
        clf_state = torch.load(args.classifier_weights, map_location=device)
        clf.backbone.load_state_dict(clf_state["backbone"])
        clf.classifier_head.load_state_dict(clf_state["classifier"])
        clf.rotation_head.load_state_dict(clf_state["rotation"])
        if "feature_reg" in clf_state and getattr(clf, "feature_reg_head", None) is not None:
            clf.feature_reg_head.load_state_dict(clf_state["feature_reg"])

    records = clf.build_records_for_samples(
        papila_samples,
        crop_scale=args.crop_scale,
        progress_prefix="papila_eval",
    )
    if not records:
        raise SystemExit("Unable to build any records; check geometry predictions or labels.")

    metrics = evaluate_records(clf, records, device=device, batch_size=args.batch_size)
    print(f"Samples evaluated: {int(metrics['count'])}")
    print(f"AUC: {metrics['auc']:.4f}" if not np.isnan(metrics['auc']) else "AUC: NaN")
    print(f"Accuracy @0.5: {metrics['accuracy']:.4f}")
    print(f"Mean glaucoma prob: {metrics['mean_prob']:.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["sample_id", "prob_glaucoma", "label"])
            clf.backbone.eval()
            clf.classifier_head.eval()
            dataset = RefugeClassificationDataset(
                records,
                transform=clf.eval_transform,
                polar_transform=clf.polar_transform,
                size=clf.crop_size,
            )
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            with torch.no_grad():
                for batch in tqdm(loader, desc="Papila Output", leave=False, unit="batch"):
                    images = batch["image"].to(device)
                    polars = batch["polar"].to(device)
                    extra_feats = batch["features"].to(device)
                    ids = batch["sample_id"]
                    labels = batch["label"].tolist()
                    feats_img = clf.backbone(images)
                    feats = feats_img
                    if clf.use_polar:
                        feats_polar = clf.backbone(polars)
                        feats = torch.cat([feats, feats_polar], dim=1)
                    if clf.extra_feature_dim > 0:
                        feats = torch.cat([feats, extra_feats], dim=1)
                    logits = clf.classifier_head(feats)
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist()
                    for sid, prob, label in zip(ids, probs, labels):
                        writer.writerow([sid, prob, label])
        print(f"Per-sample probabilities written to {args.output}")


if __name__ == "__main__":
    main()
