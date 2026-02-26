"""REFUGE glaucoma classification with rotation-based TTT."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import random

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import functional as TF
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from skimage.transform import warp_polar
from tqdm import tqdm

from classes.geometry_features import (
    FEATURE_DIM,
    EPS,
    compute_geometry_features,
    disc_cup_from_mask_image,
)
from classes.refuge_preprocessing import RefugePreprocessing, RefugeSample
from classes.refuge_segmentation import RefugeSegmentation
from classes.unet_segmenter import UNetSegmenter


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------


def _default_image_transform(size: int = 256) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _augment_image_transform(size: int = 256) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _crop_from_geometry(image: Image.Image, geometry: Dict[str, float], size: int = 256) -> Image.Image:
    cx, cy = geometry["centre_x"], geometry["centre_y"]
    r = geometry["crop_radius"]
    left = max(0.0, cx - r)
    upper = max(0.0, cy - r)
    right = min(image.width, cx + r)
    lower = min(image.height, cy + r)
    crop = image.crop((left, upper, right, lower))
    return crop.resize((size, size), Image.BILINEAR)


def _geometry_from_mask(mask: np.ndarray, scale: float) -> Dict[str, float]:
    mask = np.asarray(mask) > 0
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise RuntimeError("Empty mask; cannot derive geometry")
    ys, xs = coords[:, 0], coords[:, 1]
    centre_x = float(xs.mean())
    centre_y = float(ys.mean())
    width = float(xs.max() - xs.min())
    height = float(ys.max() - ys.min())
    diameter = max(width, height)
    radius = diameter / 2.0
    crop_radius = radius * scale
    return {
        "centre_x": centre_x,
        "centre_y": centre_y,
        "radius": radius,
        "crop_radius": crop_radius,
        "crop_size": crop_radius * 2.0,
    }


def _compute_feature_vector(disc_mask: np.ndarray, cup_mask: np.ndarray) -> np.ndarray:
    return compute_geometry_features(disc_mask, cup_mask)


def _compute_polar_image(crop: Image.Image, size: int) -> Image.Image:
    arr = np.asarray(crop).astype(np.float32) / 255.0
    radius = min(arr.shape[0], arr.shape[1]) / 2.0
    polar = warp_polar(
        arr,
        radius=radius,
        scaling="linear",
        channel_axis=-1,
    )
    polar = np.clip(polar, 0.0, 1.0)
    polar_img = Image.fromarray((polar * 255).astype(np.uint8))
    return polar_img.resize((size, size), Image.BILINEAR)


def _crop_mask_from_geometry(mask: np.ndarray, geometry: Dict[str, float], size: int) -> np.ndarray:
    mask_img = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    cx, cy = geometry["centre_x"], geometry["centre_y"]
    r = geometry["crop_radius"]
    left = max(0.0, cx - r)
    upper = max(0.0, cy - r)
    right = min(mask_img.width, cx + r)
    lower = min(mask_img.height, cy + r)
    crop = mask_img.crop((left, upper, right, lower)).resize((size, size), Image.NEAREST)
    return (np.asarray(crop) > 0).astype(np.uint8)


@dataclass
class RefugeClassificationRecord:
    sample: RefugeSample
    geometry: Dict[str, float]
    disc_mask: Optional[np.ndarray] = None
    cup_mask: Optional[np.ndarray] = None


class RefugeClassificationDataset(Dataset):
    def __init__(
        self,
        records: Sequence[RefugeClassificationRecord],
        transform: transforms.Compose,
        polar_transform: transforms.Compose,
        size: int = 256,
    ) -> None:
        self.records = list(records)
        self.transform = transform
        self.polar_transform = polar_transform
        self.size = size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        image = Image.open(rec.sample.image_path).convert("RGB")
        crop = _crop_from_geometry(image, rec.geometry, size=self.size)
        polar_image = _compute_polar_image(crop, size=self.size)
        tensor = self.transform(crop)
        polar_tensor = self.polar_transform(polar_image)

        features = np.zeros((FEATURE_DIM,), dtype=np.float32)
        if rec.disc_mask is not None and rec.cup_mask is not None:
            disc_crop = _crop_mask_from_geometry(rec.disc_mask, rec.geometry, self.size)
            cup_crop = _crop_mask_from_geometry(rec.cup_mask, rec.geometry, self.size)
            features = _compute_feature_vector(disc_crop, cup_crop)

        feature_tensor = torch.from_numpy(features).float()
        label = rec.sample.label
        if label is None:
            raise ValueError(f"Sample {rec.sample.sample_id} is missing glaucoma label")
        return {
            "image": tensor,
            "polar": polar_tensor,
            "features": feature_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "sample_id": rec.sample.sample_id,
        }


class RefugeTTTDataset(Dataset):
    """Dataset providing unlabeled crops for test-time training."""

    def __init__(self, records: Sequence[RefugeClassificationRecord], transform: transforms.Compose, size: int = 256) -> None:
        self.records = list(records)
        self.transform = transform
        self.size = size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> torch.Tensor:
        rec = self.records[idx]
        image = Image.open(rec.sample.image_path).convert("RGB")
        crop = _crop_from_geometry(image, rec.geometry, size=self.size)
        return self.transform(crop)


class UNetGeometryProvider:
    """Callable wrapper that derives disc geometry using a trained UNetSegmenter."""

    def __init__(
        self,
        segmenter: UNetSegmenter,
        threshold: float = 0.5,
        tta: bool = False,
    ) -> None:
        self.segmenter = segmenter
        self.threshold = threshold
        self.tta = tta
        self.segmenter.model.eval()

    def __call__(self, sample: RefugeSample, scale: float) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        image = Image.open(sample.image_path).convert("RGB")
        resized = self.segmenter.preprocess_image(image)
        tensor = transforms.ToTensor()(resized)
        tensor = self.segmenter._normalize_tensor(tensor)
        tensor = tensor.unsqueeze(0).to(self.segmenter.device)
        with torch.no_grad():
            logits = self.segmenter.model(tensor)
            if self.tta:
                t_h = torch.flip(tensor, dims=[3])
                log_h = self.segmenter.model(t_h)
                log_h = torch.flip(log_h, dims=[3])
                t_v = torch.flip(tensor, dims=[2])
                log_v = self.segmenter.model(t_v)
                log_v = torch.flip(log_v, dims=[2])
                logits = (logits + log_h + log_v) / 3.0
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        disc_pred = (probs[0] > self.threshold).astype(np.uint8) * 255
        cup_pred = (probs[1] > self.threshold).astype(np.uint8) * 255
        disc_img = Image.fromarray(disc_pred, mode="L").resize(image.size, Image.NEAREST)
        cup_img = Image.fromarray(cup_pred, mode="L").resize(image.size, Image.NEAREST)
        disc_mask = (np.array(disc_img, dtype=np.uint8) > 0).astype(np.uint8)
        cup_mask = (np.array(cup_img, dtype=np.uint8) > 0).astype(np.uint8)
        cup_mask = (cup_mask > 0) & (disc_mask > 0)
        cup_mask = cup_mask.astype(np.uint8)
        geom = _geometry_from_mask(disc_mask, scale)
        return geom, disc_mask, cup_mask


# ---------------------------------------------------------------------------
# Classification module
# ---------------------------------------------------------------------------


class ArcMarginProduct(nn.Module):
    """Additive angular margin (ArcFace) head."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 30.0,
        m: float = 0.5,
        easy_margin: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = float(s)
        self.m = float(m)
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input: torch.Tensor, label: Optional[torch.Tensor] = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if label is None:
            return cosine * self.s

        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=0.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s
        return logits


class RefugeClassification:
    """Train and evaluate REFUGE glaucoma classifiers with TTT support."""

    def __init__(
        self,
        preprocessing: RefugePreprocessing,
        segmentation: RefugeSegmentation,
        backbone: Optional[nn.Module] = None,
        geometry_fn: Optional[
            Callable[
                [RefugeSample, float],
                Tuple[Dict[str, float], Optional[np.ndarray], Optional[np.ndarray]],
            ]
        ] = None,
        cache_dir: Optional[Path] = None,
        use_all_labeled: bool = False,
        auto_val_ratio: float = 0.1,
        use_margin: bool = False,
        margin_s: float = 30.0,
        margin_m: float = 0.5,
    ) -> None:
        self.preprocessing = preprocessing
        self.segmentation = segmentation
        if backbone is not None:
            self.backbone = backbone
            in_features = getattr(self.backbone, "_feature_dim", None)
            if in_features is None:
                if hasattr(self.backbone, "fc") and hasattr(self.backbone.fc, "in_features"):
                    in_features = self.backbone.fc.in_features  # type: ignore[attr-defined]
                    self.backbone.fc = nn.Identity()  # type: ignore[attr-defined]
                else:
                    raise ValueError(
                        "Provided backbone must have '_feature_dim' or expose fc.in_features"
                    )
        else:
            self.backbone = self._default_backbone()
            in_features = getattr(self.backbone, "_feature_dim", None)
            if in_features is None:
                in_features = self.backbone.fc.in_features  # type: ignore[attr-defined]
                self.backbone.fc = nn.Identity()  # type: ignore[attr-defined]
        self.feature_dim = in_features
        self.use_polar = True
        self.extra_feature_dim = FEATURE_DIM
        combined_dim = self.feature_dim * (1 + int(self.use_polar)) + self.extra_feature_dim
        self.margin_s = float(margin_s)
        self.margin_m = float(margin_m)
        self.use_margin = bool(use_margin)
        if self.use_margin:
            self.classifier_head = ArcMarginProduct(
                combined_dim, 2, s=self.margin_s, m=self.margin_m
            )
        else:
            self.classifier_head = nn.Linear(combined_dim, 2)
        self.rotation_head = nn.Linear(self.feature_dim, 4)

        self.train_dataset: Optional[RefugeClassificationDataset] = None
        self.val_dataset: Optional[RefugeClassificationDataset] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.ttt_transform = _default_image_transform()
        self.train_transform = _augment_image_transform()
        self.eval_transform = _default_image_transform()
        self.polar_transform = _default_image_transform()
        self.crop_scale = 2.5
        self.crop_size = 256
        self.geometry_cache: Dict[
            str, Tuple[Dict[str, float], Optional[np.ndarray], Optional[np.ndarray]]
        ] = {}
        self.train_records: List[RefugeClassificationRecord] = []
        self.val_records: List[RefugeClassificationRecord] = []
        self._geometry_fn = geometry_fn
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_all_labeled = use_all_labeled
        self.auto_val_ratio = auto_val_ratio

    # ------------------------------------------------------------------
    @staticmethod
    def _default_backbone() -> nn.Module:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Identity()
        setattr(model, "_feature_dim", in_features)
        return model

    # ------------------------------------------------------------------
    def build_datasets(
        self,
        crop_scale: float = 2.5,
        crop_size: int = 256,
        batch_size: int = 16,
        num_workers: int = 4,
    ) -> None:
        self.crop_scale = crop_scale
        self.crop_size = crop_size
        self.train_transform = _augment_image_transform(crop_size)
        self.eval_transform = _default_image_transform(crop_size)
        self.ttt_transform = _default_image_transform(crop_size)
        self.polar_transform = _default_image_transform(crop_size)

        manifest = list(self.preprocessing.build_manifest())
        train_records: List[RefugeClassificationRecord] = []
        val_records: List[RefugeClassificationRecord] = []

        allowed_splits = {"train", "val"}
        candidates = [
            sample
            for sample in manifest
            if sample.label is not None and sample.split in allowed_splits
        ]

        print(
            f"[classifier] Building datasets from {len(candidates)} labelled samples (train/val)"
        )

        skipped: List[str] = []
        for sample in tqdm(
            candidates,
            desc="Preparing records",
            unit="sample",
            leave=False,
        ):
            try:
                geom, disc_mask, cup_mask = self._resolve_geometry(sample, crop_scale)
            except RuntimeError:
                skipped.append(sample.sample_id)
                continue
            record = RefugeClassificationRecord(
                sample=sample,
                geometry=geom,
                disc_mask=disc_mask,
                cup_mask=cup_mask,
            )
            if sample.split == "train" or (
                self.use_all_labeled and sample.split == "val"
            ):
                train_records.append(record)
            else:
                val_records.append(record)

        if skipped:
            print(
                f"[classifier] WARNING: {len(skipped)}/{len(candidates)} samples skipped "
                f"due to empty segmentation mask: {skipped}"
            )

        if (not val_records or self.use_all_labeled) and train_records and self.auto_val_ratio > 0.0:
            rng = random.Random(42)
            label_groups: Dict[int, List[RefugeClassificationRecord]] = {}
            for rec in train_records:
                label = int(rec.sample.label or 0)
                label_groups.setdefault(label, []).append(rec)

            new_train: List[RefugeClassificationRecord] = []
            new_val: List[RefugeClassificationRecord] = []
            for recs in label_groups.values():
                rng.shuffle(recs)
                if len(recs) <= 1:
                    new_train.extend(recs)
                    continue
                val_count = max(1, int(round(len(recs) * self.auto_val_ratio)))
                if val_count >= len(recs):
                    val_count = len(recs) - 1
                new_val.extend(recs[:val_count])
                new_train.extend(recs[val_count:])

            if not new_val:
                # Fallback: ensure at least one validation sample if possible
                if len(new_train) > 1:
                    new_val.append(new_train.pop())

            if new_val:
                val_records = new_val
                train_records = new_train

        self.train_records = train_records
        self.val_records = val_records

        print(
            f"[classifier] Records ready → train: {len(train_records)}, val: {len(val_records)}"
        )

        self.train_dataset = RefugeClassificationDataset(
            train_records,
            transform=self.train_transform,
            polar_transform=self.polar_transform,
            size=crop_size,
        )
        self.val_dataset = RefugeClassificationDataset(
            val_records,
            transform=self.eval_transform,
            polar_transform=self.polar_transform,
            size=crop_size,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        print(
            "[classifier] DataLoaders prepared — training batches will start shortly"
        )

    # ------------------------------------------------------------------
    def _resolve_geometry(
        self, sample: RefugeSample, scale: float
    ) -> Tuple[Dict[str, float], Optional[np.ndarray], Optional[np.ndarray]]:
        key = self._cache_key(sample.sample_id, scale)
        cached = self.geometry_cache.get(key)
        if cached is not None:
            return cached

        cache_path = self._cache_path(sample.sample_id, scale)
        if cache_path is not None and cache_path.exists():
            data = np.load(cache_path, allow_pickle=False)
            geom = {
                "centre_x": float(data["centre_x"]),
                "centre_y": float(data["centre_y"]),
                "radius": float(data["radius"]),
                "crop_radius": float(data["crop_radius"]),
                "crop_size": float(data["crop_size"]),
            }
            disc_mask = None
            cup_mask = None
            if int(data["has_disc"]):
                disc_mask = data["disc_mask"].astype(np.uint8)
            if int(data["has_cup"]):
                cup_mask = data["cup_mask"].astype(np.uint8)
            self.geometry_cache[key] = (geom, disc_mask, cup_mask)
            return geom, disc_mask, cup_mask

        disc_mask: Optional[np.ndarray] = None
        cup_mask: Optional[np.ndarray] = None

        if sample.mask_path and sample.mask_path.exists():
            mask_img = Image.open(sample.mask_path).convert("RGB")
            disc_mask, cup_mask = disc_cup_from_mask_image(mask_img)
            geom = _geometry_from_mask(disc_mask, scale)
        elif self._geometry_fn is not None:
            geom, disc_mask, cup_mask = self._geometry_fn(sample, scale)
        else:
            geom = self.segmentation.infer_disc_geometry(sample, scale=scale)
            try:
                pred_mask = self.segmentation.predict_mask(sample).numpy()
                disc_mask = pred_mask.astype(np.uint8)
            except Exception:
                disc_mask = None
            cup_mask = None

        if cache_path is not None:
            try:
                np.savez_compressed(
                    cache_path,
                    centre_x=geom["centre_x"],
                    centre_y=geom["centre_y"],
                    radius=geom["radius"],
                    crop_radius=geom["crop_radius"],
                    crop_size=geom.get("crop_size", geom["crop_radius"] * 2.0),
                    disc_mask=disc_mask if disc_mask is not None else np.array([], dtype=np.uint8),
                    cup_mask=cup_mask if cup_mask is not None else np.array([], dtype=np.uint8),
                    has_disc=int(disc_mask is not None),
                    has_cup=int(cup_mask is not None),
                )
            except Exception:
                pass

        self.geometry_cache[key] = (geom, disc_mask, cup_mask)
        return geom, disc_mask, cup_mask

    def set_geometry_fn(
        self,
        geometry_fn: Optional[
            Callable[
                [RefugeSample, float],
                Tuple[Dict[str, float], Optional[np.ndarray], Optional[np.ndarray]],
            ]
        ],
    ) -> None:
        self._geometry_fn = geometry_fn
        self.geometry_cache.clear()

    def build_records_for_samples(
        self,
        samples: Sequence[RefugeSample],
        crop_scale: Optional[float] = None,
        progress_prefix: Optional[str] = None,
    ) -> List[RefugeClassificationRecord]:
        scale = crop_scale if crop_scale is not None else self.crop_scale
        records: List[RefugeClassificationRecord] = []
        skipped: List[str] = []
        iterator: Iterable[RefugeSample]
        if progress_prefix is not None:
            iterator = tqdm(samples, desc=progress_prefix, unit="sample", leave=False)
        else:
            iterator = samples
        labeled = [s for s in samples if s.label is not None]
        for sample in iterator:
            if sample.label is None:
                continue
            try:
                geom, disc_mask, cup_mask = self._resolve_geometry(sample, scale)
            except RuntimeError:
                skipped.append(sample.sample_id)
                continue
            records.append(
                RefugeClassificationRecord(
                    sample=sample,
                    geometry=geom,
                    disc_mask=disc_mask,
                    cup_mask=cup_mask,
                )
            )
        prefix = f"[{progress_prefix}]" if progress_prefix else "[classifier]"
        if skipped:
            print(
                f"{prefix} WARNING: {len(skipped)}/{len(labeled)} samples skipped "
                f"due to empty segmentation mask: {skipped}"
            )
        else:
            print(f"{prefix} All {len(labeled)} samples processed successfully.")
        return records

    def clear_disk_cache(self) -> None:
        """Delete all cached geometry/mask .npz files in cache_dir."""
        if self.cache_dir is None or not self.cache_dir.exists():
            return
        removed = 0
        for f in self.cache_dir.glob("*.npz"):
            f.unlink()
            removed += 1
        self.geometry_cache.clear()
        print(f"[classifier] Cleared {removed} cached geometry files from {self.cache_dir}")

    def _cache_key(self, sample_id: str, scale: float) -> str:
        scale_tag = int(round(scale * 100))
        return f"{sample_id}_s{scale_tag}"

    def _cache_path(self, sample_id: str, scale: float) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{self._cache_key(sample_id, scale)}.npz"

    # ------------------------------------------------------------------
    def train(
        self,
        epochs: int = 30,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        device: Optional[str] = None,
        rotation_weight: float = 0.5,
        checkpoint_dir: Optional[Path] = None,
    ) -> Dict[str, float]:
        if self.train_loader is None or self.val_loader is None:
            self.build_datasets()

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone.to(device)
        self.classifier_head.to(device)
        self.rotation_head.to(device)

        params = list(self.backbone.parameters()) + list(self.classifier_head.parameters()) + list(self.rotation_head.parameters())
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        clf_loss = nn.CrossEntropyLoss()
        rot_loss = nn.CrossEntropyLoss()

        best_auc = 0.0
        history: Dict[str, float] = {}

        epoch_iter = tqdm(range(1, epochs + 1), desc="Epochs", unit="epoch")

        print(
            f"[classifier] Starting training for {epochs} epochs with batch size {self.train_loader.batch_size}"
        )

        for epoch in epoch_iter:
            self.backbone.train()
            self.classifier_head.train()
            self.rotation_head.train()
            running_loss = 0.0

            batch_iter = tqdm(
                self.train_loader,  # type: ignore[arg-type]
                desc=f"Train {epoch}/{epochs}",
                leave=False,
                unit="batch",
            )

            for batch in batch_iter:
                images = batch["image"].to(device)
                polars = batch["polar"].to(device)
                extra_feats = batch["features"].to(device)
                labels = batch["label"].to(device)
                optimizer.zero_grad()

                feats_img = self.backbone(images)
                feats = feats_img
                if self.use_polar:
                    feats_polar = self.backbone(polars)
                    feats = torch.cat([feats, feats_polar], dim=1)
                if self.extra_feature_dim > 0:
                    feats = torch.cat([feats, extra_feats], dim=1)
                if self.use_margin:
                    logits = self.classifier_head(feats, labels)
                else:
                    logits = self.classifier_head(feats)
                loss_cls = clf_loss(logits, labels)

                rot_imgs, rot_labels = self._build_rotation_batch(images)
                feats_rot = self.backbone(rot_imgs)
                logits_rot = self.rotation_head(feats_rot)
                loss_rot = rot_loss(logits_rot, rot_labels)

                loss = loss_cls + rotation_weight * loss_rot
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)

            train_loss = running_loss / len(self.train_loader.dataset)  # type: ignore[arg-type]
            metrics = self.evaluate(device=device)
            history[f"epoch_{epoch}_loss"] = train_loss
            history[f"epoch_{epoch}_auc"] = metrics.get("auc", float("nan"))

            auc_val = metrics.get("auc", 0.0)
            epoch_iter.set_postfix(loss=f"{train_loss:.4f}", auc=f"{auc_val:.4f}")

            if auc_val > best_auc:
                best_auc = metrics["auc"]
                if checkpoint_dir is not None:
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "backbone": self.backbone.state_dict(),
                        "classifier": self.classifier_head.state_dict(),
                        "rotation": self.rotation_head.state_dict(),
                    }, checkpoint_dir / "refuge_classifier_best.pt")

        return {"best_auc": best_auc, **history}

    # ------------------------------------------------------------------
    def evaluate(
        self,
        split: str = "val",
        apply_ttt: bool = False,
        device: Optional[str] = None,
    ) -> Dict[str, float]:
        if split != "val":
            raise ValueError("Only validation split supported currently")
        if self.val_loader is None:
            self.build_datasets()

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone.to(device)
        self.classifier_head.to(device)
        self.rotation_head.to(device)

        if apply_ttt:
            ttt_ds = RefugeTTTDataset(self.val_records, transform=self.ttt_transform, size=self.crop_size)
            ttt_loader = DataLoader(ttt_ds, batch_size=32, shuffle=False)
            self.apply_ttt(ttt_loader, device=device)

        self.backbone.eval()
        self.classifier_head.eval()
        preds: List[float] = []
        targets: List[int] = []

        with torch.no_grad():
            val_iter = tqdm(self.val_loader, desc="Validate", leave=False, unit="batch")
            for batch in val_iter:  # type: ignore[arg-type]
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                polars = batch["polar"].to(device)
                extra_feats = batch["features"].to(device)
                feats_img = self.backbone(images)
                feats = feats_img
                if self.use_polar:
                    feats_polar = self.backbone(polars)
                    feats = torch.cat([feats, feats_polar], dim=1)
                if self.extra_feature_dim > 0:
                    feats = torch.cat([feats, extra_feats], dim=1)
                if self.use_margin:
                    logits = self.classifier_head(feats)
                else:
                    logits = self.classifier_head(feats)
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds.extend(probs.cpu().numpy().tolist())
                targets.extend(labels.cpu().numpy().tolist())

        auc = 0.0
        try:
            if len(set(targets)) > 1:
                auc = float(roc_auc_score(targets, preds))
        except ValueError:
            auc = 0.0

        return {"auc": auc}

    # ------------------------------------------------------------------
    def apply_ttt(self, loader: DataLoader, device: Optional[str] = None, steps: int = 1, lr: float = 1e-5) -> None:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone.to(device)
        self.rotation_head.to(device)
        self.backbone.train()
        self.rotation_head.train()

        optimizer = torch.optim.Adam(list(self.backbone.parameters()) + list(self.rotation_head.parameters()), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for _ in range(steps):
            for batch in tqdm(loader, desc="TTT adapt", leave=False, unit="batch"):
                if isinstance(batch, dict):
                    images = batch["image"].to(device)
                else:
                    images = batch.to(device)
                optimizer.zero_grad()
                rot_imgs, rot_labels = self._build_rotation_batch(images)
                feats = self.backbone(rot_imgs)
                logits = self.rotation_head(feats)
                loss = criterion(logits, rot_labels)
                loss.backward()
                optimizer.step()

    # ------------------------------------------------------------------
    def extract_backbone(self) -> nn.Module:
        return self.backbone

    def save_checkpoint(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "backbone": self.backbone.state_dict(),
            "classifier": self.classifier_head.state_dict(),
            "rotation": self.rotation_head.state_dict(),
        }, output_dir / "refuge_classifier.pt")

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        payload = torch.load(checkpoint_path, map_location="cpu")
        self.backbone.load_state_dict(payload["backbone"])
        self.classifier_head.load_state_dict(payload["classifier"])
        self.rotation_head.load_state_dict(payload["rotation"])

    # ------------------------------------------------------------------
    def _build_rotation_batch(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rotations = [0, 90, 180, 270]
        rotated = []
        labels = []
        for idx, angle in enumerate(rotations):
            rot = TF.rotate(images, angle)
            rotated.append(rot)
            labels.append(torch.full((images.size(0),), idx, dtype=torch.long, device=images.device))
        batch = torch.cat(rotated, dim=0)
        batch_labels = torch.cat(labels, dim=0)
        return batch, batch_labels
