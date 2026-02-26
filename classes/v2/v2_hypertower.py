from __future__ import annotations
import os
import math
import argparse
import json
import time
import copy
import csv
from pathlib import Path
import shutil
import random as pyrandom
from dataclasses import dataclass
import pandas as pd
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from classes.v2.data_bundle import DataBundle
from classes.v2.dataset import ClinicalDataset
from classes.early_stop import EarlyStopper
from classes.unet_segmenter import UNetSegmenter
from classes.geometry_features import (
    FEATURE_DIM,
    compute_geometry_features,
    disc_cup_from_mask_image,
)
from classes.refuge_classification import _geometry_from_mask
from classes.v2.bridges import Bridge, VoteBridge
from classes.v2.hypertower_logger import HypertowerLogger
from classes.v2.towers import ImageTower, MDTower
from torchvision import transforms

# from clinical_data import ClinicalData
# from dataset import ClinicalDataset
# from image_tower import ImageTower
# from md_tower import MDTower
# from bridge import Bridge, VoteBridge
from random import random
from sklearn.metrics import roc_auc_score, cohen_kappa_score, f1_score, matthews_corrcoef, recall_score
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from typing import Optional, Tuple
from types import SimpleNamespace
from classes.backbones import BACKBONES
from classes.v2.papila_builders import build_papila_data
from classes.v2.split_manager import PatientFirstSplitManager


LOG_FIELDS = [
"epoch",
"eval_loss",
"acc_fused", "acc_img", "acc_md",
"auc_fused", "auc_img", "auc_md",
"top2_fused", "top2_img", "top2_md",
"margin_fused", "margin_img", "margin_md",
"agree_fused_img", "agree_fused_md",
"pct_fused", "pct_img", "pct_md",
"phase",
"holdout_loss",
"holdout_acc_fused",
"holdout_acc_img",
"holdout_acc_md",
"holdout_auc_fused",
"holdout_auc_img",
"holdout_auc_md",
]


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 0.0,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Standard focal loss wrapper. When gamma=0 it reduces to cross entropy.
    weight should be per-class weights (same semantics as CrossEntropyLoss).
    """
    if gamma <= 0:
        return F.cross_entropy(logits, targets, weight=weight, reduction=reduction)

    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()

    targets = targets.long().view(-1, 1)
    logpt = log_probs.gather(1, targets)
    pt = probs.gather(1, targets)

    focal_factor = (1.0 - pt).clamp_min(0.0) ** gamma
    loss = -focal_factor * logpt

    if weight is not None:
        class_weight = weight.gather(0, targets.view(-1))
        loss = loss * class_weight.view(-1, 1)

    loss = loss.view(-1)
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


class UNetImageCropper:
    def __init__(
        self,
        manifest_path: Path,
        weights_path: Path,
        normalize: str = "per_image",
        threshold: float = 0.5,
        tta: bool = False,
        scale: float = 2.5,
        target_size: int = 224,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.segmenter = UNetSegmenter(
            manifest_path=manifest_path,
            normalize=normalize,
        )
        state = torch.load(weights_path, map_location=self.segmenter.device)
        state_dict = state.get("model", state)
        self.segmenter.model.load_state_dict(state_dict)
        self.segmenter.model.to(self.segmenter.device)
        self.segmenter.model.eval()

        self.threshold = threshold
        self.tta = tta
        self.scale = scale
        self.target_size = target_size
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.to_tensor = transforms.ToTensor()

    def _cache_path(self, image_path: Path) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        stem = image_path.stem
        return self.cache_dir / f"{stem}_s{int(self.scale * 100)}.npz"

    def clear_cache(self) -> None:
        if self.cache_dir is None or not self.cache_dir.exists():
            return
        removed = sum(1 for f in self.cache_dir.glob("*.npz") if f.unlink() or True)
        print(f"[UNetImageCropper] Cleared {removed} cached crop files from {self.cache_dir}")

    def _infer_masks(self, image: Image.Image) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        resized = self.segmenter.preprocess_image(image)
        tensor = self.to_tensor(resized).unsqueeze(0).to(self.segmenter.device)

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
        disc_mask = np.array(disc_img, dtype=np.uint8)
        cup_img = Image.fromarray(cup_pred, mode="L").resize(image.size, Image.NEAREST)
        cup_mask = (np.array(cup_img, dtype=np.uint8) > 0).astype(np.uint8)
        cup_mask = (cup_mask > 0) & (disc_mask > 0)
        cup_mask = cup_mask.astype(np.uint8)
        disc_mask = (disc_mask > 0).astype(np.uint8)
        return disc_mask, cup_mask

    def _compute_crop_info(self, image: Image.Image, image_path: Path) -> Optional[dict]:
        image_path = Path(image_path).resolve()
        cache_path = self._cache_path(image_path)
        cached_bounds = None
        if cache_path is not None and cache_path.exists():
            data = np.load(cache_path, allow_pickle=False)
            try:
                cached_bounds = {
                    "left": float(data["left"]),
                    "upper": float(data["upper"]),
                    "right": float(data["right"]),
                    "lower": float(data["lower"]),
                }
                if "features" in data.files:
                    cached_bounds["features"] = data["features"].astype(np.float32)
                    return cached_bounds
            except KeyError:
                cached_bounds = None

        masks = self._infer_masks(image)
        if masks is None:
            return cached_bounds
        disc_mask, cup_mask = masks
        try:
            geom = _geometry_from_mask(disc_mask, self.scale)
        except Exception:
            return cached_bounds
        cx = geom["centre_x"]
        cy = geom["centre_y"]
        r = geom["crop_radius"]
        left = max(0.0, cx - r)
        upper = max(0.0, cy - r)
        right = min(float(image.width), cx + r)
        lower = min(float(image.height), cy + r)
        features = compute_geometry_features(disc_mask, cup_mask)

        info = {
            "left": left,
            "upper": upper,
            "right": right,
            "lower": lower,
            "features": features,
        }
        if cache_path is not None:
            np.savez(
                cache_path,
                left=left,
                upper=upper,
                right=right,
                lower=lower,
                width=float(image.width),
                height=float(image.height),
                scale=self.scale,
                target_size=self.target_size,
                features=features,
            )
        return info

    def __call__(self, image: Image.Image, image_path: Path) -> Image.Image:
        info = self._compute_crop_info(image, image_path)
        if info is None:
            return image
        left = info["left"]
        upper = info["upper"]
        right = info["right"]
        lower = info["lower"]
        if right <= left or lower <= upper:
            return image
        crop = image.crop((left, upper, right, lower))
        return crop.resize((self.target_size, self.target_size), Image.BILINEAR)

    def geometry_features(self, image: Image.Image, image_path: Path) -> Optional[np.ndarray]:
        info = self._compute_crop_info(image, image_path)
        if info is None:
            return None
        features = info.get("features")
        if features is None:
            return None
        return np.asarray(features, dtype=np.float32)


class ManifestImageCropper:
    def __init__(
        self,
        manifest_path: Path,
        scale: float = 2.5,
        target_size: int = 224,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.scale = scale
        self.target_size = target_size
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(manifest_path)
        self.entries: Dict[str, dict] = {}
        for _, row in df.iterrows():
            img_path = Path(row["image_path"]).resolve()
            self.entries[str(img_path)] = {
                "annotation_disc": row.get("annotation_disc"),
                "annotation_cup": row.get("annotation_cup"),
                "annotation_type_disc": row.get("annotation_type_disc"),
                "annotation_type_cup": row.get("annotation_type_cup"),
            }

    def _cache_path(self, image_path: Path) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{image_path.stem}_s{int(self.scale * 100)}.npz"

    def clear_cache(self) -> None:
        if self.cache_dir is None or not self.cache_dir.exists():
            return
        removed = sum(1 for f in self.cache_dir.glob("*.npz") if f.unlink() or True)
        print(f"[ManifestImageCropper] Cleared {removed} cached crop files from {self.cache_dir}")

    @staticmethod
    def _load_contour(path: Path) -> np.ndarray:
        coords = np.loadtxt(path)
        if coords.ndim == 1:
            coords = coords.reshape(-1, 2)
        return coords

    @staticmethod
    def _contour_to_mask(coords: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        if coords is None or coords.size == 0:
            return np.zeros((size[1], size[0]), dtype=np.uint8)
        img = Image.new("L", size, 0)
        draw = ImageDraw.Draw(img)
        points = [tuple(map(float, pt)) for pt in coords]
        draw.polygon(points, outline=1, fill=1)
        return np.array(img, dtype=np.uint8)

    def _load_masks(self, entry: dict, image: Image.Image) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        disc_path = entry.get("annotation_disc")
        cup_path = entry.get("annotation_cup")
        disc_type = (entry.get("annotation_type_disc") or "").lower()
        cup_type = (entry.get("annotation_type_cup") or "").lower()

        disc_mask: Optional[np.ndarray] = None
        cup_mask: Optional[np.ndarray] = None

        if disc_path and not pd.isna(disc_path):
            disc_path = Path(disc_path)
            try:
                if disc_type == "mask":
                    mask_img = Image.open(disc_path)
                    mask_img = mask_img.resize(image.size, Image.NEAREST)
                    disc_mask, cup_from_mask = disc_cup_from_mask_image(mask_img)
                    if cup_from_mask.sum() > 0:
                        cup_mask = cup_from_mask
                elif disc_type == "contour":
                    coords = self._load_contour(disc_path)
                    disc_mask = self._contour_to_mask(coords, image.size)
            except Exception:
                disc_mask = None

        if cup_mask is None and cup_path and not pd.isna(cup_path):
            cup_path = Path(cup_path)
            try:
                if cup_type == "mask":
                    mask_img = Image.open(cup_path)
                    mask_img = mask_img.resize(image.size, Image.NEAREST)
                    _, cup_mask = disc_cup_from_mask_image(mask_img)
                elif cup_type == "contour":
                    coords = self._load_contour(cup_path)
                    cup_mask = self._contour_to_mask(coords, image.size)
            except Exception:
                cup_mask = None

        if disc_mask is None:
            return None
        disc_mask = (disc_mask > 0).astype(np.uint8)
        if cup_mask is None:
            cup_mask = np.zeros_like(disc_mask, dtype=np.uint8)
        cup_mask = ((cup_mask > 0) & (disc_mask > 0)).astype(np.uint8)
        return disc_mask, cup_mask

    def _compute_crop_info(self, image: Image.Image, image_path: Path) -> Optional[dict]:
        image_path = Path(image_path).resolve()
        entry = self.entries.get(str(image_path))
        if entry is None:
            return None
        cache_path = self._cache_path(image_path)
        cached_bounds = None
        if cache_path is not None and cache_path.exists():
            data = np.load(cache_path, allow_pickle=False)
            try:
                cached_bounds = {
                    "left": float(data["left"]),
                    "upper": float(data["upper"]),
                    "right": float(data["right"]),
                    "lower": float(data["lower"]),
                }
                if "features" in data.files:
                    cached_bounds["features"] = data["features"].astype(np.float32)
                    return cached_bounds
            except KeyError:
                cached_bounds = None

        masks = self._load_masks(entry, image)
        if masks is None:
            return cached_bounds
        disc_mask, cup_mask = masks
        try:
            geom = _geometry_from_mask(disc_mask, self.scale)
        except Exception:
            return cached_bounds
        cx = geom["centre_x"]
        cy = geom["centre_y"]
        r = geom["crop_radius"]
        left = max(0.0, cx - r)
        upper = max(0.0, cy - r)
        right = min(float(image.width), cx + r)
        lower = min(float(image.height), cy + r)
        features = compute_geometry_features(disc_mask, cup_mask)

        info = {
            "left": left,
            "upper": upper,
            "right": right,
            "lower": lower,
            "features": features,
        }
        if cache_path is not None:
            np.savez(
                cache_path,
                left=left,
                upper=upper,
                right=right,
                lower=lower,
                width=float(image.width),
                height=float(image.height),
                scale=self.scale,
                target_size=self.target_size,
                features=features,
            )
        return info

    def __call__(self, image: Image.Image, image_path: Path) -> Image.Image:
        info = self._compute_crop_info(image, image_path)
        if info is None:
            return image
        left = info["left"]
        upper = info["upper"]
        right = info["right"]
        lower = info["lower"]

        if right <= left or lower <= upper:
            return image
        crop = image.crop((left, upper, right, lower))
        return crop.resize((self.target_size, self.target_size), Image.BILINEAR)

    def geometry_features(self, image: Image.Image, image_path: Path) -> Optional[np.ndarray]:
        info = self._compute_crop_info(image, image_path)
        if info is None:
            return None
        features = info.get("features")
        if features is None:
            return None
        return np.asarray(features, dtype=np.float32)

def _num(x):
    """float(x) or None if NaN/None/invalid."""
    try:
        v = float(x)
    except Exception:
        return None
    return None if math.isnan(v) else v

class _ClinicalView:
    """Minimal shim so ClinicalDataset can iterate an epoch-specific DataFrame
    while still delegating encoding/paths/labels to the DataBundle object."""
    def __init__(self, base: DataBundle, df):
        self.base = base
        self.df = df

    @property
    def image_dir(self):
        return self.base.image_dir

    @property
    def clinical_dir(self):
        return self.base.clinical_dir

    @property
    def id_cols(self):
        return ("Patient ID", "eyeID")

    @property
    def label_col(self):
        return self.base.label_col

    @property
    def filename_template(self):
        # prefer whatever the dataset defined; otherwise fall back to RET{pid}{eye}.jpg style
        return getattr(self.base, "filename_template", "RET{pid:03d}{eye}.jpg")

    @property
    def dim(self):
        return self.base.feature_dim

    def encode_metadata(self, row):
        # DataBundle returns numpy; convert to torch here to keep towers torch-only
        import torch as _torch
        vec = self.base.vectorize_row(row)
        return _torch.as_tensor(vec, dtype=_torch.float32)

    def get_image_path(self, row):
        return self.base.get_image_path(row)

    def get_label(self, row):
        return int(row[self.base.label_col])


class HyperTower:


    def __init__(self, clinical: DataBundle, args):
        # Expects a fully built DataBundle (add_df handled upstream)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.clinical = clinical
        sample_mode = str(getattr(args, "sample_mode", "eye") or "eye").lower()
        tower_warmup = getattr(args, "warmup_tower_epochs", None)
        fused_warmup = getattr(args, "warmup_fused_epochs", None)
        if tower_warmup is None:
            tower_warmup = 8 if sample_mode == "patient" else 2
        if fused_warmup is None:
            fused_warmup = 5 if sample_mode == "patient" else 3
        self.warmup_tower_epochs = int(tower_warmup)
        self.warmup_fused_epochs = int(fused_warmup)
        self.args = args  # cache for checkpointing
        self._early = None
        self.aux_img = getattr(args, "aux_img", 0.05)
        self.aux_md  = getattr(args, "aux_md", 0.05)
        self.aux_detach = getattr(args, "aux_detach", True)
        # SE placement configuration
        se_where = getattr(args, "se_where", "bridge")  # 'bridge'|'tower'|'both'|'none'
        use_se_tower = se_where in ("tower", "both")
        # Optional disc-centric cropping
        self.image_preprocessor = None
        crop_manifest = getattr(args, "img_crop_manifest", None)
        crop_weights = getattr(args, "img_crop_weights", None)
        use_gt = getattr(args, "img_crop_gt", False)
        if crop_manifest:
            crop_cache = getattr(args, "img_crop_cache", Path("cache_data/hypertower_crops"))
            crop_cache = Path(crop_cache)
            persist_cache = bool(getattr(args, "persist_img_crop_cache", False))
            if use_gt:
                self.image_preprocessor = ManifestImageCropper(
                    manifest_path=Path(crop_manifest),
                    scale=getattr(args, "img_crop_scale", 2.5),
                    target_size=getattr(args, "img_crop_size", 224),
                    cache_dir=crop_cache,
                )
                if not persist_cache:
                    self.image_preprocessor.clear_cache()
                print(f"[HyperTower] GT disc cropper enabled → cache at {crop_cache}")
            elif crop_weights:
                self.image_preprocessor = UNetImageCropper(
                    manifest_path=Path(crop_manifest),
                    weights_path=Path(crop_weights),
                    normalize=getattr(args, "img_crop_normalize", "per_image"),
                    threshold=getattr(args, "img_crop_threshold", 0.5),
                    tta=getattr(args, "img_crop_tta", False),
                    scale=getattr(args, "img_crop_scale", 2.5),
                    target_size=getattr(args, "img_crop_size", 224),
                    cache_dir=crop_cache,
                )
                if not persist_cache:
                    self.image_preprocessor.clear_cache()
                print(f"[HyperTower] UNet disc cropper enabled → cache at {crop_cache}")
            else:
                print("[HyperTower] img_crop_manifest provided but no weights/gt flag; skipping cropping")

        self.use_geometry_features = bool(getattr(args, "img_geometry_features", False))
        if self.use_geometry_features and self.image_preprocessor is None:
            raise ValueError("img_geometry_features requires --img-crop-manifest with either --img-crop-weights or --img-crop-gt.")
        self.geometry_dim = FEATURE_DIM if self.use_geometry_features else 0

        # Towers derive their dimensions from DataBundle
        self.img_tower = ImageTower(
                backbone=getattr(args, "backbone", "efficientnet_b0"),
                freeze_ratio=getattr(args, "freeze_ratio", 0.0),
                use_se=use_se_tower,
                se_reduction=getattr(args, "se_reduction_tower", getattr(args, "se_reduction", 16)),
                se_pre_norm=getattr(args, "se_pre_norm_tower", getattr(args, "se_pre_norm", True)),
                augment=getattr(args, "img_augment", True),
                geometry_dim=self.geometry_dim,
            ).to(self.device)
        self.md_tower = MDTower(
                self.clinical,
                use_se=use_se_tower,
                se_reduction=getattr(args, "se_reduction_tower", getattr(args, "se_reduction", 16)),
                se_pre_norm=getattr(args, "se_pre_norm_tower", getattr(args, "se_pre_norm", True)),
            ).to(self.device)
        self.mode = args.fusion_mode  # cache
        # Training hyperparams / objects
        self.batch_size = args.batch_size
        # Main epochs exclude warmup; total training epochs adds warmup phases.
        self.epochs = int(args.epochs)
        self.total_epochs = int(self.warmup_tower_epochs + self.warmup_fused_epochs + self.epochs)
        print(
            f"[HyperTower] Warmup schedule: tower={self.warmup_tower_epochs}, "
            f"fused={self.warmup_fused_epochs}, main={self.epochs}, total={self.total_epochs}"
        )
        self.lr = args.lr
        self.fold = args.fold
        self.criterion = nn.CrossEntropyLoss()
        self._ce_weight = self.criterion.weight.detach().clone() if self.criterion.weight is not None else None
        if self._ce_weight is not None:
            self._ce_weight = self._ce_weight.to(self.device)
        self._ce_reduction = self.criterion.reduction
        self.focal_gamma = float(getattr(args, "focal_gamma", 0.0) or 0.0)
        # Run and model directories come from the script
        self.run_dir = Path(getattr(args, "run_dir", ".")).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # Per-run log locations to avoid collisions across concurrent workers
        self.epoch_log_path = self.run_dir / "epoch_log.csv"
        self.train_log_path = self.run_dir / "train.log"
        self.models_dir = Path(getattr(args, "models_dir", "models")).resolve()
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.holdout_df = getattr(args, "holdout_df", None)
        self.holdout_loader = None
        if isinstance(self.holdout_df, pd.DataFrame) and not self.holdout_df.empty:
            if getattr(args, "eval_mode", "multiclass") == "binary":
                label_col = getattr(self.clinical, "label_col", None)
                if label_col and label_col in self.holdout_df.columns:
                    self.holdout_df = self.holdout_df[self.holdout_df[label_col].isin([0, 1])].reset_index(drop=True)
            self.holdout_loader = self._make_loader_for_df(self.holdout_df, is_train=False)
            print(f"[HyperTower] Holdout loader prepared with {len(self.holdout_df)} samples")

        if self.mode == "vote":
            # Per-tower classification heads (logits) + vote combiner
            self.head_img = nn.Linear(self.img_tower.out_dim, args.num_classes).to(self.device)
            self.head_md  = nn.Linear(self.md_tower.out_dim, args.num_classes).to(self.device)
            self.vote     = VoteBridge(num_classes=args.num_classes).to(self.device)

            # Optimizer: towers + heads + vote
            self.optimizer = torch.optim.Adam(
                list(self.img_tower.parameters())
                + list(self.md_tower.parameters())
                + list(self.head_img.parameters())
                + list(self.head_md.parameters())
                + list(self.vote.parameters()),
                lr=self.lr,
            )
        else:
            # Existing feature-fusion bridge
            self.bridge = Bridge(
                img_dim=self.img_tower.out_dim,
                meta_dim=self.md_tower.out_dim,
                num_classes=args.num_classes,
                fusion_dim=256,
                mode=args.fusion_mode,
                use_se=(se_where in ("bridge","both")) and getattr(args, "use_se", True),
                se_reduction=(getattr(args, "se_reduction", 16) if getattr(args, "use_se", True) else 0),
                se_pre_norm=getattr(args, "se_pre_norm", True)
            ).to(self.device)

            self.optimizer = torch.optim.Adam(
                list(self.img_tower.parameters())
                + list(self.md_tower.parameters())
                + list(self.bridge.parameters()),
                lr=self.lr,
            )


        # EMA-forgiveness + BCD knobs; logging stays simple for now
        self.ema_alpha = args.ema_alpha
        self.bcd_prob = args.bcd_prob
        self.ema_fused_loss = None

        # Shared V2 logger for train.log + epoch_log.csv
        self.ht_logger = HypertowerLogger(
            run_dir=self.run_dir,
            train_log_path=self.train_log_path,
            epoch_log_path=self.epoch_log_path,
        )
        self.logger = self.ht_logger.logger

        # Dynamic BCD configuration (epoch baselines + batch nudges)
        self.bcd_cfg = {
            "metric": getattr(args, "bcd_metric", "auc"),
            "p0":     getattr(args, "bcd_p0", 0.20),
            "k":      getattr(args, "bcd_k", 0.4),
            "pmin":   getattr(args, "bcd_min", 0.05),
            "pmax":   getattr(args, "bcd_max", 0.30),
            "alpha_batch": getattr(args, "bcd_alpha_batch", 0.2),
            "alpha_tower": getattr(args, "bcd_alpha_tower", 0.3),
            "explore_floor": getattr(args, "bcd_explore_floor", 0.15),
            "entropy_ema": getattr(args, "entropy_ema", 0.7),
        }

        # Track last epoch's eval metrics for baseline deficits; initialize safely
        self.last_eval = {
            "acc_fused": 0.0, "acc_img": 0.0, "acc_md": 0.0,
            "auc_fused": 0.0, "auc_img": 0.0, "auc_md": 0.0,
        }
        # Running EMA of per-head uncertainty (entropy in [0,1])
        self.entropy_ema = {"fused": 0.5, "img": 0.5, "md": 0.5}

        # DataLoaders are (re)built each epoch from DataBundle's splits
        self.train_loader = None
        self.test_loader = None
        self._last_step_mix = None

        if not getattr(self.clinical, "folds", None):
            raise RuntimeError("DataBundle has no built folds. Did you call add_df(...) upstream?")

    @staticmethod
    def _confusion(preds, labels):
        # Quick TP/TN/FP/FN for binary debug (kept as-is)
        tp = ((preds == 1) & (labels == 1)).sum().item()
        tn = ((preds == 0) & (labels == 0)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()
        return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

    def _unpack_batch(self, batch):
        if len(batch) == 4:
            imgs, metas, geometry, labels = batch
        else:
            imgs, metas, labels = batch
            if self.geometry_dim > 0:
                geometry = torch.zeros(imgs.size(0), self.geometry_dim, dtype=torch.float32)
            else:
                geometry = None
        return imgs, metas, geometry, labels

    def _classification_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Shared classification loss (CrossEntropy or focal depending on gamma).
        """
        weight = self._ce_weight
        if weight is not None and weight.device != logits.device:
            weight = weight.to(logits.device)
        return focal_loss(
            logits,
            labels,
            gamma=self.focal_gamma,
            weight=weight,
            reduction=self._ce_reduction,
        )

    def _step_image_only(self, imgs, metas, geometry, labels):
        if self.mode == "vote":
            img_feats = self.img_tower(imgs, geometry)
            out_img   = self.head_img(img_feats)
            loss_i    = self._classification_loss(out_img, labels)
            preds_i   = out_img.argmax(dim=1)
            acc_i     = (preds_i == labels).float().mean().item()
            cm = self._confusion(preds_i, labels)

            self.optimizer.zero_grad()
            loss_i.backward()
            self.optimizer.step()
            return None, loss_i.item(), None, None, acc_i, None, {"img": cm}
        else:# Optimize the image head alone (used when BCD chooses image-only)
                img_feats = self.img_tower(imgs, geometry)
                out_img = self.bridge.classifier_img(img_feats)
                loss_i = self._classification_loss(out_img, labels)
                preds_i = out_img.argmax(dim=1)
                acc_i = (preds_i == labels).float().mean().item()
                cm = self._confusion(preds_i, labels)
                self.optimizer.zero_grad()
                loss_i.backward()
                self.optimizer.step()
                return None, loss_i.item(), None, None, acc_i, None, {"img": cm}

    def _step_meta_only(self, imgs, metas, geometry, labels):
        # Optimize the metadata head alone (used when BCD chooses meta-only)
        if self.mode == "vote":
            meta_feats = self.md_tower(metas)
            out_md     = self.head_md(meta_feats)
            loss_m     = self._classification_loss(out_md, labels)
            preds_m    = out_md.argmax(dim=1)
            acc_m      = (preds_m == labels).float().mean().item()
            cm = self._confusion(preds_m, labels)

            self.optimizer.zero_grad()
            loss_m.backward()
            self.optimizer.step()
            return None, None, loss_m.item(), None, None, acc_m, {"meta": cm}
        else:
            meta_feats = self.md_tower(metas)
            out_md = self.bridge.classifier_md(meta_feats)
            loss_m = self._classification_loss(out_md, labels)
            preds_m = out_md.argmax(dim=1)
            acc_m = (preds_m == labels).float().mean().item()
            cm = self._confusion(preds_m, labels)
            self.optimizer.zero_grad()
            loss_m.backward()
            self.optimizer.step()
            return None, None, loss_m.item(), None, None, acc_m, {"meta": cm}

    def _step_fused(self, imgs, metas, geometry, labels):
        if self.mode == "vote":
            img_feats = self.img_tower(imgs, geometry)
            md_feats  = self.md_tower(metas)

            out_img = self.head_img(img_feats)
            out_md  = self.head_md(md_feats)
            out_fused = self.vote(out_img, out_md)

            preds_f = out_fused.argmax(dim=1)
            acc_f   = (preds_f == labels).float().mean().item()
            loss_f  = self._classification_loss(out_fused, labels)
            loss_f_val = float(loss_f.detach().item())

            # Optional: keep your EMA + small aux tower losses just like before
            if self.ema_fused_loss is None:
                self.ema_fused_loss = loss_f_val
            L_total = (1.0 - self.ema_alpha) * loss_f + self.ema_alpha * loss_f.new_tensor(self.ema_fused_loss)

            aux_terms = 0.0
            if self.aux_img > 0:
                aux_terms = aux_terms + self.aux_img * self._classification_loss(
                    out_img.detach() if self.aux_detach else out_img,
                    labels,
                )
            if self.aux_md > 0:
                aux_terms = aux_terms + self.aux_md * self._classification_loss(
                    out_md.detach() if self.aux_detach else out_md,
                    labels,
                )
            L_total = L_total + aux_terms

            self.optimizer.zero_grad()
            L_total.backward()
            self.optimizer.step()

            self.ema_fused_loss = self.ema_alpha * self.ema_fused_loss + (1.0 - self.ema_alpha) * loss_f_val

            # Confusion tables for logging
            cm = {
                "fused": self._confusion(preds_f, labels),
                "img":   self._confusion(out_img.argmax(dim=1), labels),
                "meta":  self._confusion(out_md.argmax(dim=1), labels),
            }
            acc_i = (out_img.argmax(dim=1) == labels).float().mean().item()
            acc_m = (out_md.argmax(dim=1)  == labels).float().mean().item()

            with torch.no_grad():
                loss_img_val = self._classification_loss(out_img, labels).item()
                loss_md_val = self._classification_loss(out_md, labels).item()
            return loss_f_val, loss_img_val, loss_md_val, acc_f, acc_i, acc_m, cm
        else:
            img_feats = self.img_tower(imgs, geometry)
            meta_feats = self.md_tower(metas)
            out_fused, out_img, out_md = self.bridge(img_feats, meta_feats)

            preds_f = out_fused.argmax(dim=1)
            acc_f = (preds_f == labels).float().mean().item()
            loss_f = self._classification_loss(out_fused, labels)
            loss_f_val = float(loss_f.detach().item())

            cm = {"fused": self._confusion(preds_f, labels)}
            loss_i = acc_i = loss_m = acc_m = None

            if out_img is not None:
                with torch.no_grad():
                    preds_i = out_img.argmax(dim=1)
                    acc_i = (preds_i == labels).float().mean().item()
                    loss_i = self._classification_loss(out_img, labels).item()
                    cm["img"] = self._confusion(preds_i, labels)
            if out_md is not None:
                with torch.no_grad():
                    preds_m = out_md.argmax(dim=1)
                    acc_m = (preds_m == labels).float().mean().item()
                    loss_m = self._classification_loss(out_md, labels).item()
                    cm["meta"] = self._confusion(preds_m, labels)

            # EMA-blended fused loss
            if self.ema_fused_loss is None:
                self.ema_fused_loss = loss_f_val
            L_total = (1.0 - self.ema_alpha) * loss_f + self.ema_alpha * loss_f.new_tensor(self.ema_fused_loss)

            # Auxiliary tower losses (small weights). Use detached features to calibrate heads only.
            aux_terms = 0.0
            if self.aux_img > 0 and out_img is not None:
                logits_img_for_aux = self.bridge.classifier_img(img_feats.detach()) if self.aux_detach else out_img
                aux_terms = aux_terms + self.aux_img * self._classification_loss(logits_img_for_aux, labels)
            if self.aux_md > 0 and out_md is not None:
                logits_md_for_aux = self.bridge.classifier_md(meta_feats.detach()) if self.aux_detach else out_md
                aux_terms = aux_terms + self.aux_md * self._classification_loss(logits_md_for_aux, labels)

            L_total = L_total + aux_terms

            self.optimizer.zero_grad()
            L_total.backward()
            self.optimizer.step()

            # update EMA after the step
            self.ema_fused_loss = self.ema_alpha * self.ema_fused_loss + (1.0 - self.ema_alpha) * loss_f_val

            return loss_f_val, loss_i, loss_m, acc_f, acc_i, acc_m, cm


    def _make_loader_for_df(self, df, is_train: bool = False):
        # Rebuild a DataLoader for the current epoch's split
        view = _ClinicalView(self.clinical, df)
        geometry_provider = self.image_preprocessor if self.geometry_dim > 0 else None
        ds = ClinicalDataset(
            view,
            self.img_tower.transform,
            image_preprocessor=self.image_preprocessor,
            geometry_provider=geometry_provider,
            geometry_dim=self.geometry_dim,
        )

        # Optional: class-balanced bootstrapped sampling for TRAIN only
        if is_train and getattr(self.args, "balanced_sampler", False):
            import numpy as _np
            y = _np.asarray(df[self.clinical.label_col].values)
            # inverse-frequency weights per class
            uniq, counts = _np.unique(y, return_counts=True)
            inv = {c: (1.0 / cnt if cnt > 0 else 0.0) for c, cnt in zip(uniq, counts)}
            w = _np.array([inv[c] for c in y], dtype=_np.float32)
            sampler = WeightedRandomSampler(weights=w, num_samples=len(y), replacement=True)
            return DataLoader(ds, batch_size=self.batch_size, sampler=sampler, shuffle=False)

        return DataLoader(ds, batch_size=self.batch_size, shuffle=not is_train)

    # ---- Dynamic BCD helpers ----
    def _metric_value(self, name: str) -> float:
        # Pull either AUC or ACC from last_eval, falling back if NaN/zero
        if self.bcd_cfg["metric"] == "auc":
            v = self.last_eval.get(f"auc_{name}", 0.0)
            if v == v:  # not NaN
                return float(v)
            # fallback to accuracy
            return float(self.last_eval.get(f"acc_{name}", 0.0))
        return float(self.last_eval.get(f"acc_{name}", 0.0))

    def _epoch_bcd_baseline(self):
        # Compute epoch-level baselines: p_bcd_epoch and tower weights w_i, w_m
        p0 = self.bcd_cfg["p0"]; k = self.bcd_cfg["k"]
        pmin = self.bcd_cfg["pmin"]; pmax = self.bcd_cfg["pmax"]
        eps = 1e-6
        Af = self._metric_value("fused"); Ai = self._metric_value("img"); Am = self._metric_value("md")
        di = max(0.0, Af - Ai); dm = max(0.0, Af - Am)
        p_bcd_epoch = max(pmin, min(pmax, p0 + k * (di + dm) / 2.0))
        wi = (di + eps) / (di + dm + 2 * eps)
        wm = 1.0 - wi
        return p_bcd_epoch, wi, wm, {"di": di, "dm": dm, "Af": Af, "Ai": Ai, "Am": Am}

    @staticmethod
    def _entropy_from_logits(logits: torch.Tensor, num_classes: int) -> float:
        # Returns entropy normalized to [0,1] using log(K) denominator
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            ent = -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=1)
            ent = ent / np.log(num_classes)
            return float(ent.mean().item())

    def _batch_uncertainty(self, imgs: torch.Tensor, metas: torch.Tensor, geometry: Optional[torch.Tensor] = None) -> dict:
        self.img_tower.eval(); self.md_tower.eval()
        if self.mode == "vote":
            self.head_img.eval(); self.head_md.eval(); self.vote.eval()
            with torch.no_grad():
                if geometry is None and self.geometry_dim > 0:
                    geometry = torch.zeros(imgs.size(0), self.geometry_dim, device=imgs.device, dtype=imgs.dtype)
                img_feats = self.img_tower(imgs, geometry)
                md_feats  = self.md_tower(metas)
                out_img   = self.head_img(img_feats)
                out_md    = self.head_md(md_feats)
                out_fused = self.vote(out_img, out_md)
                K = out_fused.shape[1]
                e_f = self._entropy_from_logits(out_fused, K)
                e_i = self._entropy_from_logits(out_img,   K)
                e_m = self._entropy_from_logits(out_md,    K)
            # restore train()
            self.img_tower.train(); self.md_tower.train()
            self.head_img.train(); self.head_md.train(); self.vote.train()
        else:
            self.bridge.eval()
            with torch.no_grad():
                if geometry is None and self.geometry_dim > 0:
                    geometry = torch.zeros(imgs.size(0), self.geometry_dim, device=imgs.device, dtype=imgs.dtype)
                img_feats = self.img_tower(imgs, geometry)
                meta_feats = self.md_tower(metas)
                out_fused, out_img, out_md = self.bridge(img_feats, meta_feats)
                K = out_fused.shape[1]
                e_f = self._entropy_from_logits(out_fused, K)
                e_i = self._entropy_from_logits(out_img, K) if out_img is not None else 0.5
                e_m = self._entropy_from_logits(out_md, K) if out_md is not None else 0.5
            self.img_tower.train(); self.md_tower.train(); self.bridge.train()

        a = self.bcd_cfg["entropy_ema"]
        self.entropy_ema["fused"] = a * self.entropy_ema["fused"] + (1 - a) * e_f
        self.entropy_ema["img"]   = a * self.entropy_ema["img"]   + (1 - a) * e_i
        self.entropy_ema["md"]    = a * self.entropy_ema["md"]    + (1 - a) * e_m
        return {"fused": self.entropy_ema["fused"], "img": self.entropy_ema["img"], "md": self.entropy_ema["md"]}

    def _write_epoch_log(self, row: dict, path: str | Path | None = None):
        self.ht_logger.write_epoch_row(row=row, path=path)

    def _snapshot(self):
        state = {
            "img_tower": self.img_tower.state_dict(),
            "md_tower":  self.md_tower.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "args": vars(self.args),
        }
        if hasattr(self, "bridge"):   state["bridge"]  = self.bridge.state_dict()
        if hasattr(self, "head_img"): state["head_img"] = self.head_img.state_dict()
        if hasattr(self, "head_md"):  state["head_md"]  = self.head_md.state_dict()
        return state


    def train(self):
        best_path = None
        if getattr(self.args, "checkpoint_best", False):
            best_path = str((self.models_dir / "best.pth").resolve())
        
        # Always track the best model even if we don't stop early
        monitor = getattr(self.args, "early_metric", None)
        if not monitor:
            # Prefer AUC-based monitoring by default; fall back to loss only if specified
            if self.mode == "image_only":
                monitor = "auc_img"
            elif self.mode == "metadata_only":
                monitor = "auc_md"
            else:  # fused or vote
                monitor = "auc_fused"
        mode = getattr(self.args, "early_mode", "auto")
        # Optional switch: monitor holdout metrics instead of validation
        use_holdout_monitor = bool(getattr(self.args, "early_monitor_holdout", False))
        if use_holdout_monitor and self.holdout_loader is None:
            print("[early] Requested holdout monitoring but no holdout set is configured; falling back to validation metrics.")
            use_holdout_monitor = False
        if use_holdout_monitor:
            # If the monitor isn't already a holdout metric, prepend it
            if not monitor.startswith("holdout_"):
                monitor = f"holdout_{monitor}"
            # Force sensible default if no monitor was provided
            if monitor == "holdout_auc_fused" and mode == "auto":
                mode = "max"
        if mode == "auto":
            mode = ("min" if ("loss" in monitor.lower()) else "max")
        self._best_info = {
            "monitor": monitor,
            "mode": mode,
            "min_delta": float(getattr(self.args, "early_min_delta", 0.0)),
            "value": (-math.inf if mode == "max" else math.inf),
            "epoch": -1,
            "state": None,
            "save_path": best_path,
        }
        # keep args in sync so EarlyStopper uses the same monitor/mode
        self.args.early_metric = monitor
        self.args.early_mode = mode
        # Separate tracker for best holdout performance (if a holdout set is provided)
        self._holdout_best = None
        if self.holdout_loader is not None:
            self._holdout_best = {
                "monitor": "holdout_auc_fused",
                "mode": "max",
                "min_delta": 0.0,
                "value": -math.inf,
                "epoch": -1,
                "state": None,
                "save_path": str((self.models_dir / "model_holdout_best.pt").resolve()),
            }

        self._early = None
        if getattr(self.args, "early_stop", False):
            self._early = EarlyStopper(
                monitor=monitor,
                mode=mode,
                patience=self.args.early_patience,
                min_delta=self.args.early_min_delta,
                save_path=best_path,
                restore_best=True,
            )
        # Gradual thaw config
        gradual = bool(getattr(self.args, "gradual_thaw", False))
        thaw_ratio = float(getattr(self.args, "thaw_ratio", 0.33))
        thaw_phase = int(getattr(self.args, "thaw_phase_duration", 5))
        thaw_target = str(getattr(self.args, "thaw_target", "image"))
        thaw_start_epoch = int(getattr(self.args, "thaw_start_epoch", -1))
        thaw_initial_freeze = bool(getattr(self.args, "thaw_initial_freeze", False))
        if thaw_start_epoch < 0:
            thaw_start_epoch = int(getattr(self, "warmup_tower_epochs", 0))

        for epoch in range(self.total_epochs):
            # Apply thaw schedule at epoch start: start fully frozen and unfreeze by ratios
            if gradual and thaw_phase > 0:
                try:
                    if epoch < thaw_start_epoch:
                        if thaw_initial_freeze:
                            freeze_r = 1.0
                            if thaw_target in ("image", "both") and hasattr(self, "img_tower"):
                                self.img_tower.set_freeze_ratio(freeze_r)
                            if thaw_target in ("metadata", "both") and hasattr(self, "md_tower"):
                                self.md_tower.set_freeze_ratio(freeze_r)
                        # else: leave current requires_grad as constructed (no forced freeze)
                    else:
                        phase_idx = (epoch - thaw_start_epoch) // thaw_phase
                        freeze_r = max(0.0, 1.0 - phase_idx * thaw_ratio)
                        if thaw_target in ("image", "both") and hasattr(self, "img_tower"):
                            self.img_tower.set_freeze_ratio(freeze_r)
                        if thaw_target in ("metadata", "both") and hasattr(self, "md_tower"):
                            self.md_tower.set_freeze_ratio(freeze_r)
                except Exception:
                    pass
            # build splits/loaders per-epoch
            train_df, test_df = self.clinical.get_split_dfs(self.fold)
            if hasattr(self.bridge, "reset_se_stats"):
                self.bridge.reset_se_stats()
            if getattr(self, "args", None) is None:
                self.args = argparse.Namespace()  # if not stashed already
            # Expect user to pass --eval_mode and --num-classes==2 for binary
            if getattr(self.args, "eval_mode", "multiclass") == "binary":
                # Drop Suspect (assumed label==2 in PAPILA spreadsheets)
                train_df = train_df[train_df[self.clinical.label_col].isin([0,1])].copy()
                test_df  = test_df [test_df [self.clinical.label_col].isin([0,1])].copy()
            self.train_loader = self._make_loader_for_df(train_df, is_train=True)
            self.test_loader  = self._make_loader_for_df(test_df, is_train=False)

            self.img_tower.train(); self.md_tower.train(); self.bridge.train()

            total_losses = {"fused":0.0,"image_only":0.0,"metadata_only":0.0}
            total_accs   = {"fused":0.0,"image_only":0.0,"metadata_only":0.0}
            counts       = {"fused":0,   "image_only":0,   "metadata_only":0}
            step_counts  = {"fused":0,   "image_only":0,   "metadata_only":0}

            # phase selection
            in_tower_warmup = epoch < self.warmup_tower_epochs
            in_fused_warmup = (self.warmup_tower_epochs <= epoch < (self.warmup_tower_epochs + self.warmup_fused_epochs))

            # epoch baselines (needed for splits and logging)
            p_bcd_epoch, wi_epoch, wm_epoch, diag = self._epoch_bcd_baseline()

            if in_tower_warmup:
                # tower-only warmup: alternate or use epoch split; default to 0.5 if empty
                wi = wi_epoch if (diag["Ai"] or diag["Am"]) else 0.5
                for batch_idx, batch in enumerate(self.train_loader):
                    imgs, metas, geometry, labels = self._unpack_batch(batch)
                    imgs = imgs.to(self.device)
                    metas = metas.to(self.device)
                    labels = labels.to(self.device)
                    geometry = geometry.to(self.device) if geometry is not None else None
                    if random() < wi:
                        step_type = "image"
                        loss_f, loss_i, loss_m, acc_f, acc_i, acc_m, cm = self._step_image_only(imgs, metas, geometry, labels)
                        step_counts["image_only"] += 1
                    else:
                        step_type = "meta"
                        loss_f, loss_i, loss_m, acc_f, acc_i, acc_m, cm = self._step_meta_only(imgs, metas, geometry, labels)
                        step_counts["metadata_only"] += 1

                    self.logger.info(f"epoch={epoch} batch={batch_idx} phase=tower_warmup step={step_type} "
                                    f"loss_f={loss_f} loss_i={loss_i} loss_m={loss_m} acc_f={acc_f} acc_i={acc_i} acc_m={acc_m} cm={cm}")

                    if loss_i is not None:
                        total_losses["image_only"] += loss_i; total_accs["image_only"] += acc_i; counts["image_only"] += 1
                    if loss_m is not None:
                        total_losses["metadata_only"] += loss_m; total_accs["metadata_only"] += acc_m; counts["metadata_only"] += 1

            elif in_fused_warmup:
                # fused-only warmup
                for batch_idx, batch in enumerate(self.train_loader):
                    imgs, metas, geometry, labels = self._unpack_batch(batch)
                    imgs = imgs.to(self.device)
                    metas = metas.to(self.device)
                    labels = labels.to(self.device)
                    geometry = geometry.to(self.device) if geometry is not None else None
                    step_type = "fused"
                    loss_f, loss_i, loss_m, acc_f, acc_i, acc_m, cm = self._step_fused(imgs, metas, geometry, labels)
                    step_counts["fused"] += 1

                    self.logger.info(f"epoch={epoch} batch={batch_idx} phase=fused_warmup step={step_type} "
                                    f"loss_f={loss_f} acc_f={acc_f} cm={cm}")

                    total_losses["fused"] += loss_f; total_accs["fused"] += acc_f; counts["fused"] += 1

            else:
                # dynamic BCD (fusion-centric)
                self.logger.info(
                    f"epoch={epoch} bcd_epoch={p_bcd_epoch:.3f} wi_epoch={wi_epoch:.3f} wm_epoch={wm_epoch:.3f} "
                    f"deficits={{img:{diag['di']:.3f}, md:{diag['dm']:.3f}}} metrics={{Af:{diag['Af']:.3f}, Ai:{diag['Ai']:.3f}, Am:{diag['Am']:.3f}}}"
                )
                for batch_idx, batch in enumerate(self.train_loader):
                    imgs, metas, geometry, labels = self._unpack_batch(batch)
                    imgs = imgs.to(self.device)
                    metas = metas.to(self.device)
                    labels = labels.to(self.device)
                    geometry = geometry.to(self.device) if geometry is not None else None
                    ents = self._batch_uncertainty(imgs, metas, geometry)

                    # flip rule: higher fused entropy => fewer tower-only batches => more fused training
                    alpha_b = self.bcd_cfg["alpha_batch"]; pmin = self.bcd_cfg["pmin"]; pmax = self.bcd_cfg["pmax"]
                    p_bcd_batch = (1 - alpha_b) * p_bcd_epoch + alpha_b * (1.0 - ents["fused"])
                    p_bcd_batch = max(pmin, min(pmax, p_bcd_batch))

                    # image vs meta split
                    alpha_t = self.bcd_cfg["alpha_tower"]; floor = self.bcd_cfg["explore_floor"]; eps = 1e-6
                    s_img = ents["img"] / (ents["img"] + ents["md"] + eps)
                    p_img_batch = (1 - alpha_t) * wi_epoch + alpha_t * s_img
                    p_img_batch = max(floor, min(1.0 - floor, p_img_batch))

                    u = random(); v = random()
                    if u < p_bcd_batch:
                        if v < p_img_batch:
                            step_type = "image"
                            loss_f, loss_i, loss_m, acc_f, acc_i, acc_m, cm = self._step_image_only(imgs, metas, geometry, labels)
                            step_counts["image_only"] += 1
                        else:
                            step_type = "meta"
                            loss_f, loss_i, loss_m, acc_f, acc_i, acc_m, cm = self._step_meta_only(imgs, metas, geometry, labels)
                            step_counts["metadata_only"] += 1
                    else:
                        step_type = "fused"
                        loss_f, loss_i, loss_m, acc_f, acc_i, acc_m, cm = self._step_fused(imgs, metas, geometry, labels)
                        step_counts["fused"] += 1

                    self.logger.info(
                        f"epoch={epoch} batch={batch_idx} phase=dynamic step={step_type} "
                        f"p_bcd_batch={p_bcd_batch:.3f} p_img_batch={p_img_batch:.3f} "
                        f"ents={{f:{ents['fused']:.3f}, i:{ents['img']:.3f}, m:{ents['md']:.3f}}} "
                        f"loss_f={loss_f} loss_i={loss_i} loss_m={loss_m} acc_f={acc_f} acc_i={acc_i} acc_m={acc_m} cm={cm}"
                    )

                    if loss_f is not None:
                        total_losses["fused"] += loss_f; total_accs["fused"] += acc_f; counts["fused"] += 1
                    if loss_i is not None:
                        total_losses["image_only"] += loss_i; total_accs["image_only"] += acc_i; counts["image_only"] += 1
                    if loss_m is not None:
                        total_losses["metadata_only"] += loss_m; total_accs["metadata_only"] += acc_m; counts["metadata_only"] += 1

            # epoch prints
            avg_loss = total_losses["fused"] / counts["fused"] if counts["fused"] else 0.0
            avg_fused = total_accs["fused"] / counts["fused"] if counts["fused"] else 0.0
            avg_img   = total_accs["image_only"] / counts["image_only"] if counts["image_only"] else 0.0
            avg_md    = total_accs["metadata_only"] / counts["metadata_only"] if counts["metadata_only"] else 0.0

            # realized mix for logging
            tot_steps = sum(step_counts.values()) or 1
            self._last_step_mix = {
                "pct_fused": step_counts["fused"] / tot_steps,
                "pct_img":   step_counts["image_only"] / tot_steps,
                "pct_md":    step_counts["metadata_only"] / tot_steps,
                "phase":     "tower_warmup" if in_tower_warmup else ("fused_warmup" if in_fused_warmup else "dynamic")
            }

            print(
                f"[Epoch {epoch+1}/{self.total_epochs}] Train Loss: {avg_loss:.4f} | "
                f"Fused Acc: {100*avg_fused:.2f}% | Img Acc: {100*avg_img:.2f}% | Md Acc: {100*avg_md:.2f}%"
            )
            row, stop_now = self.evaluate(epoch)
            if stop_now:
                print(f"[early] stopping on '{self.args.early_metric}' "
                      f"with best={self._early.best:.5f} at epoch {epoch+1 - self._early.bad_epochs}")
                break

        torch.save(self._snapshot(), str(self.models_dir / "model_last.pt"))
        # Ensure holdout-best checkpoint is written
        if getattr(self, "_holdout_best", None):
            hb = self._holdout_best
            if hb.get("state") is not None and hb.get("save_path"):
                try:
                    torch.save(hb["state"], hb["save_path"])
                except Exception:
                    pass

        def _copy_best_roc(epoch_idx: int, tag: str):
            if epoch_idx is None or epoch_idx < 0:
                return
            src_dir = self.run_dir / "roc_curves"
            if not src_dir.exists():
                return
            prefix = f"epoch{epoch_idx + 1}_"
            dest_dir = self.run_dir / f"roc_curves_{tag}"
            dest_dir.mkdir(parents=True, exist_ok=True)
            for path in src_dir.glob(f"{prefix}*.json"):
                try:
                    shutil.copy2(path, dest_dir / path.name)
                except Exception:
                    pass
        if getattr(self, "_early", None):
            print(f"[early] restoring best model at epoch {self._early.best_epoch+1}: {self._early.best:.5f}")
            self._early.restore(self)
        elif getattr(self, "_best_info", None) and self._best_info.get("state") is not None:
            be = self._best_info
            print(f"[best] restoring best model at epoch {be['epoch']+1}: {be['value']:.5f} (monitor={be['monitor']})")
            self._restore_from_state(be["state"])
        torch.save(self._snapshot(), str(self.models_dir / "model_best.pt"))
        # Snapshot ROC curves for both main-best and holdout-best epochs
        if getattr(self, "_early", None) and getattr(self._early, "best_epoch", None) is not None:
            _copy_best_roc(self._early.best_epoch, "best")
        elif getattr(self, "_best_info", None):
            _copy_best_roc(self._best_info.get("epoch", -1), "best")
        if getattr(self, "_holdout_best", None):
            _copy_best_roc(self._holdout_best.get("epoch", -1), "holdout_best")
        self.ht_logger.close()



    def _evaluate_split(self, loader, epoch: int, split: str):
        """
        Shared evaluation helper that computes core metrics (loss/accuracy/AUC/etc.)
        for a given dataloader. Returns a dict with scalar metrics or None if the
        loader is empty.
        """
        if loader is None:
            return None

        total_loss = 0.0
        total_counts = {"fused": 0, "image_only": 0, "metadata_only": 0}
        total_accs = {"fused": 0.0, "image_only": 0.0, "metadata_only": 0.0}
        all_y = []
        probs_f_list, probs_i_list, probs_m_list = [], [], []
        agree_f_img = 0
        agree_f_md = 0
        n_img_comp = 0
        n_md_comp = 0

        with torch.no_grad():
            for batch in loader:
                imgs, metas, geometry, labels = self._unpack_batch(batch)
                imgs = imgs.to(self.device)
                metas = metas.to(self.device)
                labels = labels.to(self.device)
                geometry = geometry.to(self.device) if geometry is not None else None

                if self.mode == "vote":
                    img_feats = self.img_tower(imgs, geometry)
                    md_feats = self.md_tower(metas)
                    out_img = self.head_img(img_feats)
                    out_md = self.head_md(md_feats)
                    out_fused = self.vote(out_img, out_md)
                else:
                    img_feats = self.img_tower(imgs, geometry)
                    md_feats = self.md_tower(metas)
                    outputs = self.bridge(img_feats, md_feats)
                    if isinstance(outputs, tuple):
                        out_fused, out_img, out_md = outputs
                    else:
                        out_fused, out_img, out_md = outputs, None, None

                all_y.append(labels.cpu().numpy())
                probs_f_list.append(F.softmax(out_fused, dim=1).cpu().numpy())

                loss = self._classification_loss(out_fused, labels)
                total_loss += loss.item()

                preds_f = out_fused.argmax(dim=1)
                total_accs["fused"] += (preds_f == labels).float().mean().item()
                total_counts["fused"] += 1

                if out_img is not None:
                    preds_i = out_img.argmax(dim=1)
                    total_accs["image_only"] += (preds_i == labels).float().mean().item()
                    total_counts["image_only"] += 1
                    probs_i_list.append(F.softmax(out_img, dim=1).cpu().numpy())

                    agree_f_img += (preds_f == preds_i).sum().item()
                    n_img_comp += preds_f.numel()

                if out_md is not None:
                    preds_m = out_md.argmax(dim=1)
                    total_accs["metadata_only"] += (preds_m == labels).float().mean().item()
                    total_counts["metadata_only"] += 1
                    probs_m_list.append(F.softmax(out_md, dim=1).cpu().numpy())

                    agree_f_md += (preds_f == preds_m).sum().item()
                    n_md_comp += preds_f.numel()

        if total_counts["fused"] == 0:
            return None

        avg_loss = total_loss / total_counts["fused"]
        avg_fused = total_accs["fused"] / total_counts["fused"]
        avg_img = (total_accs["image_only"] / total_counts["image_only"]) if total_counts["image_only"] else 0.0
        avg_md = (total_accs["metadata_only"] / total_counts["metadata_only"]) if total_counts["metadata_only"] else 0.0

        y_true = np.concatenate(all_y, axis=0) if all_y else np.array([])
        y_prob_f = np.concatenate(probs_f_list, axis=0) if probs_f_list else None
        y_prob_i = np.concatenate(probs_i_list, axis=0) if probs_i_list else None
        y_prob_m = np.concatenate(probs_m_list, axis=0) if probs_m_list else None

        def compute_multiclass_roc(y_true_np, prob_np):
            if prob_np is None or prob_np.size == 0:
                return None
            C = int(prob_np.shape[1])
            classes = list(range(C))
            y_bin = label_binarize(y_true_np, classes=classes)
            per_class = {}
            aucs = []
            for c in range(C):
                try:
                    fpr, tpr, _ = roc_curve(y_bin[:, c], prob_np[:, c])
                    auc_val = float(auc(fpr, tpr)) if len(fpr) > 1 else float("nan")
                    per_class[c] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc_val}
                    aucs.append(auc_val)
                except Exception:
                    per_class[c] = {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "auc": float("nan")}

            try:
                fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), prob_np[:, :C].ravel())
                auc_micro = float(auc(fpr_micro, tpr_micro))
                micro = {"fpr": fpr_micro.tolist(), "tpr": tpr_micro.tolist(), "auc": auc_micro}
            except Exception:
                micro = None

            auc_vals = [v["auc"] for v in per_class.values() if v["auc"] == v["auc"]]
            macro_auc = float(np.mean(auc_vals)) if auc_vals else float("nan")
            return {"per_class": per_class, "micro": micro, "macro_auc": macro_auc}

        def macro_ovr_auc(y, p):
            try:
                y = np.asarray(y)
                if p is None:
                    return float("nan")
                p = np.asarray(p)
                if p.ndim == 2 and p.shape[1] == 2:
                    return roc_auc_score(y, p[:, 1])
                if p.ndim == 1 or p.shape[1] == 1:
                    return roc_auc_score(y, p.ravel())
                return roc_auc_score(y, p, multi_class="ovr", average="macro")
            except Exception:
                return float("nan")

        def top2_acc(y, p):
            if p is None:
                return float("nan")
            if p.ndim != 2:
                return float("nan")
            k = 2 if p.shape[1] >= 2 else 1
            topk = np.argpartition(-p, kth=k - 1, axis=1)[:, :k]
            return np.mean((topk == y[:, None]).any(axis=1).astype(np.float32))

        def avg_margin(p):
            if p is None or p.ndim != 2:
                return float("nan")
            s = np.sort(p, axis=1)[:, ::-1]
            if s.shape[1] == 1:
                return float("nan")
            return float(np.mean(s[:, 0] - s[:, 1]))

        roc_f = compute_multiclass_roc(y_true, y_prob_f)
        roc_i = compute_multiclass_roc(y_true, y_prob_i)
        roc_m = compute_multiclass_roc(y_true, y_prob_m)

        roc_dir = self.run_dir / "roc_curves"
        roc_dir.mkdir(parents=True, exist_ok=True)

        def dump_roc(blob, head: str):
            if blob is None:
                return
            suffix = head if split == "val" else f"{split}_{head}"
            path = roc_dir / f"epoch{epoch + 1}_{suffix}.json"
            with open(path, "w") as f:
                json.dump(blob, f)

        dump_roc(roc_f, "fused")
        dump_roc(roc_i, "image")
        dump_roc(roc_m, "metadata")

        auc_f = macro_ovr_auc(y_true, y_prob_f) if y_prob_f is not None else float("nan")
        auc_i = macro_ovr_auc(y_true, y_prob_i) if y_prob_i is not None else float("nan")
        auc_m = macro_ovr_auc(y_true, y_prob_m) if y_prob_m is not None else float("nan")

        top2_f = top2_acc(y_true, y_prob_f)
        top2_i = top2_acc(y_true, y_prob_i)
        top2_m = top2_acc(y_true, y_prob_m)
        mar_f = avg_margin(y_prob_f)
        mar_i = avg_margin(y_prob_i)
        mar_m = avg_margin(y_prob_m)

        agree_rate_f_img = (agree_f_img / n_img_comp) if n_img_comp else float("nan")
        agree_rate_f_md = (agree_f_md / n_md_comp) if n_md_comp else float("nan")

        return {
            "loss": float(avg_loss),
            "acc_fused": float(avg_fused),
            "acc_img": float(avg_img),
            "acc_md": float(avg_md),
            "auc_fused": float(auc_f) if auc_f == auc_f else float("nan"),
            "auc_img": float(auc_i) if auc_i == auc_i else float("nan"),
            "auc_md": float(auc_m) if auc_m == auc_m else float("nan"),
            "top2_fused": float(top2_f) if top2_f == top2_f else float("nan"),
            "top2_img": float(top2_i) if top2_i == top2_i else float("nan"),
            "top2_md": float(top2_m) if top2_m == top2_m else float("nan"),
            "margin_fused": float(mar_f) if mar_f == mar_f else float("nan"),
            "margin_img": float(mar_i) if mar_i == mar_i else float("nan"),
            "margin_md": float(mar_m) if mar_m == mar_m else float("nan"),
            "agree_fused_img": float(agree_rate_f_img) if agree_rate_f_img == agree_rate_f_img else float("nan"),
            "agree_fused_md": float(agree_rate_f_md) if agree_rate_f_md == agree_rate_f_md else float("nan"),
        }


    def evaluate(self, epoch: int):
        # Evaluation additionally collects per-head probabilities to report macro AUC
        self.img_tower.eval()
        self.md_tower.eval()
        if self.mode == "vote":
            self.head_img.eval(); self.head_md.eval(); self.vote.eval()
        else:
            self.bridge.eval()

        metrics_val = self._evaluate_split(self.test_loader, epoch, "val")
        metrics_holdout = self._evaluate_split(self.holdout_loader, epoch, "holdout") if self.holdout_loader else None

        if metrics_val is None:
            raise RuntimeError("Validation loader produced no batches; cannot compute metrics.")

        def ffmt(x):
            return "NA" if (x != x) else f"{x:.3f}"

        print(
            f"[Epoch {epoch+1}/{self.total_epochs}] Eval  Loss: {metrics_val['loss']:.4f} | "
            f"Fused Acc: {100*metrics_val['acc_fused']:.2f}% | "
            f"Img Acc: {100*metrics_val['acc_img']:.2f}% | "
            f"Md Acc: {100*metrics_val['acc_md']:.2f}% | "
            f"Fused AUC: {ffmt(metrics_val['auc_fused'])} | "
            f"Img AUC: {ffmt(metrics_val['auc_img'])} | "
            f"Md AUC: {ffmt(metrics_val['auc_md'])}"
        )

        if metrics_holdout:
            print(
                f"[Epoch {epoch+1}/{self.total_epochs}] Holdout Loss: {metrics_holdout['loss']:.4f} | "
                f"Fused Acc: {100*metrics_holdout['acc_fused']:.2f}% | "
                f"Img Acc: {100*metrics_holdout['acc_img']:.2f}% | "
                f"Md Acc: {100*metrics_holdout['acc_md']:.2f}% | "
                f"Fused AUC: {ffmt(metrics_holdout['auc_fused'])} | "
                f"Img AUC: {ffmt(metrics_holdout['auc_img'])} | "
                f"Md AUC: {ffmt(metrics_holdout['auc_md'])}"
            )

        self.last_eval = {
            "acc_fused": float(metrics_val["acc_fused"]),
            "acc_img": float(metrics_val["acc_img"]),
            "acc_md": float(metrics_val["acc_md"]),
            "auc_fused": float(metrics_val["auc_fused"]) if metrics_val["auc_fused"] == metrics_val["auc_fused"] else 0.0,
            "auc_img": float(metrics_val["auc_img"]) if metrics_val["auc_img"] == metrics_val["auc_img"] else 0.0,
            "auc_md": float(metrics_val["auc_md"]) if metrics_val["auc_md"] == metrics_val["auc_md"] else 0.0,
        }

        row = {
            "epoch": int(epoch + 1),
            "eval_loss": _num(metrics_val["loss"]),
            "acc_fused": _num(metrics_val["acc_fused"]),
            "acc_img": _num(metrics_val["acc_img"]),
            "acc_md": _num(metrics_val["acc_md"]),
            "auc_fused": _num(metrics_val["auc_fused"]),
            "auc_img": _num(metrics_val["auc_img"]),
            "auc_md": _num(metrics_val["auc_md"]),
            "top2_fused": _num(metrics_val["top2_fused"]),
            "top2_img": _num(metrics_val["top2_img"]),
            "top2_md": _num(metrics_val["top2_md"]),
            "margin_fused": _num(metrics_val["margin_fused"]),
            "margin_img": _num(metrics_val["margin_img"]),
            "margin_md": _num(metrics_val["margin_md"]),
            "agree_fused_img": _num(metrics_val["agree_fused_img"]),
            "agree_fused_md": _num(metrics_val["agree_fused_md"]),
        }

        if metrics_holdout:
            row.update({
                "holdout_loss": _num(metrics_holdout["loss"]),
                "holdout_acc_fused": _num(metrics_holdout["acc_fused"]),
                "holdout_acc_img": _num(metrics_holdout["acc_img"]),
                "holdout_acc_md": _num(metrics_holdout["acc_md"]),
                "holdout_auc_fused": _num(metrics_holdout["auc_fused"]),
                "holdout_auc_img": _num(metrics_holdout["auc_img"]),
                "holdout_auc_md": _num(metrics_holdout["auc_md"]),
            })
        else:
            row.setdefault("holdout_loss", None)
            row.setdefault("holdout_acc_fused", None)
            row.setdefault("holdout_acc_img", None)
            row.setdefault("holdout_acc_md", None)
            row.setdefault("holdout_auc_fused", None)
            row.setdefault("holdout_auc_img", None)
            row.setdefault("holdout_auc_md", None)

        # realized step mix from training (if present)
        mix = getattr(self, "_last_step_mix", None)
        if mix:
            row.update({
                "pct_fused": _num(mix.get("pct_fused")),
                "pct_img":   _num(mix.get("pct_img")),
                "pct_md":    _num(mix.get("pct_md")),
                "phase":     mix.get("phase"),
            })

        # SE gate stats (if the bridge exposes them)
        # fetch without clearing
        bridge_module = getattr(self, "bridge", None)
        get_se_stats = getattr(bridge_module, "get_se_stats", None) if bridge_module is not None else None
        if callable(get_se_stats):
            se_stats = get_se_stats(reset=False)
            if se_stats:
                row.update({
                    "se_mean":        _num(se_stats.get("mean")),
                    "se_std":         _num(se_stats.get("std")),
                    "se_pct_lt_0.2":  _num(se_stats.get("pct_lt_0.2")),
                    "se_pct_gt_0.8":  _num(se_stats.get("pct_gt_0.8")),
                })
                # debug print BEFORE reset so you can see the true count
                try:
                    print(f"SE gate samples seen: {getattr(getattr(bridge_module, 'se_log', None), '_n', 0)}")
                except Exception:
                    pass
            # now clear for the next epoch
            try:
                get_se_stats(reset=True)
            except Exception:
                pass
        row.setdefault(self.args.early_metric, None)
        # early-stop decision (no break here)
        should_stop = False
        if getattr(self, "_early", None):
            should_stop = self._early.step(row, trainer=self, epoch=epoch)
            row.update({
                "early_best_so_far" : float(self._early.best),
                "early_bad_epochs" : int(self._early.bad_epochs),
                "early_improved" : int(self._early.last_improved),
                "early_monitor" : self.args.early_metric,
            })
            # concise console status each epoch
            mon = self.args.early_metric
            cur = row.get(mon, None)
            if cur is not None and cur == cur:  # not NaN
                msg = (
                    f"[early] epoch {epoch+1}: {mon}={cur:.5f} | best={self._early.best:.5f} | "
                    f"bad_epochs={self._early.bad_epochs}/{self._early.patience} | improved={'yes' if self._early.last_improved else 'no'}"
                )
                print(msg)
                try:
                    self.logger.info(msg)
                except Exception:
                    pass

        # Holdout best-tracker (if a holdout set is present)
        if getattr(self, "_holdout_best", None):
            mon_h = self._holdout_best["monitor"]
            mode_h = self._holdout_best["mode"]
            min_delta_h = self._holdout_best["min_delta"]
            val_h = row.get(mon_h, None)
            improved_h = False
            if val_h is not None and val_h == val_h:
                best_val_h = self._holdout_best["value"]
                if mode_h == "max":
                    improved_h = (val_h > best_val_h + min_delta_h)
                else:
                    improved_h = (val_h < best_val_h - min_delta_h)
            if improved_h:
                self._holdout_best["value"] = float(val_h)
                self._holdout_best["epoch"] = int(epoch)
                st_h = self._snapshot()
                self._holdout_best["state"] = st_h
                if self._holdout_best.get("save_path"):
                    try:
                        torch.save(st_h, self._holdout_best["save_path"])
                    except Exception:
                        pass
                print(f"[holdout_best] ↑ new best {mon_h}={val_h:.5f} at epoch {epoch+1}")
            row.update({
                "holdout_best_monitor": mon_h,
                "holdout_best_so_far": (None if self._holdout_best["value"] in (math.inf, -math.inf) else float(self._holdout_best["value"])),
                "holdout_best_epoch": (None if self._holdout_best["epoch"] < 0 else int(self._holdout_best["epoch"] + 1)),
            })

        # Best-tracker (always-on): save best model snapshot and optional checkpoint
        if getattr(self, "_best_info", None):
            # If early stopper is active, mirror its best info and avoid duplicate prints
            if getattr(self, "_early", None):
                row.update({
                    "best_monitor": self._best_info["monitor"],
                    "best_so_far": float(self._early.best) if self._early.best == self._early.best else None,
                    "best_epoch": (int(self._early.best_epoch) + 1) if (self._early.best_epoch is not None) else None,
                })
                # skip independent tracking/prints to avoid duplication
                self._write_epoch_log(row)
                return row, should_stop
            mon = self._best_info["monitor"]
            mode = self._best_info["mode"]
            min_delta = self._best_info["min_delta"]
            val = row.get(mon, None)
            improved = False
            if val is not None and val == val:  # not NaN
                best_val = self._best_info["value"]
                if mode == "max":
                    improved = (val > best_val + min_delta)
                else:
                    improved = (val < best_val - min_delta)
            if improved:
                self._best_info["value"] = float(val)
                self._best_info["epoch"] = int(epoch)
                st = self._snapshot()
                self._best_info["state"] = st
                if self._best_info.get("save_path"):
                    try:
                        torch.save(st, self._best_info["save_path"])
                    except Exception:
                        pass
                print(f"[best] ↑ new best {mon}={val:.5f} at epoch {epoch+1}")
            # Add best-so-far to row
            row.update({
                "best_monitor": mon,
                "best_so_far": (None if self._best_info["value"] in (math.inf, -math.inf) else float(self._best_info["value"])),
                "best_epoch": (None if self._best_info["epoch"] < 0 else int(self._best_info["epoch"] + 1)),
            })

        # finally: write once, after all updates
        self._write_epoch_log(row)
        return row, should_stop

    def _restore_from_state(self, st: dict):
        if not st:
            return
        self.img_tower.load_state_dict(st["img_tower"])
        self.md_tower.load_state_dict(st["md_tower"])
        if "bridge" in st and hasattr(self, "bridge"):
            self.bridge.load_state_dict(st["bridge"])
        if "head_img" in st and hasattr(self, "head_img"):
            self.head_img.load_state_dict(st["head_img"])
        if "head_md" in st and hasattr(self, "head_md"):
            self.head_md.load_state_dict(st["head_md"])
        if "optimizer" in st:
            self.optimizer.load_state_dict(st["optimizer"])


# --- V2 loader integration (no v1 hypertower import) ---

from classes.v2 import build_papila_profile
from classes.v2.slot_dataset import SlotDataset, slot_collate


__all__ = [
    "HyperTower",
    "V2HyperTower",
    "V2ModeComparisonOps",
    "V2ModeComparator",
]


def _tuple_collate(batch: list[dict[str, object]]):
    """
    Convert a slot-collated batch dict into (imgs, metas, labels) tuples so
    the legacy HyperTower training loop can consume it.
    """
    data = slot_collate(batch)
    imgs = data.get("image_1")
    metas = data.get("matrix_1")
    labels = data.get("label_1")
    if labels is not None and not torch.is_tensor(labels):
        labels = torch.as_tensor(labels, dtype=torch.long)
    return imgs, metas, labels


class V2HyperTower(HyperTower):
    """
    HyperTower that swaps the loader pipeline for V2 slot datasets.
    This preserves the original training loop, logging, and metrics.
    """

    def __init__(self, clinical, args):
        sample_mode = getattr(args, "sample_mode", "eye")
        self._v2_profile = build_papila_profile(
            patient_col="Patient ID",
            label_col=getattr(clinical, "label_col", "Diagnosis"),
            sample_mode=sample_mode,
        )
        super().__init__(clinical, args)

    def _make_loader_for_df(self, df, is_train: bool = False):
        samples = self._v2_profile.build_samples(df=df, clinical=self.clinical)
        dataset = SlotDataset(
            samples,
            self._v2_profile.slot_descriptors(),
            image_transform=self.img_tower.transform,
            image_preprocessor=self.image_preprocessor,
        )

        # Optional: class-balanced bootstrapped sampling for TRAIN only
        if is_train and getattr(self.args, "balanced_sampler", False):
            y = np.asarray(df[self.clinical.label_col].values)
            uniq, counts = np.unique(y, return_counts=True)
            inv = {c: (1.0 / cnt if cnt > 0 else 0.0) for c, cnt in zip(uniq, counts)}
            w = np.array([inv[c] for c in y], dtype=np.float32)
            sampler = WeightedRandomSampler(weights=w, num_samples=len(y), replacement=True)
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                shuffle=False,
                collate_fn=_tuple_collate,
            )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=not is_train,
            collate_fn=_tuple_collate,
        )


class V2ModeComparisonOps:
    """
    Shared mode-comparison training/eval ops hosted in V2 code so scripts
    remain orchestration-only.
    """

    @staticmethod
    def _to_label_tensor(labels, device: torch.device) -> torch.Tensor:
        if torch.is_tensor(labels):
            return labels.to(device=device, dtype=torch.long)
        return torch.as_tensor(labels, dtype=torch.long, device=device)

    @staticmethod
    def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
        for p in module.parameters():
            p.requires_grad = enabled

    @staticmethod
    def _set_single_phase(model: nn.Module, phase: str) -> None:
        if phase == "tower_warmup":
            V2ModeComparisonOps._set_requires_grad(model.img_tower, True)
            V2ModeComparisonOps._set_requires_grad(model.md_tower, True)
            V2ModeComparisonOps._set_requires_grad(model.bridge.classifier_img, True)
            V2ModeComparisonOps._set_requires_grad(model.bridge.classifier_md, True)
            V2ModeComparisonOps._set_requires_grad(model.bridge.W_img, False)
            V2ModeComparisonOps._set_requires_grad(model.bridge.W_md, False)
            V2ModeComparisonOps._set_requires_grad(model.bridge.classifier_fused, False)
            return
        if phase == "fused_warmup":
            V2ModeComparisonOps._set_requires_grad(model.img_tower, False)
            V2ModeComparisonOps._set_requires_grad(model.md_tower, False)
            V2ModeComparisonOps._set_requires_grad(model.bridge.classifier_img, False)
            V2ModeComparisonOps._set_requires_grad(model.bridge.classifier_md, False)
            V2ModeComparisonOps._set_requires_grad(model.bridge.W_img, True)
            V2ModeComparisonOps._set_requires_grad(model.bridge.W_md, True)
            V2ModeComparisonOps._set_requires_grad(model.bridge.classifier_fused, True)
            return
        V2ModeComparisonOps._set_requires_grad(model, True)

    @staticmethod
    def _set_bilateral_phase(model: nn.Module, phase: str) -> None:
        if phase == "tower_warmup":
            V2ModeComparisonOps._set_requires_grad(model.eye_img_tower, True)
            V2ModeComparisonOps._set_requires_grad(model.eye_md_tower, True)
            V2ModeComparisonOps._set_requires_grad(model.joint_img, True)
            V2ModeComparisonOps._set_requires_grad(model.joint_md, True)
            V2ModeComparisonOps._set_requires_grad(model.aux_img, True)
            V2ModeComparisonOps._set_requires_grad(model.aux_md, True)
            V2ModeComparisonOps._set_requires_grad(model.bridge, False)
            return
        if phase == "fused_warmup":
            V2ModeComparisonOps._set_requires_grad(model.eye_img_tower, False)
            V2ModeComparisonOps._set_requires_grad(model.eye_md_tower, False)
            V2ModeComparisonOps._set_requires_grad(model.joint_img, False)
            V2ModeComparisonOps._set_requires_grad(model.joint_md, False)
            V2ModeComparisonOps._set_requires_grad(model.aux_img, False)
            V2ModeComparisonOps._set_requires_grad(model.aux_md, False)
            V2ModeComparisonOps._set_requires_grad(model.bridge, True)
            return
        V2ModeComparisonOps._set_requires_grad(model, True)

    @staticmethod
    def train_single_epoch(model: nn.Module, loader: DataLoader, opt, device: torch.device, *, phase: str, bcd_prob: float = 0.5):
        model.train()
        V2ModeComparisonOps._set_single_phase(model, phase)
        total_loss = total_correct = total_n = 0
        for batch in loader:
            x = batch.get("image_1")
            m = batch.get("matrix_1")
            y = batch.get("label_1")
            if not torch.is_tensor(x) or not torch.is_tensor(m):
                continue
            x = x.to(device)
            m = m.to(device)
            y = V2ModeComparisonOps._to_label_tensor(y, device)
            img_feats = model.img_tower(x)
            md_feats = model.md_tower(m)

            if phase == "tower_warmup":
                logits_i = model.bridge.classifier_img(img_feats)
                logits_m = model.bridge.classifier_md(md_feats)
                loss = 0.5 * (F.cross_entropy(logits_i, y) + F.cross_entropy(logits_m, y))
                logits = 0.5 * (F.softmax(logits_i, dim=1) + F.softmax(logits_m, dim=1))
            elif phase == "fused_warmup":
                logits, _, _ = model.bridge(img_feats, md_feats)
                loss = F.cross_entropy(logits, y)
            else:
                if random() < bcd_prob:
                    if random() < 0.5:
                        logits = model.bridge.classifier_img(img_feats)
                    else:
                        logits = model.bridge.classifier_md(md_feats)
                    loss = F.cross_entropy(logits, y)
                else:
                    logits, _, _ = model.bridge(img_feats, md_feats)
                    loss = F.cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            bs = y.shape[0]
            total_loss += float(loss.item()) * bs
            total_correct += int((logits.argmax(1) == y).sum())
            total_n += bs
        return (
            total_loss / total_n if total_n else float("nan"),
            total_correct / total_n if total_n else float("nan"),
        )

    @staticmethod
    def train_bilateral_epoch(model: nn.Module, loader: DataLoader, opt, device: torch.device, *, phase: str, bcd_prob: float = 0.5):
        model.train()
        V2ModeComparisonOps._set_bilateral_phase(model, phase)
        total_loss = total_correct = total_n = 0
        for batch in loader:
            x1 = batch.get("image_1")
            m1 = batch.get("matrix_1")
            x2 = batch.get("image_2")
            m2 = batch.get("matrix_2")
            y = batch.get("label_1")
            if not (torch.is_tensor(x1) and torch.is_tensor(m1) and torch.is_tensor(x2) and torch.is_tensor(m2)):
                continue
            x1 = x1.to(device); m1 = m1.to(device)
            x2 = x2.to(device); m2 = m2.to(device)
            y = V2ModeComparisonOps._to_label_tensor(y, device)
            joint_img, joint_md = model.encode_joint(x1, m1, x2, m2)

            if phase == "tower_warmup":
                logits_i = model.aux_img(joint_img)
                logits_m = model.aux_md(joint_md)
                loss = 0.5 * (F.cross_entropy(logits_i, y) + F.cross_entropy(logits_m, y))
                logits = 0.5 * (F.softmax(logits_i, dim=1) + F.softmax(logits_m, dim=1))
            elif phase == "fused_warmup":
                logits, _, _ = model.bridge(joint_img, joint_md)
                loss = F.cross_entropy(logits, y)
            else:
                if random() < bcd_prob:
                    if random() < 0.5:
                        logits = model.aux_img(joint_img)
                    else:
                        logits = model.aux_md(joint_md)
                    loss = F.cross_entropy(logits, y)
                else:
                    logits, _, _ = model.bridge(joint_img, joint_md)
                    loss = F.cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            bs = y.shape[0]
            total_loss += float(loss.item()) * bs
            total_correct += int((logits.argmax(1) == y).sum())
            total_n += bs
        return (
            total_loss / total_n if total_n else float("nan"),
            total_correct / total_n if total_n else float("nan"),
        )

    @staticmethod
    def collect_probs_classic(model: nn.Module, loader: DataLoader, device: torch.device):
        model.eval()
        y_chunks, p_chunks = [], []
        with torch.no_grad():
            for batch in loader:
                x1 = batch.get("image_1"); m1 = batch.get("matrix_1")
                x2 = batch.get("image_2"); m2 = batch.get("matrix_2")
                y = batch.get("label_1")
                if not (torch.is_tensor(x1) and torch.is_tensor(m1) and torch.is_tensor(x2) and torch.is_tensor(m2)):
                    continue
                y_t = V2ModeComparisonOps._to_label_tensor(y, device)
                p_od = F.softmax(model(x1.to(device), m1.to(device)), dim=1)
                p_os = F.softmax(model(x2.to(device), m2.to(device)), dim=1)
                y_np = y_t.cpu().numpy()
                y_chunks += [y_np, y_np]
                p_chunks += [p_od.cpu().numpy(), p_os.cpu().numpy()]
        if not y_chunks:
            return np.array([], dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(y_chunks), np.concatenate(p_chunks, axis=0)

    @staticmethod
    def collect_probs_ensemble(model: nn.Module, loader: DataLoader, device: torch.device):
        model.eval()
        y_chunks, p_chunks = [], []
        with torch.no_grad():
            for batch in loader:
                x1 = batch.get("image_1"); m1 = batch.get("matrix_1")
                x2 = batch.get("image_2"); m2 = batch.get("matrix_2")
                y = batch.get("label_1")
                if not (torch.is_tensor(x1) and torch.is_tensor(m1) and torch.is_tensor(x2) and torch.is_tensor(m2)):
                    continue
                y_t = V2ModeComparisonOps._to_label_tensor(y, device)
                p_od = F.softmax(model(x1.to(device), m1.to(device)), dim=1)
                p_os = F.softmax(model(x2.to(device), m2.to(device)), dim=1)
                p = 0.5 * (p_od + p_os)
                y_chunks.append(y_t.cpu().numpy())
                p_chunks.append(p.cpu().numpy())
        if not y_chunks:
            return np.array([], dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(y_chunks), np.concatenate(p_chunks, axis=0)

    @staticmethod
    def collect_probs_bilateral(model: nn.Module, loader: DataLoader, device: torch.device):
        model.eval()
        y_chunks, p_chunks = [], []
        with torch.no_grad():
            for batch in loader:
                x1 = batch.get("image_1"); m1 = batch.get("matrix_1")
                x2 = batch.get("image_2"); m2 = batch.get("matrix_2")
                y = batch.get("label_1")
                if not (torch.is_tensor(x1) and torch.is_tensor(m1) and torch.is_tensor(x2) and torch.is_tensor(m2)):
                    continue
                y_t = V2ModeComparisonOps._to_label_tensor(y, device)
                p = F.softmax(model(x1.to(device), m1.to(device), x2.to(device), m2.to(device)), dim=1)
                y_chunks.append(y_t.cpu().numpy())
                p_chunks.append(p.cpu().numpy())
        if not y_chunks:
            return np.array([], dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(y_chunks), np.concatenate(p_chunks, axis=0)


class V2ModeComparator:
    """
    V2 entrypoint for the three-mode comparison workflow:
    classic, ensemble, bilateral.
    """

    @staticmethod
    def build_parser():
        return build_parser()

    @staticmethod
    def run(cli_args=None):
        parser = V2ModeComparator.build_parser()
        args = parser.parse_args(cli_args)
        return run_mode(args)


# --- Mode comparison orchestration (migrated from mode_compare_engine) ---

def seed_everything(seed: int) -> None:
    pyrandom.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class SingleEyeHT(nn.Module):
    """
    ImageTower + MDTower + Bridge, trained on eye-level samples.
    Supports both Classic (eye-level) and Ensemble (patient-level averaging) eval.
    """

    def __init__(
        self,
        *,
        backbone: str,
        freeze_ratio: float,
        augment: bool,
        clinical_data,
        num_classes: int,
        md_hidden_dim: int = 128,
        fusion_dim: int = 256,
    ):
        super().__init__()
        self.img_tower = ImageTower(
            backbone=backbone,
            freeze_ratio=freeze_ratio,
            augment=augment,
            use_se=False,
        )
        self.md_tower = MDTower(
            clinical_data=clinical_data,
            hidden_dim=md_hidden_dim,
            use_se=False,
        )
        self.bridge = Bridge(
            img_dim=self.img_tower.out_dim,
            meta_dim=self.md_tower.out_dim,
            num_classes=num_classes,
            fusion_dim=fusion_dim,
            mode="fused",
            use_se=False,
        )

    @property
    def transform(self):
        return self.img_tower.transform

    def forward(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        img_feats = self.img_tower(x)
        md_feats  = self.md_tower(meta)
        out_f, _, _ = self.bridge(img_feats, md_feats)
        return out_f


class BilateralHT(nn.Module):
    """
    Bilateral mode with joint towers:
    - shared eye-level towers encode OD/OS independently
    - joint image and metadata towers combine OD/OS embeddings
    - standard Bridge fuses joint image + joint metadata embeddings
    """

    def __init__(
        self,
        *,
        backbone: str,
        freeze_ratio: float,
        augment: bool,
        clinical_data,
        num_classes: int,
        md_hidden_dim: int = 128,
        fusion_dim: int = 256,
    ):
        super().__init__()
        self.eye_img_tower = ImageTower(
            backbone=backbone,
            freeze_ratio=freeze_ratio,
            augment=augment,
            use_se=False,
        )
        self.eye_md_tower = MDTower(
            clinical_data=clinical_data,
            hidden_dim=md_hidden_dim,
            use_se=False,
        )
        img_dim = self.eye_img_tower.out_dim
        md_dim = self.eye_md_tower.out_dim
        self.joint_img = nn.Sequential(
            nn.Linear(2 * img_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, img_dim),
        )
        self.joint_md = nn.Sequential(
            nn.Linear(2 * md_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, md_dim),
        )
        self.bridge = Bridge(
            img_dim=img_dim,
            meta_dim=md_dim,
            num_classes=num_classes,
            fusion_dim=fusion_dim,
            mode="fused",
            use_se=False,
        )
        # Auxiliary heads for tower warmup / BCD tower steps.
        self.aux_img = nn.Linear(img_dim, num_classes)
        self.aux_md = nn.Linear(md_dim, num_classes)

    @property
    def transform(self):
        return self.eye_img_tower.transform

    def encode_joint(
        self,
        x_od: torch.Tensor,
        meta_od: torch.Tensor,
        x_os: torch.Tensor,
        meta_os: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_od = self.eye_img_tower(x_od)
        md_od  = self.eye_md_tower(meta_od)
        img_os = self.eye_img_tower(x_os)
        md_os  = self.eye_md_tower(meta_os)
        joint_img = self.joint_img(torch.cat([img_od, img_os], dim=1))
        joint_md = self.joint_md(torch.cat([md_od, md_os], dim=1))
        return joint_img, joint_md

    def forward(
        self,
        x_od:   torch.Tensor,
        meta_od: torch.Tensor,
        x_os:   torch.Tensor,
        meta_os: torch.Tensor,
    ) -> torch.Tensor:
        joint_img, joint_md = self.encode_joint(x_od, meta_od, x_os, meta_os)
        out_f, _, _ = self.bridge(joint_img, joint_md)
        return out_f


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    V2ModeComparisonOps._set_requires_grad(module, enabled)


def _set_single_phase(model: SingleEyeHT, phase: str) -> None:
    V2ModeComparisonOps._set_single_phase(model, phase)


def _set_bilateral_phase(model: BilateralHT, phase: str) -> None:
    V2ModeComparisonOps._set_bilateral_phase(model, phase)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def filter_eye_samples(samples: list[dict]) -> list[dict]:
    """Keep any single-eye sample with a valid image, matrix, and label."""
    return [
        s for s in samples
        if s.get("image_1") is not None
        and s.get("matrix_1") is not None
        and s.get("label_1") is not None
    ]


def filter_bilateral_samples(samples: list[dict]) -> list[dict]:
    """Keep only patient-level samples where both eyes are fully present."""
    return [
        s for s in samples
        if s.get("image_1") is not None
        and s.get("matrix_1") is not None
        and s.get("image_2") is not None
        and s.get("matrix_2") is not None
        and s.get("label_1") is not None
    ]


def make_loader(
    samples: list[dict],
    slots: dict,
    *,
    image_transform,
    image_preprocessor=None,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    ds = SlotDataset(
        samples,
        slots,
        image_transform=image_transform,
        image_preprocessor=image_preprocessor,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=slot_collate,
    )


def to_label_tensor(labels, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(labels):
        return labels.to(device=device, dtype=torch.long)
    return torch.as_tensor(labels, dtype=torch.long, device=device)


def _drop_mixed_label_patients(df, *, patient_col: str, label_col: str):
    per_patient = (
        df.groupby(patient_col)[label_col]
        .agg(lambda s: set(pd.to_numeric(s, errors="coerce").dropna().astype(int).tolist()))
    )
    mixed = [pid for pid, labels in per_patient.items() if len(labels) > 1]
    if not mixed:
        return df, []
    return df[~df[patient_col].isin(mixed)].reset_index(drop=True), mixed


def _relabel_mixed_patients_to_max(df, *, patient_col: str, label_col: str):
    """Set all rows for each patient to that patient's max observed label."""
    out = df.copy()
    labels = pd.to_numeric(out[label_col], errors="coerce")
    patient_max = labels.groupby(out[patient_col]).transform("max")
    changed_rows = int((labels != patient_max).fillna(False).sum())
    out[label_col] = patient_max.astype(int)
    per_patient_unique = out.groupby(patient_col)[label_col].nunique(dropna=True)
    still_mixed = per_patient_unique[per_patient_unique > 1].index.tolist()
    return out.reset_index(drop=True), changed_rows, still_mixed


def build_image_preprocessor_from_args(args):
    crop_manifest = getattr(args, "img_crop_manifest", None)
    crop_weights = getattr(args, "img_crop_weights", None)
    use_gt = bool(getattr(args, "img_crop_gt", False))
    if not crop_manifest:
        return None
    crop_cache = Path(getattr(args, "img_crop_cache", Path("cache_data/hypertower_crops")))
    persist_cache = bool(getattr(args, "persist_img_crop_cache", False))
    if use_gt:
        pre = ManifestImageCropper(
            manifest_path=Path(crop_manifest),
            scale=getattr(args, "img_crop_scale", 2.5),
            target_size=getattr(args, "img_crop_size", 224),
            cache_dir=crop_cache,
        )
        if not persist_cache:
            pre.clear_cache()
        print(f"[V2 modes] GT disc cropper enabled -> cache at {crop_cache}", flush=True)
        return pre
    if crop_weights:
        pre = UNetImageCropper(
            manifest_path=Path(crop_manifest),
            weights_path=Path(crop_weights),
            normalize=getattr(args, "img_crop_normalize", "per_image"),
            threshold=getattr(args, "img_crop_threshold", 0.5),
            tta=getattr(args, "img_crop_tta", False),
            scale=getattr(args, "img_crop_scale", 2.5),
            target_size=getattr(args, "img_crop_size", 224),
            cache_dir=crop_cache,
        )
        if not persist_cache:
            pre.clear_cache()
        print(f"[V2 modes] UNet disc cropper enabled -> cache at {crop_cache}", flush=True)
        return pre
    print(
        "[V2 modes] img_crop_manifest provided but no --img-crop-gt or --img-crop-weights; cropping disabled.",
        flush=True,
    )
    return None


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def compute_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error: weighted mean of |confidence - accuracy| per bin."""
    if y_true.size == 0:
        return float("nan")
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (confidences >= lo) & (
            confidences <= hi if i == n_bins - 1 else confidences < hi
        )
        if not mask.any():
            continue
        bin_acc  = float((predictions[mask] == y_true[mask]).mean())
        bin_conf = float(confidences[mask].mean())
        ece += float(mask.sum()) / n * abs(bin_conf - bin_acc)
    return float(ece)


def compute_extended_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    num_classes: int,
    n_bins: int = 10,
    preds_override: Optional[np.ndarray] = None,
) -> dict:
    nan = float("nan")
    if y_true.size == 0:
        return dict(
            kappa=nan, mcc=nan, macro_f1=nan,
            per_class_recall=np.full(num_classes, nan), ece=nan,
        )
    preds = preds_override if preds_override is not None else probs.argmax(axis=1)
    try:
        kappa = float(cohen_kappa_score(y_true, preds))
    except Exception:
        kappa = nan
    try:
        mcc = float(matthews_corrcoef(y_true, preds))
    except Exception:
        mcc = nan
    try:
        macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
    except Exception:
        macro_f1 = nan
    try:
        pcr = recall_score(
            y_true, preds, average=None,
            labels=list(range(num_classes)), zero_division=0,
        ).astype(float)
    except Exception:
        pcr = np.full(num_classes, nan)
    ece = compute_ece(y_true, probs, n_bins=n_bins)
    return dict(kappa=kappa, mcc=mcc, macro_f1=macro_f1, per_class_recall=pcr, ece=ece)


def tune_binary_threshold(y_true: np.ndarray, p1: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.5
    grid = np.linspace(0.0, 1.0, 1001)
    best_t, best_acc = 0.5, -1.0
    for t in grid:
        pred = (p1 >= t).astype(int)
        acc = float((pred == y_true).mean())
        if acc > best_acc or (acc == best_acc and abs(t - 0.5) < abs(best_t - 0.5)):
            best_acc, best_t = acc, float(t)
    return best_t


def multiclass_acc_with_bias(y_true: np.ndarray, probs: np.ndarray, bias: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    logits = np.log(np.clip(probs, 1e-8, 1.0)) + bias.reshape(1, -1)
    return float((np.argmax(logits, axis=1) == y_true).mean())


def tune_multiclass_bias(y_true: np.ndarray, probs: np.ndarray, *, iters: int = 2) -> np.ndarray:
    if y_true.size == 0 or probs.size == 0:
        return np.zeros((0,), dtype=float)
    c = probs.shape[1]
    bias = np.zeros((c,), dtype=float)
    grid = np.linspace(-1.0, 1.0, 41)
    for _ in range(iters):
        for k in range(c):
            best_v = bias[k]
            best_acc = multiclass_acc_with_bias(y_true, probs, bias)
            old = bias[k]
            for v in grid:
                bias[k] = float(v)
                acc = multiclass_acc_with_bias(y_true, probs, bias)
                if acc > best_acc or (acc == best_acc and abs(v) < abs(best_v)):
                    best_acc, best_v = acc, float(v)
            bias[k] = best_v
            if np.isnan(best_acc):
                bias[k] = old
    return bias


def _svf(vec) -> Optional[str]:
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=float)
    if arr.size == 0:
        return None
    return "|".join(f"{float(v):.4f}" for v in arr.tolist())


def _score_arrays(y_true: np.ndarray, probs: np.ndarray, num_classes: int):
    """Returns (acc, auc, n)."""
    if y_true.size == 0:
        return float("nan"), float("nan"), 0
    acc = float((probs.argmax(1) == y_true).mean())
    try:
        auc = (
            float(roc_auc_score(y_true, probs[:, 1]))
            if num_classes == 2
            else float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
        )
    except Exception:
        auc = float("nan")
    return acc, auc, int(len(y_true))


def build_eval_transform(backbone: str):
    """Deterministic eval transform matching backbone normalization."""
    key = (backbone or "").lower()
    if key not in BACKBONES:
        raise ValueError(f"Unsupported backbone '{backbone}'")
    spec = BACKBONES[key]
    mean = getattr(spec.weights_default, "meta", {}).get("mean", (0.485, 0.456, 0.406))
    std = getattr(spec.weights_default, "meta", {}).get("std", (0.229, 0.224, 0.225))
    crop = 299 if key == "inception_v3" else 224
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


# ---------------------------------------------------------------------------
# Train / collect
# ---------------------------------------------------------------------------

def train_single_epoch(
    model: SingleEyeHT,
    loader: DataLoader,
    opt,
    device,
    *,
    phase: str,
    bcd_prob: float = 0.5,
):
    return V2ModeComparisonOps.train_single_epoch(
        model, loader, opt, device, phase=phase, bcd_prob=bcd_prob
    )


def train_bilateral_epoch(
    model: BilateralHT,
    loader: DataLoader,
    opt,
    device,
    *,
    phase: str,
    bcd_prob: float = 0.5,
):
    return V2ModeComparisonOps.train_bilateral_epoch(
        model, loader, opt, device, phase=phase, bcd_prob=bcd_prob
    )


def collect_probs_classic(
    model: SingleEyeHT,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Classic eye-level eval using the bilateral val loader.
    OD and OS are treated as independent samples (both contribute to the
    arrays with the same patient label).  Returns (y_true [2N], probs [2N, C]).
    """
    return V2ModeComparisonOps.collect_probs_classic(model, loader, device)


def collect_probs_ensemble(
    model: SingleEyeHT,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Patient-level ensemble eval: average OD and OS softmax probabilities.
    Returns (y_true [N], probs [N, C]).
    """
    return V2ModeComparisonOps.collect_probs_ensemble(model, loader, device)


def collect_probs_single_components(
    model: SingleEyeHT,
    loader: DataLoader,
    device: torch.device,
    *,
    aggregate_patient: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect fused/img/md probabilities for SingleEyeHT.
    - aggregate_patient=False: eye-level (OD/OS as independent samples)
    - aggregate_patient=True : patient-level (average OD/OS per head)
    """
    model.eval()
    y_chunks = []
    pf_chunks, pi_chunks, pm_chunks = [], [], []
    with torch.no_grad():
        for batch in loader:
            x1 = batch.get("image_1"); m1 = batch.get("matrix_1")
            x2 = batch.get("image_2"); m2 = batch.get("matrix_2")
            y = batch.get("label_1")
            if not (torch.is_tensor(x1) and torch.is_tensor(m1) and torch.is_tensor(x2) and torch.is_tensor(m2)):
                continue
            y_t = V2ModeComparisonOps._to_label_tensor(y, device)

            def _per_eye_probs(x, m):
                img_feats = model.img_tower(x.to(device))
                md_feats = model.md_tower(m.to(device))
                out_f, out_i, out_m = model.bridge(img_feats, md_feats)
                return (
                    F.softmax(out_f, dim=1),
                    F.softmax(out_i, dim=1),
                    F.softmax(out_m, dim=1),
                )

            pf_od, pi_od, pm_od = _per_eye_probs(x1, m1)
            pf_os, pi_os, pm_os = _per_eye_probs(x2, m2)

            if aggregate_patient:
                y_chunks.append(y_t.cpu().numpy())
                pf_chunks.append((0.5 * (pf_od + pf_os)).cpu().numpy())
                pi_chunks.append((0.5 * (pi_od + pi_os)).cpu().numpy())
                pm_chunks.append((0.5 * (pm_od + pm_os)).cpu().numpy())
            else:
                y_np = y_t.cpu().numpy()
                y_chunks += [y_np, y_np]
                pf_chunks += [pf_od.cpu().numpy(), pf_os.cpu().numpy()]
                pi_chunks += [pi_od.cpu().numpy(), pi_os.cpu().numpy()]
                pm_chunks += [pm_od.cpu().numpy(), pm_os.cpu().numpy()]

    if not y_chunks:
        z = np.zeros((0, 0), dtype=np.float32)
        return np.array([], dtype=np.int64), z, z, z
    return (
        np.concatenate(y_chunks),
        np.concatenate(pf_chunks, axis=0),
        np.concatenate(pi_chunks, axis=0),
        np.concatenate(pm_chunks, axis=0),
    )


def collect_probs_bilateral(
    model: BilateralHT,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Patient-level bilateral eval. Returns (y_true [N], probs [N, C])."""
    return V2ModeComparisonOps.collect_probs_bilateral(model, loader, device)


def collect_probs_bilateral_components(
    model: BilateralHT,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect fused/img/md probabilities for bilateral joint-tower model."""
    model.eval()
    y_chunks = []
    pf_chunks, pi_chunks, pm_chunks = [], [], []
    with torch.no_grad():
        for batch in loader:
            x1 = batch.get("image_1"); m1 = batch.get("matrix_1")
            x2 = batch.get("image_2"); m2 = batch.get("matrix_2")
            y = batch.get("label_1")
            if not (torch.is_tensor(x1) and torch.is_tensor(m1) and torch.is_tensor(x2) and torch.is_tensor(m2)):
                continue
            y_t = V2ModeComparisonOps._to_label_tensor(y, device)
            joint_img, joint_md = model.encode_joint(
                x1.to(device), m1.to(device), x2.to(device), m2.to(device)
            )
            out_f, _, _ = model.bridge(joint_img, joint_md)
            out_i = model.aux_img(joint_img)
            out_m = model.aux_md(joint_md)
            y_chunks.append(y_t.cpu().numpy())
            pf_chunks.append(F.softmax(out_f, dim=1).cpu().numpy())
            pi_chunks.append(F.softmax(out_i, dim=1).cpu().numpy())
            pm_chunks.append(F.softmax(out_m, dim=1).cpu().numpy())
    if not y_chunks:
        z = np.zeros((0, 0), dtype=np.float32)
        return np.array([], dtype=np.int64), z, z, z
    return (
        np.concatenate(y_chunks),
        np.concatenate(pf_chunks, axis=0),
        np.concatenate(pi_chunks, axis=0),
        np.concatenate(pm_chunks, axis=0),
    )


# ---------------------------------------------------------------------------
# Tuning helpers
# ---------------------------------------------------------------------------

def _tune_and_snap(
    y: np.ndarray,
    p: np.ndarray,
    acc: float,
    num_classes: int,
    args,
    n_bins: int,
) -> tuple[dict, float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Apply threshold/bias tuning and compute extended metrics.
    Returns (snap_dict, tuned_auc, threshold, bias).
    """
    thr  = 0.5 if num_classes == 2 else float("nan")
    bias = None
    ext_preds = None

    if args.tune_binary_threshold and num_classes == 2 and y.size > 0:
        thr = tune_binary_threshold(y, p[:, 1])
        ext_preds = (p[:, 1] >= thr).astype(int)
        acc = float((ext_preds == y).mean())
    elif args.tune_multiclass_bias and num_classes > 2 and y.size > 0:
        bias = tune_multiclass_bias(y, p)
        logits = np.log(np.clip(p, 1e-8, 1.0)) + bias.reshape(1, -1)
        ext_preds = np.argmax(logits, axis=1)
        acc = float((ext_preds == y).mean())

    ext = compute_extended_metrics(y, p, num_classes, n_bins=n_bins, preds_override=ext_preds)
    _, auc, n = _score_arrays(y, p, num_classes)

    snap = dict(
        auc=auc, acc=acc, n=n,
        kappa=ext["kappa"], mcc=ext["mcc"], macro_f1=ext["macro_f1"],
        per_class_recall=ext["per_class_recall"], ece=ext["ece"],
        threshold=thr, bias=bias,
    )
    return snap, auc, thr, bias


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

def _nan() -> float:
    return float("nan")


@dataclass
class FoldResult:
    mode: str
    fold: int
    # Epoch where each model hit its peak val AUC
    best_epoch_single: int   # SingleEyeHT — selected by ensemble val AUC
    best_epoch_bilat:  int   # BilateralHT — selected by bilateral val AUC
    # Classic (eye-level eval of SingleEyeHT; n = 2 * ensemble_val_n)
    classic_val_auc:       float
    classic_val_acc:       float
    classic_val_kappa:     float
    classic_val_mcc:       float
    classic_val_f1:        float
    classic_val_recall:    Optional[str]
    classic_val_ece:       float
    classic_val_threshold: float
    classic_val_bias:      Optional[str]
    classic_val_n:         int
    # Ensemble (patient-level eval of same SingleEyeHT)
    ensemble_val_auc:       float
    ensemble_val_acc:       float
    ensemble_val_kappa:     float
    ensemble_val_mcc:       float
    ensemble_val_f1:        float
    ensemble_val_recall:    Optional[str]
    ensemble_val_ece:       float
    ensemble_val_threshold: float
    ensemble_val_bias:      Optional[str]
    ensemble_val_n:         int
    # Bilateral (BilateralHT patient-level)
    bilat_val_auc:       float
    bilat_val_acc:       float
    bilat_val_kappa:     float
    bilat_val_mcc:       float
    bilat_val_f1:        float
    bilat_val_recall:    Optional[str]
    bilat_val_ece:       float
    bilat_val_threshold: float
    bilat_val_bias:      Optional[str]
    bilat_val_n:         int
    # Holdout metrics (evaluated at the same epoch as best val; nan if no holdout)
    classic_holdout_auc:  float
    classic_holdout_acc:  float
    ensemble_holdout_auc: float
    ensemble_holdout_acc: float
    bilat_holdout_auc:    float
    bilat_holdout_acc:    float
    holdout_n:            int    # number of holdout bilateral samples
    # Training sample counts
    single_train_n: int
    bilat_train_n:  int


@dataclass
class FoldArtifacts:
    y_true_classic: Optional[np.ndarray]
    probs_classic: Optional[np.ndarray]
    y_true_ensemble: Optional[np.ndarray]
    probs_ensemble: Optional[np.ndarray]
    y_true_bilat: Optional[np.ndarray]
    probs_bilat: Optional[np.ndarray]


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _f(v) -> Optional[float]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), 6)


def _sv(vec) -> Optional[str]:
    if vec is None:
        return None
    return "|".join(f"{float(v):.4f}" for v in vec)


# ---------------------------------------------------------------------------
# Main fold runner
# ---------------------------------------------------------------------------

def run_fold(
    fold: int,
    split,
    mode: str,
    args,
    device: torch.device,
    data,
    num_classes: int,
    profile_eye,
    profile_patient,
    image_preprocessor,
    fold_dir: Path,
    tower_mode: str,
) -> tuple[FoldResult, FoldArtifacts]:
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

    # ---- samples -------------------------------------------------------------
    eye_train   = filter_eye_samples(
        profile_eye.build_samples(df=split.train, clinical=data)
    )
    bilat_train = filter_bilateral_samples(
        profile_patient.build_samples(df=split.train, clinical=data)
    )
    bilat_val   = filter_bilateral_samples(
        profile_patient.build_samples(df=split.val, clinical=data)
    )

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
            y_true_classic=None,
            probs_classic=None,
            y_true_ensemble=None,
            probs_ensemble=None,
            y_true_bilat=None,
            probs_bilat=None,
        )

    # ---- models --------------------------------------------------------------
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

    # ---- loaders -------------------------------------------------------------
    train_single_loader = None
    train_bilat_loader = None
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

    # ---- holdout loader (if holdout patients are available) ------------------
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

    opt_single = torch.optim.Adam(single.parameters(), lr=args.lr) if run_single else None
    opt_bilateral = torch.optim.Adam(bilateral.parameters(), lr=args.lr) if run_bilat else None

    # ---- epoch log -----------------------------------------------------------
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

    # ---- best-epoch trackers -------------------------------------------------
    best_single_auc = -1.0  # tracked by ensemble AUC
    best_bilat_auc  = -1.0
    best_epoch_single = 0
    best_epoch_bilat  = 0
    best_single_state: Optional[dict] = None
    best_bilat_state: Optional[dict] = None
    snap_classic:  dict = {}
    snap_ensemble: dict = {}
    snap_bilat:    dict = {}
    # holdout snaps (metrics captured at the same epoch as best val)
    snap_holdout_single: dict = {}
    snap_holdout_bilat:  dict = {}
    # separate best-holdout trackers (for checkpointing)
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

    # ---- epoch loop ----------------------------------------------------------
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
                single,
                train_single_loader,
                opt_single,
                device,
                phase=phase_single,
                bcd_prob=float(args.bcd_prob),
            )
        else:
            sl_loss, sl_acc = nan, nan

        if run_bilat and bilat_active:
            bl_loss, bl_acc = train_bilateral_epoch(
                bilateral,
                train_bilat_loader,
                opt_bilateral,
                device,
                phase=phase_bilat,
                bcd_prob=float(args.bcd_prob),
            )
        else:
            bl_loss, bl_acc = nan, nan

        if run_single and tower_mode == "single":
            y_cl, p_cl, p_cl_img, p_cl_md = collect_probs_single_components(
                single, val_loader, device, aggregate_patient=False
            )
            cl_acc, cl_auc, cl_n = _score_arrays(y_cl, p_cl, num_classes)
            cl_acc_img = float((p_cl_img.argmax(1) == y_cl).mean()) if y_cl.size else nan
            cl_acc_md = float((p_cl_md.argmax(1) == y_cl).mean()) if y_cl.size else nan
            _, cl_auc_img, _ = _score_arrays(y_cl, p_cl_img, num_classes)
            _, cl_auc_md, _ = _score_arrays(y_cl, p_cl_md, num_classes)
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
            en_acc_md = float((p_en_md.argmax(1) == y_en).mean()) if y_en.size else nan
            _, en_auc_img, _ = _score_arrays(y_en, p_en_img, num_classes)
            _, en_auc_md, _ = _score_arrays(y_en, p_en_md, num_classes)
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
            y_bi, p_bi, p_bi_img, p_bi_md = collect_probs_bilateral_components(bilateral, val_loader, device)
            bi_acc, bi_auc, bi_n = _score_arrays(y_bi, p_bi, num_classes)
            bi_acc_img = float((p_bi_img.argmax(1) == y_bi).mean()) if y_bi.size else nan
            bi_acc_md = float((p_bi_md.argmax(1) == y_bi).mean()) if y_bi.size else nan
            _, bi_auc_img, _ = _score_arrays(y_bi, p_bi_img, num_classes)
            _, bi_auc_md, _ = _score_arrays(y_bi, p_bi_md, num_classes)
        else:
            y_bi = np.array([], dtype=np.int64)
            p_bi = np.zeros((0, 0), dtype=np.float32)
            bi_acc = bi_auc = nan
            bi_n = 0
            bi_acc_img = bi_acc_md = bi_auc_img = bi_auc_md = nan

        # --- holdout evaluation -----------------------------------------------
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
                y_bi_h, p_bi_h, _, _ = collect_probs_bilateral_components(bilateral, holdout_loader, device)
                _, bi_auc_h, _ = _score_arrays(y_bi_h, p_bi_h, num_classes)
                bi_acc_h = float((p_bi_h.argmax(1) == y_bi_h).mean()) if y_bi_h.size else nan
            else:
                bi_auc_h = bi_acc_h = nan
        else:
            cl_auc_h = cl_acc_h = en_auc_h = en_acc_h = bi_auc_h = bi_acc_h = nan

        # Best-epoch checks: checkpointing is restricted to the main phase only.
        target_single_auc = cl_auc if tower_mode == "single" else en_auc
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
            # capture holdout metrics at this val-best epoch
            snap_holdout_single = {"auc": float(target_holdout_single_auc), "acc": float(cl_acc_h if tower_mode == "single" else en_acc_h)}

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
            # capture holdout metrics at this val-best epoch
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
            "phase_single": phase_single,
            "phase_bilat": phase_bilat,
            "main_epoch_single": main_epoch_single,
            "main_epoch_bilat": main_epoch_bilat,
            "single_active": int(single_active),
            "bilat_active": int(bilat_active),
            "single_train_loss": _f(sl_loss), "single_train_acc": _f(sl_acc),
            "classic_val_auc":      _f(cl_auc),   "classic_val_acc":   _f(cl_acc),  "classic_val_n":  cl_n,
            "ensemble_val_auc":     _f(en_auc),   "ensemble_val_acc":  _f(en_acc),  "ensemble_val_n": en_n,
            "bilat_train_loss":     _f(bl_loss),  "bilat_train_acc":   _f(bl_acc),
            "bilat_val_auc":        _f(bi_auc),   "bilat_val_acc":     _f(bi_acc),  "bilat_val_n":    bi_n,
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

    # Export best-epoch prediction artifacts for easier side-by-side analysis.
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
        classic_holdout_auc=snap_holdout_single.get("auc", nan) if tower_mode == "single" else nan,
        classic_holdout_acc=snap_holdout_single.get("acc", nan) if tower_mode == "single" else nan,
        ensemble_holdout_auc=snap_holdout_single.get("auc", nan) if tower_mode == "ensemble" else nan,
        ensemble_holdout_acc=snap_holdout_single.get("acc", nan) if tower_mode == "ensemble" else nan,
        bilat_holdout_auc=snap_holdout_bilat.get("auc", nan),
        bilat_holdout_acc=snap_holdout_bilat.get("acc", nan),
        holdout_n=len(holdout_bilat),
        single_train_n=len(eye_train),
        bilat_train_n=len(bilat_train),
    ), FoldArtifacts(
        y_true_classic=y_cl_best,
        probs_classic=p_cl_best,
        y_true_ensemble=y_en_best,
        probs_ensemble=p_en_best,
        y_true_bilat=y_bi_best,
        probs_bilat=p_bi_best,
    )


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _summary(results: list[FoldResult]) -> dict:
    def _ms(vals):
        v = np.array([x for x in vals if x is not None and not np.isnan(float(x))], dtype=float)
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

    # Deltas: ensemble − classic (eval strategy effect, same model)
    #         bilateral − ensemble (bilateral training effect)
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
# CLI
# ---------------------------------------------------------------------------

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
        "--folds",
        type=int,
        default=None,
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
    # Optional ROI cropping (GT manifest or UNet-generated mask crop)
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
    # Backward-compatible alias; default behavior is now to keep raw labels.
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


def parse_args():
    return build_parser().parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_mode(args) -> Path:
    # Defensive local import for CLI/entrypoint execution paths.
    import csv

    device = choose_device(args.device)
    seed_everything(args.seed)

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
    image_preprocessor = build_image_preprocessor_from_args(args)

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"hypertower_modes_{ts}"
    out_dir = Path(args.output_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = args.eval_mode
    tower_mode = "single" if args.tower_mode == "classic" else args.tower_mode
    df_mode = data.df.copy()

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

    profile_eye = build_papila_profile(
        patient_col="Patient ID", label_col=args.label_col, sample_mode="eye"
    )
    profile_patient = build_papila_profile(
        patient_col="Patient ID", label_col=args.label_col, sample_mode="patient"
    )

    tm_dir = out_dir / mode / tower_mode
    tm_dir.mkdir(parents=True, exist_ok=True)
    fold_results: list[FoldResult] = []
    for fold in range(n_folds):
        seed_everything(args.seed + fold * 100)
        fold_dir = tm_dir / f"fold{fold}"
        fold_dir.mkdir(exist_ok=True)

        print(f"\n[{mode}:{tower_mode}] fold {fold+1}/{n_folds}", flush=True)
        result, artifacts = run_fold(
            fold=fold,
            split=plans[fold],
            mode=mode,
            args=args,
            device=device,
            data=data,
            num_classes=num_classes,
            profile_eye=profile_eye,
            profile_patient=profile_patient,
            image_preprocessor=image_preprocessor,
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

    summary = _summary(fold_results)
    _print_summary(f"{mode}:{tower_mode}", summary, tower_mode=tower_mode)
    metric_key = {
        "single": "classic_val_auc",
        "ensemble": "ensemble_val_auc",
        "bilateral": "bilat_val_auc",
    }[tower_mode]
    fold_metrics = []
    best_vals = []
    for r in fold_results:
        best_val = getattr(r, metric_key)
        fold_metrics.append(
            {
                "fold": r.fold,
                "best_metric_value": _f(best_val),
                "best_epoch": (r.best_epoch_bilat if tower_mode == "bilateral" else r.best_epoch_single),
                "monitor": metric_key,
            }
        )
        if not np.isnan(float(best_val)):
            best_vals.append(float(best_val))

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

    # Merge run-level summary across sequential invocations.
    root_summary_path = out_dir / "summary.json"
    if root_summary_path.exists():
        try:
            payload = json.loads(root_summary_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    else:
        payload = {}
    payload.setdefault("run_name", run_name)
    payload.setdefault("timestamp", ts)
    payload["config"] = vars(args)
    payload.setdefault("summaries", {})
    payload["summaries"][f"{mode}:{tower_mode}"] = summary
    root_summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nOutputs written to: {out_dir}")
    return out_dir
