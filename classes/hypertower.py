from __future__ import annotations
import os
import csv
import math
import argparse
import json
from pathlib import Path
import shutil
import logging
import pandas as pd
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from classes import ClinicalData, ClinicalDataset, ImageTower, MDTower, Bridge, VoteBridge, EarlyStopper
from classes.unet_segmenter import UNetSegmenter
from classes.geometry_features import (
    FEATURE_DIM,
    compute_geometry_features,
    disc_cup_from_mask_image,
)
from classes.refuge_classification import _geometry_from_mask
from torchvision import transforms

# from clinical_data import ClinicalData
# from dataset import ClinicalDataset
# from image_tower import ImageTower
# from md_tower import MDTower
# from bridge import Bridge, VoteBridge
from random import random
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from typing import Optional, Tuple


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
    while still delegating encoding/paths/labels to the ClinicalData object."""
    def __init__(self, base: ClinicalData, df):
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
        # ClinicalData returns numpy; convert to torch here to keep towers torch-only
        import torch as _torch
        vec = self.base.vectorize_row(row)
        return _torch.as_tensor(vec, dtype=_torch.float32)

    def get_image_path(self, row):
        return self.base.get_image_path(row)

    def get_label(self, row):
        return int(row[self.base.label_col])


class HyperTower:


    def __init__(self, clinical: ClinicalData, args):
        # Expects a fully built ClinicalData (add_df handled upstream)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.clinical = clinical
        self.warmup_tower_epochs = getattr(args, "warmup_tower_epochs", 2)
        self.warmup_fused_epochs = getattr(args, "warmup_fused_epochs", 3)
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
            crop_cache = getattr(args, "img_crop_cache", Path("analysis_data/hypertower_crops"))
            crop_cache = Path(crop_cache)
            if use_gt:
                self.image_preprocessor = ManifestImageCropper(
                    manifest_path=Path(crop_manifest),
                    scale=getattr(args, "img_crop_scale", 2.5),
                    target_size=getattr(args, "img_crop_size", 224),
                    cache_dir=crop_cache,
                )
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
                print(f"[HyperTower] UNet disc cropper enabled → cache at {crop_cache}")
            else:
                print("[HyperTower] img_crop_manifest provided but no weights/gt flag; skipping cropping")

        self.use_geometry_features = bool(getattr(args, "img_geometry_features", False))
        if self.use_geometry_features and self.image_preprocessor is None:
            raise ValueError("img_geometry_features requires --img-crop-manifest with either --img-crop-weights or --img-crop-gt.")
        self.geometry_dim = FEATURE_DIM if self.use_geometry_features else 0

        # Towers derive their dimensions from ClinicalData
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
        self.epochs = args.epochs
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

        # File logger (per-run); simple line format, no duplication to root
        self.logger = logging.getLogger("hypertower")
        self.logger.setLevel(logging.INFO)
        # Replace existing handlers to avoid duplicate lines across folds
        self.logger.handlers = []
        fh = logging.FileHandler(str(self.train_log_path))
        fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        self.logger.addHandler(fh)
        self.logger.propagate = False

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

        # DataLoaders are (re)built each epoch from ClinicalData's splits
        self.train_loader = None
        self.test_loader = None
        self._last_step_mix = None

        if not getattr(self.clinical, "folds", None):
            raise RuntimeError("ClinicalData has no built folds. Did you call add_df(...) upstream?")

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
        """
        Write an epoch row. Initializes the CSV with a stable header on first call.
        Later calls auto-fill missing columns with None so nothing gets dropped.
        """
        # known optional columns we might add later
        OPTIONAL_COLS = [
            "pct_fused", "pct_img", "pct_md", "phase",
            "se_mean", "se_std", "se_pct_lt_0.2", "se_pct_gt_0.8",
            "holdout_loss", "holdout_acc_fused", "holdout_acc_img", "holdout_acc_md",
            "holdout_auc_fused", "holdout_auc_img", "holdout_auc_md",
        ]

        # On first call: create file, lock fieldnames
        if not hasattr(self, "_epoch_log_writer"):
            # union of current row keys + optional columns so header includes them even if None now
            fieldnames = list(dict.fromkeys([*row.keys(), *OPTIONAL_COLS]))
            self._epoch_log_path = Path(path) if path is not None else self.epoch_log_path
            self._epoch_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._epoch_log_fp = open(self._epoch_log_path, "w", newline="")
            self._epoch_log_writer = csv.DictWriter(self._epoch_log_fp, fieldnames=fieldnames)
            self._epoch_log_writer.writeheader()
            self._epoch_log_fields = fieldnames  # remember for future rows

        # Ensure all header fields exist in this row
        for k in self._epoch_log_fields:
            row.setdefault(k, None)

        # Write and flush
        self._epoch_log_writer.writerow({k: row.get(k) for k in self._epoch_log_fields})
        self._epoch_log_fp.flush()

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

        for epoch in range(self.epochs):
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
                f"[Epoch {epoch+1}/{self.epochs}] Train Loss: {avg_loss:.4f} | "
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
            f"[Epoch {epoch+1}/{self.epochs}] Eval  Loss: {metrics_val['loss']:.4f} | "
            f"Fused Acc: {100*metrics_val['acc_fused']:.2f}% | "
            f"Img Acc: {100*metrics_val['acc_img']:.2f}% | "
            f"Md Acc: {100*metrics_val['acc_md']:.2f}% | "
            f"Fused AUC: {ffmt(metrics_val['auc_fused'])} | "
            f"Img AUC: {ffmt(metrics_val['auc_img'])} | "
            f"Md AUC: {ffmt(metrics_val['auc_md'])}"
        )

        if metrics_holdout:
            print(
                f"[Epoch {epoch+1}/{self.epochs}] Holdout Loss: {metrics_holdout['loss']:.4f} | "
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
