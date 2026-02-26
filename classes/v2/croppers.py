"""Optic-disc image croppers and preprocessor factory for V2."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torchvision import transforms

from classes.geometry_features import compute_geometry_features, disc_cup_from_mask_image
from classes.refuge_classification import _geometry_from_mask
from classes.unet_segmenter import UNetSegmenter


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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_image_preprocessor_from_args(args):
    """Construct the correct image cropper from CLI args, or return None."""
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
