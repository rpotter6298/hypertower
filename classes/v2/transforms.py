from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple, Union
import numpy as np
from PIL import Image

from torchvision import transforms

from classes.backbones import BACKBONES


IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class ImageTransformConfig:
    """
    Mirrors the hypertower v1 preprocessing:
      - Resize(256)
      - CenterCrop(crop)
      - Optional augmentations (H/V flip, rotation, color jitter)
      - ToTensor + Normalize(mean/std)
    """

    crop_size: int = 224
    resize_size: int = 256
    mean: Tuple[float, float, float] = IMAGENET_MEAN
    std: Tuple[float, float, float] = IMAGENET_STD
    augment: bool = True
    rotation_deg: int = 15
    color_jitter: Tuple[float, float, float, float] = (0.1, 0.1, 0.1, 0.05)
    hflip: bool = True
    vflip: bool = True

    def build(self) -> transforms.Compose:
        ops = [
            transforms.Resize(self.resize_size),
            transforms.CenterCrop(self.crop_size),
        ]
        if self.augment:
            if self.hflip:
                ops.append(transforms.RandomHorizontalFlip())
            if self.vflip:
                ops.append(transforms.RandomVerticalFlip())
            if self.rotation_deg:
                ops.append(transforms.RandomRotation(self.rotation_deg))
            if self.color_jitter:
                ops.append(transforms.ColorJitter(*self.color_jitter))
        ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )
        return transforms.Compose(ops)


def backbone_transform_config(backbone_name: str, augment: bool = True) -> ImageTransformConfig:
    """
    Build a transform config that matches v1 ImageTower/backbone preprocessing.
    Uses DEFAULT weights mean/std and InceptionV3 crop size when relevant.
    """
    key = (backbone_name or "").lower()
    if key not in BACKBONES:
        raise ValueError(f"Unsupported backbone '{backbone_name}'.")
    spec = BACKBONES[key]
    mean = getattr(spec.weights_default, "meta", {}).get("mean", IMAGENET_MEAN)
    std = getattr(spec.weights_default, "meta", {}).get("std", IMAGENET_STD)
    crop = 299 if key == "inception_v3" else 224
    return ImageTransformConfig(crop_size=crop, mean=mean, std=std, augment=augment)


def build_backbone_transform(backbone_name: str, augment: bool = True) -> transforms.Compose:
    return backbone_transform_config(backbone_name, augment=augment).build()


def build_imagenet_transform(augment: bool = True, crop_size: int = 224) -> transforms.Compose:
    return ImageTransformConfig(crop_size=crop_size, augment=augment).build()


@dataclass
class ResizeTransform:
    size: Union[int, Tuple[int, int]] = 256
    interpolation: int = Image.BILINEAR

    def __post_init__(self) -> None:
        self._op = transforms.Resize(self.size, interpolation=self.interpolation)

    def __call__(self, image: Image.Image) -> Image.Image:
        return self._op(image)


@dataclass
class CenterCropTransform:
    size: Union[int, Tuple[int, int]] = 224

    def __post_init__(self) -> None:
        self._op = transforms.CenterCrop(self.size)

    def __call__(self, image: Image.Image) -> Image.Image:
        return self._op(image)


class UnetMaskProvider:
    """
    Placeholder for a UNet-powered mask provider.
    This will be replaced once a UNet tower is wired in.
    """

    def __call__(self, image: Image.Image, image_path: Optional[str] = None):
        raise NotImplementedError("UNet mask provider is not wired yet.")


@dataclass
class ROICropTransform:
    """
    Crop an image using a binary mask (GT or UNet).
    Expects a mask of the same spatial size as the image; nonzero pixels are ROI.
    """

    mask_source: str = "gt"  # "gt" | "unet"
    mask_provider: Optional[Callable[[Image.Image, Optional[str]], np.ndarray]] = None
    scale: float = 2.5
    target_size: Optional[Tuple[int, int]] = (224, 224)
    fallback_to_original: bool = True

    def __post_init__(self) -> None:
        if self.mask_source not in {"gt", "unet"}:
            raise ValueError(f"mask_source must be 'gt' or 'unet', got '{self.mask_source}'.")

    def __call__(
        self,
        image: Image.Image,
        mask: Optional[Union[np.ndarray, Image.Image]] = None,
        image_path: Optional[str] = None,
    ) -> Image.Image:
        resolved_mask = mask
        if resolved_mask is None and self.mask_provider is not None:
            resolved_mask = self.mask_provider(image, image_path)
        if resolved_mask is None:
            if self.fallback_to_original:
                return image
            raise ValueError("ROI crop requested but no mask provided.")

        mask_arr = (
            np.asarray(resolved_mask)
            if not isinstance(resolved_mask, Image.Image)
            else np.array(resolved_mask)
        )
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[..., 0]
        mask_arr = mask_arr > 0
        if not np.any(mask_arr):
            return image if self.fallback_to_original else image

        ys, xs = np.where(mask_arr)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        width = (x_max - x_min + 1)
        height = (y_max - y_min + 1)
        size = max(width, height) * float(self.scale)

        left = int(round(cx - size / 2))
        right = int(round(cx + size / 2))
        upper = int(round(cy - size / 2))
        lower = int(round(cy + size / 2))

        left = max(0, left)
        upper = max(0, upper)
        right = min(image.width, right)
        lower = min(image.height, lower)
        crop = image.crop((left, upper, right, lower))
        if self.target_size is not None:
            crop = crop.resize(self.target_size, Image.BILINEAR)
        return crop


@dataclass
class JitterBundleTransform:
    """
    Augmentations bundle: flips, rotation, color jitter.
    """

    hflip: bool = True
    vflip: bool = True
    rotation_deg: int = 15
    color_jitter: Optional[Tuple[float, float, float, float]] = (0.1, 0.1, 0.1, 0.05)

    def __post_init__(self) -> None:
        ops = []
        if self.hflip:
            ops.append(transforms.RandomHorizontalFlip())
        if self.vflip:
            ops.append(transforms.RandomVerticalFlip())
        if self.rotation_deg:
            ops.append(transforms.RandomRotation(self.rotation_deg))
        if self.color_jitter:
            ops.append(transforms.ColorJitter(*self.color_jitter))
        self._op = transforms.Compose(ops) if ops else None

    def __call__(self, image: Image.Image) -> Image.Image:
        if self._op is None:
            return image
        return self._op(image)


TRANSFORM_REGISTRY = {
    "resize": ResizeTransform,
    "roi_crop": ROICropTransform,
    "center_crop": CenterCropTransform,
    "jitter_bundle": JitterBundleTransform,
}


def _parse_color_jitter(value: Optional[Union[str, Iterable[float]]]) -> Optional[Tuple[float, float, float, float]]:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if not parts:
            return None
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            return None
        if len(nums) == 1:
            return (nums[0], nums[0], nums[0], nums[0])
        if len(nums) >= 4:
            return (nums[0], nums[1], nums[2], nums[3])
        return tuple(nums + [nums[-1]] * (4 - len(nums)))  # pad to length 4
    try:
        vals = list(value)
    except TypeError:
        return None
    if not vals:
        return None
    vals = [float(v) for v in vals]
    if len(vals) == 1:
        return (vals[0], vals[0], vals[0], vals[0])
    if len(vals) >= 4:
        return (vals[0], vals[1], vals[2], vals[3])
    return tuple(vals + [vals[-1]] * (4 - len(vals)))


def build_transform_chain(
    transform_specs: Iterable[object],
    *,
    backbone_name: str,
    augment: bool = True,
    mask_provider: Optional[Callable[[Image.Image, Optional[str]], np.ndarray]] = None,
    strict: bool = True,
) -> transforms.Compose:
    """
    Build an image transform pipeline from a list of transform specs plus the
    standard ToTensor + Normalize steps. This mirrors the V1 preprocessing
    but uses the explicit transform nodes from config.
    """
    ops: list[Callable[[Image.Image], Image.Image]] = []
    for spec in transform_specs:
        transform_type = getattr(spec, "transform_type", None)
        params = getattr(spec, "params", None)
        if transform_type is None and isinstance(spec, dict):
            transform_type = spec.get("transformType") or spec.get("transform_type")
            params = spec
        params = params or {}

        if transform_type == "resize":
            size = params.get("resizeSize", 256)
            ops.append(ResizeTransform(size=size))
        elif transform_type == "center_crop":
            size = params.get("centerCropSize", 224)
            ops.append(CenterCropTransform(size=size))
        elif transform_type == "jitter_bundle":
            if not augment:
                continue
            jitter = JitterBundleTransform(
                hflip=bool(params.get("jitterHFlip", True)),
                vflip=bool(params.get("jitterVFlip", True)),
                rotation_deg=int(params.get("jitterRotation", 15) or 0),
                color_jitter=_parse_color_jitter(params.get("jitterColor"))
                if params.get("jitterColorEnabled", True)
                else None,
            )
            ops.append(jitter)
        elif transform_type == "roi_crop":
            roi = ROICropTransform(
                mask_source=params.get("roiMaskSource", "gt"),
                mask_provider=mask_provider,
                scale=float(params.get("roiScale", 2.5)),
                target_size=(int(params.get("roiTargetSize", 224)), int(params.get("roiTargetSize", 224)))
                if params.get("roiTargetSize") is not None
                else None,
                fallback_to_original=bool(params.get("roiFallback", True)),
            )
            if roi.mask_provider is None and roi.mask_source == "unet":
                if strict:
                    raise ValueError("ROI crop requires a mask provider for 'unet' source.")
            ops.append(roi)
        else:
            if strict:
                raise ValueError(f"Unsupported transform type: {transform_type!r}")

    # Always end with tensor + normalize, using backbone defaults
    cfg = backbone_transform_config(backbone_name, augment=augment)
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.mean, std=cfg.std),
        ]
    )
    return transforms.Compose(ops)
