"""transforms — image transform utilities for v4 towers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from torchvision import transforms

from v4.classes.accessory.backbones import BACKBONES, _is_timm_backbone

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD:  Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class ImageTransformConfig:
    crop_size:    int                              = 224
    resize_size:  int                              = 256
    mean:         Tuple[float, float, float]       = IMAGENET_MEAN
    std:          Tuple[float, float, float]       = IMAGENET_STD
    augment:      bool                             = True
    rotation_deg: int                              = 15
    color_jitter: Tuple[float, float, float, float] = (0.1, 0.1, 0.1, 0.05)
    hflip:        bool                             = True
    vflip:        bool                             = True

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
        ops += [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ]
        return transforms.Compose(ops)

    def build_precache(self) -> transforms.Compose:
        """Deterministic prefix: PIL → resized CHW float32 in [0, 1].

        Output is suitable for caching; per-batch ``build_postcache`` finishes
        the pipeline (augment + normalize) on tensors.
        """
        return transforms.Compose([
            transforms.Resize(self.resize_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
        ])

    def build_postcache(self) -> transforms.Compose:
        """Per-batch tail run on cached float32 [0, 1] CHW tensors.

        Augmentations operate on tensors (torchvision v1 supports this for
        Flip/Rotation/ColorJitter on tensor input).  Normalize is applied last.
        """
        ops = []
        if self.augment:
            if self.hflip:
                ops.append(transforms.RandomHorizontalFlip())
            if self.vflip:
                ops.append(transforms.RandomVerticalFlip())
            if self.rotation_deg:
                ops.append(transforms.RandomRotation(self.rotation_deg))
            if self.color_jitter:
                ops.append(transforms.ColorJitter(*self.color_jitter))
        ops.append(transforms.Normalize(mean=self.mean, std=self.std))
        return transforms.Compose(ops)


def backbone_transform_config(backbone_name: str, augment: bool = True) -> ImageTransformConfig:
    """Build an ImageTransformConfig using the backbone's default normalisation stats."""
    key = (backbone_name or "").lower()
    if _is_timm_backbone(key):
        # ConvNeXt-V2 and other timm models we currently expose are all
        # pretrained with standard ImageNet stats at 224×224.
        return ImageTransformConfig(crop_size=224, mean=IMAGENET_MEAN,
                                    std=IMAGENET_STD, augment=augment)
    if key not in BACKBONES:
        raise ValueError(f"Unknown backbone '{backbone_name}'.")
    spec = BACKBONES[key]
    mean = getattr(spec.weights_default, "meta", {}).get("mean", IMAGENET_MEAN)
    std  = getattr(spec.weights_default, "meta", {}).get("std",  IMAGENET_STD)
    crop = 299 if key == "inception_v3" else 224
    return ImageTransformConfig(crop_size=crop, mean=mean, std=std, augment=augment)


def build_backbone_transform(backbone_name: str, augment: bool = True) -> transforms.Compose:
    return backbone_transform_config(backbone_name, augment=augment).build()


def build_eval_transform(backbone_name: str) -> transforms.Compose:
    """Deterministic eval transform — no augmentation, backbone-matched normalisation."""
    return build_backbone_transform(backbone_name, augment=False)


def build_split_transforms(
    backbone_name: str, augment: bool = True
) -> tuple[transforms.Compose, transforms.Compose]:
    """Return (precache, postcache) transform pair for tensor-cached image towers.

    precache  : PIL → CHW float32 in [0, 1]   (deterministic, run once at fill)
    postcache : tensor → augmented + normalized tensor   (run per batch)
    """
    cfg = backbone_transform_config(backbone_name, augment=augment)
    return cfg.build_precache(), cfg.build_postcache()
