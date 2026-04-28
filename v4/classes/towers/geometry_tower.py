"""geometry_tower — GeometrySegEncoder for v4.

A CNN tower that takes a per-eye disc/cup *segmentation map* as input (rather
than the raw fundus image) and contributes its pooled embedding to fusion.

Seg maps are produced by an underlying loader (GT contour rasterisation or
UNet inference) during early_pass, then cached per fold.

UNet fine-tuning lives in early_pass too — the loader's `finetune(train_samples)`
call uses only the training split, then precompute() runs inference on all
fold samples (train + val + test).

Config example:
{
  "name":         "geom",
  "module":       "v4.classes.towers.geometry_tower",
  "class":        "GeometrySegEncoder",
  "data_source":  "image",
  "args": {
    "backbone":          "resnet18",
    "channels":          3,
    "target_size":       224,
    "augment":           true,
    "seg_source":        "gt",
    "contour_dir":       "Papila/ExpertsSegmentations/Contours"
  }
}
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torchvision import models

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from v4.classes.accessory.backbones import build_backbone
from v4.classes.towerbase import TowerBase


class GeometrySegEncoder(TowerBase):
    """CNN tower over disc/cup segmentation maps.

    image_data    : ImageDataView — provides get_image_path(*ids) and side_map.
                    Must implement build_seg_map_loader(source, **kwargs).
    backbone      : backbone key (see accessory/backbones.py)
    channels      : 1 (label map in [0,1]) or 3 (one-hot bg/rim/cup)
    target_size   : CNN input spatial size (cached arrays already at this size)
    augment       : random flip + 90° rotation at training time
    freeze_ratio  : fraction of early backbone blocks to freeze in [0, 1]
    seg_source    : passed to image_data.build_seg_map_loader (e.g. "gt", "unet")
    **seg_kwargs  : forwarded to build_seg_map_loader
    """

    def __init__(
        self,
        image_data,
        backbone:     str   = "resnet18",
        channels:     int   = 3,
        target_size:  int   = 224,
        augment:      bool  = True,
        freeze_ratio: float = 0.0,
        seg_source:   str   = "gt",
        **seg_kwargs: Any,
    ):
        super().__init__()
        self.image_data   = image_data
        self._channels    = channels
        self._target_size = target_size
        self._augment     = augment

        if not hasattr(image_data, "build_seg_map_loader"):
            raise TypeError(
                f"GeometrySegEncoder requires image_data to implement "
                f"build_seg_map_loader(), but {type(image_data).__name__} does not."
            )
        loader_kwargs = {
            "channels":     channels,
            "target_size":  target_size,
            **seg_kwargs,
        }
        self._loader = image_data.build_seg_map_loader(seg_source, **loader_kwargs)
        self._seg_cache: dict = {}
        self._seg_source = seg_source

        self.backbone, self._base_dim, self._blocks = build_backbone(backbone, freeze_ratio)
        if channels != 3:
            self._adapt_first_conv(channels)

        print(
            f"[GeometrySegEncoder] backbone={backbone}  channels={channels}  "
            f"target_size={target_size}  seg_source={seg_source}",
            flush=True,
        )

    # ── TowerBase interface ──────────────────────────────────────────────────

    @property
    def out_dim(self) -> int:
        return self._base_dim

    @property
    def _side_map(self) -> dict[str, str]:
        return self.image_data.side_map

    def _get(self, *ids) -> torch.Tensor:
        key = tuple(ids)
        arr = self._seg_cache.get(key)
        if arr is None:
            arr = np.zeros(
                (self._channels, self._target_size, self._target_size),
                dtype=np.float32,
            )
        if self.training and self._augment:
            arr = self._augment_array(arr)
        return torch.from_numpy(np.ascontiguousarray(arr))

    # ── EPC early_pass ───────────────────────────────────────────────────────

    def early_pass(self, context) -> None:
        data   = context.require("data")
        split  = context.require("split")

        train_samples = self._collect_samples(split.train, data)
        all_samples   = self._collect_samples(split.train, data)
        all_samples  += self._collect_samples(split.val,   data)
        if split.test is not None:
            all_samples += self._collect_samples(split.test, data)

        # Reset per-fold state if loader supports it (UNet only).
        if hasattr(self._loader, "reset_cache"):
            self._loader.reset_cache()
        if hasattr(self._loader, "reset_weights"):
            self._loader.reset_weights()
        if hasattr(self._loader, "finetune"):
            self._loader.finetune(train_samples)

        self._loader.precompute(all_samples)
        self._seg_cache = self._loader.all_seg_maps()
        print(
            f"[GeometrySegEncoder] cached {len(self._seg_cache)} seg maps for fold",
            flush=True,
        )

    # ── nn.Module forward ────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.backbone(x)
        if y.dim() > 2:
            y = y.flatten(1)
        return y

    # ── Utilities ────────────────────────────────────────────────────────────

    def set_freeze_ratio(self, ratio: float) -> None:
        r        = max(0.0, min(1.0, float(ratio)))
        n_freeze = int(math.floor(len(self._blocks) * r))
        for b in self._blocks:
            for p in b.parameters():
                p.requires_grad = True
        for b in self._blocks[:n_freeze]:
            for p in b.parameters():
                p.requires_grad = False

    # ── Internals ────────────────────────────────────────────────────────────

    def _collect_samples(self, df, data) -> list:
        """Build (pid, eye, image_path) tuples from a split DataFrame."""
        if df is None or len(df) == 0:
            return []
        pc = data.patient_col
        out = []
        for _, row in df.iterrows():
            pid = int(row[pc])
            eye = str(row.get("eyeID", "OD"))
            out.append((pid, eye, data.image.get_image_path(pid, eye)))
        return out

    @staticmethod
    def _augment_array(arr: np.ndarray) -> np.ndarray:
        """Random flip + 90° rotation on a (C, H, W) seg-map array."""
        if np.random.rand() < 0.5:
            arr = arr[:, :, ::-1]
        if np.random.rand() < 0.5:
            arr = arr[:, ::-1, :]
        k = int(np.random.randint(0, 4))
        if k:
            arr = np.rot90(arr, k=k, axes=(1, 2))
        return arr

    def _adapt_first_conv(self, in_channels: int) -> None:
        """Replace the first Conv2d to accept a non-3-channel input.

        Pretrained weights are averaged across the original input channels and
        broadcast across the new ones.
        """
        first = self._find_first_conv(self.backbone)
        new   = nn.Conv2d(
            in_channels,
            first.out_channels,
            kernel_size=first.kernel_size,
            stride=first.stride,
            padding=first.padding,
            bias=first.bias is not None,
        )
        with torch.no_grad():
            new.weight.copy_(
                first.weight.mean(dim=1, keepdim=True).expand_as(new.weight)
            )
            if first.bias is not None:
                new.bias.copy_(first.bias)
        self._replace_first_conv(self.backbone, new)

    @staticmethod
    def _find_first_conv(module: nn.Module) -> nn.Conv2d:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                return m
        raise RuntimeError("No Conv2d found in backbone")

    @classmethod
    def _replace_first_conv(cls, module: nn.Module, new_conv: nn.Conv2d) -> bool:
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                setattr(module, name, new_conv)
                return True
            if cls._replace_first_conv(child, new_conv):
                return True
        return False
