from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from torchvision import transforms

from classes.backbones import BACKBONES, list_names, load_backbone_weights
from classes.SE_attention import SEBlock
from classes.v2.data_bundle import DataBundle


def build_backbone(name: str, freeze_ratio: float = 0.0, augment: bool = True):
    """
    Operational builder:
      - instantiate with DEFAULT weights
      - strip classifier → features
      - apply ratio-based freezing over coarse blocks
      - return (model, out_dim, transform)
    """
    key = (name or "").lower()
    if key not in BACKBONES:
        raise ValueError(f"Unsupported backbone '{name}'. Valid options: {list_names()}")

    spec = BACKBONES[key]
    m = spec.ctor(weights=spec.weights_default)
    out_dim, m = spec.strip(m)
    load_backbone_weights(key, m)

    # transforms: use the weights’ mean/std, but keep your augmentation pipeline
    mean = getattr(spec.weights_default, "meta", {}).get("mean", (0.485, 0.456, 0.406))
    std = getattr(spec.weights_default, "meta", {}).get("std", (0.229, 0.224, 0.225))
    crop = 299 if key == "inception_v3" else 224

    if augment:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(crop),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(crop),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    # ratio-based freezing: freeze earliest floor(N * freeze_ratio) blocks
    fr = max(0.0, min(1.0, float(freeze_ratio)))
    blocks = spec.blocks(m)
    n = len(blocks)
    freeze_n = int(math.floor(n * fr))
    for b in blocks[:freeze_n]:
        for p in b.parameters():
            p.requires_grad = False

    return m, out_dim, transform


class ImageTower(nn.Module):
    """
    Vision backbone → pooled features.
    - backbone: one of list_names() (default 'efficientnet_b0')
    - always DEFAULT torchvision weights
    - freeze_ratio ∈ [0,1] freezes earliest floor(N*freeze_ratio) blocks
    - returns [N, out_dim] features from backbone forward
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        freeze_ratio: float = 0.0,
        use_se: bool = False,
        se_reduction: int = 16,
        se_pre_norm: bool = True,
        augment: bool = True,
        geometry_dim: int = 0,
    ):
        super().__init__()
        self.backbone, base_dim, self.transform = build_backbone(
            backbone, freeze_ratio, augment=augment
        )
        self._name = backbone
        # Keep ordered blocks for dynamic freezing/thawing
        key = (self._name or "").lower()
        self._spec = BACKBONES[key]
        self._blocks = self._spec.blocks(self.backbone)
        # Optional tower-level SE over the final feature vector
        self.base_dim = base_dim
        self.geometry_dim = max(0, int(geometry_dim))
        self.out_dim = self.base_dim + self.geometry_dim
        self.tower_ln = nn.LayerNorm(self.base_dim) if se_pre_norm else nn.Identity()
        self.tower_se = (
            SEBlock(self.base_dim, reduction=se_reduction, residual=True)
            if use_se
            else None
        )

    def forward(
        self, x: torch.Tensor, geometry: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        y = self.backbone(x)
        # sanity: pooled features, not logits
        assert y.dim() == 2 and y.size(1) == self.base_dim, (
            f"Expected features [N,{self.base_dim}], got {tuple(y.shape)}"
        )
        if self.tower_se is not None:
            y, _ = self.tower_se(self.tower_ln(y))
        if self.geometry_dim > 0:
            if geometry is None or geometry.numel() == 0:
                geom = torch.zeros(
                    y.size(0), self.geometry_dim, device=y.device, dtype=y.dtype
                )
            else:
                if geometry.dim() == 1:
                    geom = geometry.unsqueeze(0)
                else:
                    geom = geometry
                geom = geom.to(device=y.device, dtype=y.dtype)
                if geom.size(0) != y.size(0):
                    raise ValueError(
                        f"Geometry batch size mismatch: {geom.size(0)} vs {y.size(0)}"
                    )
                if geom.size(1) != self.geometry_dim:
                    raise ValueError(
                        f"Expected geometry dim {self.geometry_dim}, got {geom.size(1)}"
                    )
            y = torch.cat([y, geom], dim=1)
        return y

    def set_freeze_ratio(self, ratio: float):
        """Dynamically freeze earliest floor(N*ratio) backbone blocks."""
        r = max(0.0, min(1.0, float(ratio)))
        n = len(self._blocks)
        freeze_n = int(math.floor(n * r))
        # Unfreeze all first
        for b in self._blocks:
            for p in b.parameters():
                p.requires_grad = True
        # Freeze earliest blocks
        for b in self._blocks[:freeze_n]:
            for p in b.parameters():
                p.requires_grad = False


class SiameseImageTower(nn.Module):
    """
    Shared-weight bilateral image tower.

    Runs OD and OS images through a single shared backbone, then returns
    cat([f_mean, f_delta]) where:
        f_mean  = (f_od + f_os) / 2   -- shared bilateral representation
        f_delta = f_od - f_os          -- asymmetry, signed OD-relative

    out_dim = 2 * backbone_out_dim

    When x_os is None (single-eye fallback):
        f_mean  = f_od
        f_delta = zeros
    so the module degrades gracefully when only one eye is available.

    The shared backbone means both eyes contribute to every gradient update,
    effectively doubling the training signal for the visual pathway without
    doubling parameters.
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        freeze_ratio: float = 0.0,
        use_se: bool = False,
        se_reduction: int = 16,
        se_pre_norm: bool = True,
        augment: bool = True,
    ):
        super().__init__()
        self._tower = ImageTower(
            backbone=backbone,
            freeze_ratio=freeze_ratio,
            use_se=use_se,
            se_reduction=se_reduction,
            se_pre_norm=se_pre_norm,
            augment=augment,
            geometry_dim=0,
        )
        self.out_dim = self._tower.out_dim * 2
        self.transform = self._tower.transform

    def forward(
        self,
        x_od: torch.Tensor,
        x_os: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        f_od = self._tower(x_od)
        if x_os is None:
            f_mean = f_od
            f_delta = torch.zeros_like(f_od)
        else:
            f_os = self._tower(x_os)
            f_mean = (f_od + f_os) * 0.5
            f_delta = f_od - f_os
        return torch.cat([f_mean, f_delta], dim=1)

    def set_freeze_ratio(self, ratio: float) -> None:
        """Delegates to the shared inner tower."""
        self._tower.set_freeze_ratio(ratio)


class MDTower(nn.Module):
    """MLP over DataBundle.vectorize_row outputs (convert to torch inside tower)."""

    def __init__(
        self,
        clinical_data: DataBundle,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_se: bool = False,
        se_reduction: int = 16,
        se_pre_norm: bool = True,
    ):
        super().__init__()
        self.feature_dim = clinical_data.feature_dim
        self.out_dim = hidden_dim
        # two-block MLP so we can optionally freeze/thaw per block
        self.block0 = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.net = nn.Sequential(self.block0, self.block1)
        self.tower_ln = nn.LayerNorm(hidden_dim) if se_pre_norm else nn.Identity()
        self.tower_se = (
            SEBlock(hidden_dim, reduction=se_reduction, residual=True)
            if use_se
            else None
        )

    def forward(self, meta_np_or_torch) -> torch.Tensor:
        if isinstance(meta_np_or_torch, torch.Tensor):
            x = meta_np_or_torch
        else:
            x = torch.as_tensor(meta_np_or_torch, dtype=torch.float32)
        h = self.net(x)
        if self.tower_se is not None:
            h, _ = self.tower_se(self.tower_ln(h))
        return h

    def set_freeze_ratio(self, ratio: float):
        """Optionally freeze earliest blocks of the MLP."""
        r = max(0.0, min(1.0, float(ratio)))
        # Unfreeze all
        for p in self.block0.parameters():
            p.requires_grad = True
        for p in self.block1.parameters():
            p.requires_grad = True
        # Freeze earliest blocks based on ratio threshold
        if r >= 0.5:
            for p in self.block0.parameters():
                p.requires_grad = False
        if r >= 1.0:
            for p in self.block1.parameters():
                p.requires_grad = False
