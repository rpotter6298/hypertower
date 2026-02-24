# classes/image_tower.py
from __future__ import annotations
import math
from typing import Optional

import torch
from torch import nn
from torchvision import transforms
from classes.backbones import BACKBONES, list_names, load_backbone_weights
from classes.SE_attention import SEBlock

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
    std  = getattr(spec.weights_default, "meta", {}).get("std",  (0.229, 0.224, 0.225))
    crop = 299 if key == "inception_v3" else 224

    if augment:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(crop),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

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
    def __init__(self, backbone: str = "efficientnet_b0", freeze_ratio: float = 0.0,
                 use_se: bool = False, se_reduction: int = 16, se_pre_norm: bool = True,
                 augment: bool = True, geometry_dim: int = 0):
        super().__init__()
        self.backbone, base_dim, self.transform = build_backbone(backbone, freeze_ratio, augment=augment)
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
        self.tower_se = SEBlock(self.base_dim, reduction=se_reduction, residual=True) if use_se else None

    def forward(self, x: torch.Tensor, geometry: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = self.backbone(x)
        # sanity: pooled features, not logits
        assert y.dim() == 2 and y.size(1) == self.base_dim, \
            f"Expected features [N,{self.base_dim}], got {tuple(y.shape)}"
        if self.tower_se is not None:
            y, _ = self.tower_se(self.tower_ln(y))
        if self.geometry_dim > 0:
            if geometry is None or geometry.numel() == 0:
                geom = torch.zeros(y.size(0), self.geometry_dim, device=y.device, dtype=y.dtype)
            else:
                if geometry.dim() == 1:
                    geom = geometry.unsqueeze(0)
                else:
                    geom = geometry
                geom = geom.to(device=y.device, dtype=y.dtype)
                if geom.size(0) != y.size(0):
                    raise ValueError(f"Geometry batch size mismatch: {geom.size(0)} vs {y.size(0)}")
                if geom.size(1) != self.geometry_dim:
                    raise ValueError(f"Expected geometry dim {self.geometry_dim}, got {geom.size(1)}")
            y = torch.cat([y, geom], dim=1)
        return y

    def set_freeze_ratio(self, ratio: float):
        """Dynamically freeze earliest floor(N*ratio) backbone blocks.
        ratio in [0,1]."""
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
