"""image_tower — ImageEncoder for v4.

Self-contained: no v3 dependencies.
Inherits get_sample dispatch from TowerBase.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from v4.classes.towerbase import TowerBase
from v4.classes.accessory.backbones import build_backbone
from v4.classes.accessory.se_block import SEBlock
from v4.classes.accessory.transforms import build_backbone_transform, build_eval_transform


class ImageEncoder(TowerBase):
    """Vision backbone → pooled feature vector.

    image_data   : ImageDataView — provides load_image(*ids) and side_map
    backbone     : backbone key (see accessory/backbones.py)
    freeze_ratio : fraction of early blocks to freeze in [0, 1]
    use_se       : apply SE attention over the pooled feature vector
    augment      : include random flip/rotation/jitter in the train transform
    """

    def __init__(
        self,
        image_data,
        backbone:     str   = "efficientnet_b0",
        freeze_ratio: float = 0.0,
        use_se:       bool  = False,
        se_reduction: int   = 16,
        se_pre_norm:  bool  = True,
        augment:      bool  = True,
    ):
        super().__init__()
        self.image_data     = image_data
        self._name          = backbone
        self.backbone, self._base_dim, self._blocks = build_backbone(backbone, freeze_ratio)
        self.transform      = build_backbone_transform(backbone, augment=augment)
        self.eval_transform = build_eval_transform(backbone)

        self.tower_ln = nn.LayerNorm(self._base_dim) if se_pre_norm else nn.Identity()
        self.tower_se = SEBlock(self._base_dim, reduction=se_reduction, residual=True) if use_se else None

    # ── TowerBase interface ──────────────────────────────────────────────────

    @property
    def out_dim(self) -> int:
        return self._base_dim

    @property
    def _side_map(self) -> dict[str, str]:
        return self.image_data.side_map

    def _get(self, *ids) -> torch.Tensor:
        img = self.image_data.load_image(*ids)
        t   = self.transform if self.training else self.eval_transform
        return t(img)

    # ── nn.Module forward ────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.backbone(x)
        if self.tower_se is not None:
            y, _ = self.tower_se(self.tower_ln(y))
        return y

    # ── Utilities ────────────────────────────────────────────────────────────

    def set_freeze_ratio(self, ratio: float) -> None:
        """Dynamically freeze the earliest floor(N * ratio) backbone blocks."""
        r        = max(0.0, min(1.0, float(ratio)))
        n_freeze = int(math.floor(len(self._blocks) * r))
        for b in self._blocks:
            for p in b.parameters():
                p.requires_grad = True
        for b in self._blocks[:n_freeze]:
            for p in b.parameters():
                p.requires_grad = False
