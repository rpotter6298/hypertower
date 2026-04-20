"""image_towers — ImageEncoder, SiameseImageTower, and ImageTower (TowerBase).

Self-contained image-modality tower layer.  No dependencies on other tower
files, bridge, or model classes.  Clear contract: accepts an image batch,
returns a fixed-size embedding vector.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from v3.classes.towerbase import TowerBase, build_backbone
from v3.classes.backbones import BACKBONES
from v3.classes.SE_attention import SEBlock


# ---------------------------------------------------------------------------
# ImageEncoder — vision backbone → pooled feature vector
# ---------------------------------------------------------------------------

class ImageEncoder(nn.Module):
    """Vision backbone → pooled feature vector.

    Wraps a torchvision backbone (default weights), strips the classifier,
    and optionally appends an SE attention block and/or a geometry vector.

    Parameters
    ----------
    backbone     : backbone key (see backbones.py)
    freeze_ratio : fraction of early blocks to freeze in [0, 1]
    use_se       : apply SE attention over the pooled feature vector
    augment      : include random flip/rotation/jitter in the transform
    geometry_dim : if > 0, concatenate a geometry vector of this length
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
        self._name   = backbone
        key          = (self._name or "").lower()
        self._spec   = BACKBONES[key]
        self._blocks = self._spec.blocks(self.backbone)
        self.base_dim     = base_dim
        self.geometry_dim = max(0, int(geometry_dim))
        self.out_dim      = self.base_dim + self.geometry_dim
        self.tower_ln = nn.LayerNorm(self.base_dim) if se_pre_norm else nn.Identity()
        self.tower_se = (
            SEBlock(self.base_dim, reduction=se_reduction, residual=True)
            if use_se else None
        )

    def forward(
        self, x: torch.Tensor, geometry: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        y = self.backbone(x)
        assert y.dim() == 2 and y.size(1) == self.base_dim, (
            f"Expected features [N,{self.base_dim}], got {tuple(y.shape)}"
        )
        if self.tower_se is not None:
            y, _ = self.tower_se(self.tower_ln(y))
        if self.geometry_dim > 0:
            if geometry is None or geometry.numel() == 0:
                geom = torch.zeros(y.size(0), self.geometry_dim, device=y.device, dtype=y.dtype)
            else:
                geom = geometry.unsqueeze(0) if geometry.dim() == 1 else geometry
                geom = geom.to(device=y.device, dtype=y.dtype)
                if geom.size(0) != y.size(0):
                    raise ValueError(f"Geometry batch size mismatch: {geom.size(0)} vs {y.size(0)}")
                if geom.size(1) != self.geometry_dim:
                    raise ValueError(f"Expected geometry dim {self.geometry_dim}, got {geom.size(1)}")
            y = torch.cat([y, geom], dim=1)
        return y

    def set_freeze_ratio(self, ratio: float) -> None:
        """Dynamically freeze earliest floor(N*ratio) backbone blocks."""
        r = max(0.0, min(1.0, float(ratio)))
        freeze_n = int(math.floor(len(self._blocks) * r))
        for b in self._blocks:
            for p in b.parameters():
                p.requires_grad = True
        for b in self._blocks[:freeze_n]:
            for p in b.parameters():
                p.requires_grad = False


# ---------------------------------------------------------------------------
# SiameseImageTower — shared-weight bilateral image encoder
# ---------------------------------------------------------------------------

class SiameseImageTower(nn.Module):
    """Shared-weight bilateral image tower.

    Runs OD and OS images through a single shared backbone and returns
    cat([f_mean, f_delta]) where:
        f_mean  = (f_od + f_os) / 2   — shared bilateral representation
        f_delta = f_od - f_os          — signed asymmetry (OD-relative)

    out_dim = 2 × backbone_out_dim.  When x_os is None the tower degrades
    gracefully: f_mean = f_od, f_delta = zeros.
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
        self._tower = ImageEncoder(
            backbone=backbone,
            freeze_ratio=freeze_ratio,
            use_se=use_se,
            se_reduction=se_reduction,
            se_pre_norm=se_pre_norm,
            augment=augment,
        )
        self.out_dim   = self._tower.out_dim * 2
        self.transform = self._tower.transform

    def forward(
        self,
        x_od: torch.Tensor,
        x_os: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        f_od = self._tower(x_od)
        if x_os is None:
            return torch.cat([f_od, torch.zeros_like(f_od)], dim=1)
        f_os = self._tower(x_os)
        return torch.cat([(f_od + f_os) * 0.5, f_od - f_os], dim=1)

    def set_freeze_ratio(self, ratio: float) -> None:
        self._tower.set_freeze_ratio(ratio)


# ---------------------------------------------------------------------------
# ImageTower — TowerBase implementation
# ---------------------------------------------------------------------------

class ImageTower(TowerBase, nn.Module):
    """TowerBase implementation for the fundus image modality.

    Wraps ImageEncoder (backbone → pooled features).
    Contributes one embedding per eye slot: [z_img].
    """

    def __init__(
        self,
        *,
        backbone: str,
        freeze_ratio: float = 0.0,
        augment: bool = True,
        use_se: bool = False,
    ):
        nn.Module.__init__(self)
        self._encoder      = ImageEncoder(backbone=backbone, freeze_ratio=freeze_ratio,
                                          augment=augment, use_se=use_se)
        self._train_loader = None

    @property
    def transform(self):
        return self._encoder.transform

    @property
    def out_dim(self) -> int:
        return self._encoder.out_dim

    @property
    def embed_dims(self) -> list[int]:
        return [self._encoder.out_dim]

    @property
    def total_epochs(self) -> int:
        return 0

    @property
    def train_loader(self) -> Optional[DataLoader]:
        return self._train_loader

    def set_phase(self, phase: str) -> None:
        enabled = phase not in ("cd_warmup", "fused_warmup")
        for p in self._encoder.parameters():
            p.requires_grad = enabled

    def embed_batch(
        self,
        batch: dict,
        *,
        device: torch.device,
        slot: int = 1,
    ) -> list[torch.Tensor]:
        x = batch.get(f"image_{slot}")
        if not torch.is_tensor(x):
            raise ValueError(f"ImageTower.embed_batch: image_{slot} missing or not a tensor")
        return [self._encoder(x.to(device))]

    def prepare_fold(
        self,
        *,
        eye_train,
        bilat_train,
        bilat_val,
        bilat_test,
        image_preprocessor,
        image_cache,
        device,
        args,
    ) -> None:
        from v3.classes.loader_factory import (
            build_balanced_sampler,
            filter_eye_samples,
            make_loader,
        )
        from v3.classes.profiles import build_papila_profile

        profile_eye = build_papila_profile(
            patient_col="Patient ID", label_col=args.label_col, sample_mode="eye"
        )
        slots_eye    = profile_eye.slot_descriptors()
        use_balanced = bool(getattr(args, "balanced_sampling", False))
        _persistent  = args.num_workers > 0
        loader_kw    = dict(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_cache=image_cache,
            persistent_workers=_persistent,
        )
        eye_samples = filter_eye_samples(eye_train)
        sampler     = build_balanced_sampler(eye_samples) if use_balanced else None
        self._train_loader = make_loader(
            eye_samples, slots_eye,
            image_transform=self.transform,
            image_preprocessor=image_preprocessor,
            shuffle=True, sampler=sampler, **loader_kw,
        )
