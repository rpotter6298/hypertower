"""clinical_towers — ClinicalEncoder and ClinicalDataTower.

Self-contained: defines ClinicalEncoder directly (does not import it from
towers.py).  Imports only TowerBase from towerbase plus infrastructure
(SEBlock, DataBundle).
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from v3.classes.towerbase import TowerBase
from v3.classes.SE_attention import SEBlock
from v3.classes.data_bundle import DataBundle


# ---------------------------------------------------------------------------
# ClinicalEncoder — MLP over tabular features
# ---------------------------------------------------------------------------

class ClinicalEncoder(nn.Module):
    """MLP over DataBundle.vectorize_row outputs (converts to torch inside tower)."""

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
        # Two-block MLP so we can optionally freeze/thaw per block.
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
        for p in self.block0.parameters():
            p.requires_grad = True
        for p in self.block1.parameters():
            p.requires_grad = True
        if r >= 0.5:
            for p in self.block0.parameters():
                p.requires_grad = False
        if r >= 1.0:
            for p in self.block1.parameters():
                p.requires_grad = False


# ---------------------------------------------------------------------------
# ClinicalDataTower — TowerBase implementation
# ---------------------------------------------------------------------------

class ClinicalDataTower(TowerBase, nn.Module):
    """
    TowerBase implementation for the clinical metadata modality.

    Wraps ClinicalEncoder (MLP over tabular features).
    Contributes one embedding per eye slot: [z_cd].

    Implements ``cd_warmup_embedding`` so train_towers_epoch can identify
    this tower for cd_warmup phase via duck typing rather than isinstance checks.
    """

    def __init__(
        self,
        *,
        clinical_data,
        cd_hidden_dim: int = 128,
        cd_dropout: float = 0.1,
        use_se: bool = False,
    ):
        nn.Module.__init__(self)
        self._encoder = ClinicalEncoder(
            clinical_data=clinical_data,
            hidden_dim=cd_hidden_dim,
            dropout=cd_dropout,
            use_se=use_se,
        )

    @property
    def out_dim(self) -> int:
        return self._encoder.out_dim

    @property
    def embed_dims(self) -> list[int]:
        return [self._encoder.out_dim]

    def set_phase(self, phase: str) -> None:
        enabled = phase not in ("fused_warmup",)
        for p in self._encoder.parameters():
            p.requires_grad = enabled

    def cd_warmup_embedding(
        self,
        batch: dict,
        *,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Return clinical embedding for slot 1, or None if matrix_1 is absent."""
        m = batch.get("matrix_1")
        if not torch.is_tensor(m):
            return None
        return self._encoder(m.to(device))

    def embed_batch(
        self,
        batch: dict,
        *,
        device: torch.device,
        slot: int = 1,
    ) -> list[torch.Tensor]:
        m = batch.get(f"matrix_{slot}")
        if not torch.is_tensor(m):
            raise ValueError(f"ClinicalDataTower.embed_batch: matrix_{slot} is missing or not a tensor")
        return [self._encoder(m.to(device))]

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
        pass  # shares loader with ImageTower; no per-fold setup needed
