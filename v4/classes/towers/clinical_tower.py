"""clinical_tower — ClinicalEncoder for v4.

Self-contained: no v3 dependencies.
Inherits get_sample dispatch from TowerBase.

Geometry injection (EPC consumption)
--------------------------------------
When geom_dim > 0, ClinicalEncoder requests the "geometry_vectors" key from EPC
during early_pass and appends the geometry features to every clinical vector.
The input layer is sized to clinical_data.feature_dim + geom_dim automatically.

Config example (cd tower consuming geometry):
{
  "name":         "cd",
  "module":       "v4.classes.towers.clinical_tower",
  "class":        "ClinicalEncoder",
  "data_source":  "matrix",
  "epc_requests": ["geometry_vectors"],
  "args": {
    "hidden_dim": 128,
    "geom_dim":   5
  }
}
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from v4.classes.towerbase import TowerBase
from v4.classes.accessory.se_block import SEBlock


class ClinicalEncoder(TowerBase):
    """MLP over tabular clinical features, with optional geometry vector injection.

    clinical_data : ClinicalDataView — provides feature_dim, vectorize_entity, side_map
    hidden_dim    : output embedding dimensionality
    dropout       : applied after the first linear block
    use_se        : wrap output with SEBlock channel gating
    se_reduction  : SEBlock bottleneck factor
    se_pre_norm   : apply LayerNorm before SEBlock
    geom_dim      : number of geometry features to append from EPC (0 = disabled)
                    requires epc_requests: ["geometry_vectors"] in tower config
    """

    EPC_GEOMETRY_KEY = "geometry_vectors"

    def __init__(
        self,
        clinical_data,
        hidden_dim:  int   = 128,
        dropout:     float = 0.1,
        use_se:      bool  = False,
        se_reduction: int  = 16,
        se_pre_norm: bool  = True,
        geom_dim:    int   = 0,
    ):
        super().__init__()
        self.clinical_data  = clinical_data
        self._out_dim       = hidden_dim
        self._geom_dim      = geom_dim
        self._geom_vectors: dict | None = None  # filled by early_pass when geom_dim > 0
        feature_dim = clinical_data.feature_dim + geom_dim

        self.block0 = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
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
        self.tower_se = SEBlock(hidden_dim, reduction=se_reduction, residual=True) if use_se else None

    # ── EPC early_pass ───────────────────────────────────────────────────────

    def early_pass(self, context) -> None:
        if self._geom_dim > 0:
            self._geom_vectors = context.require(self.EPC_GEOMETRY_KEY)

    # ── TowerBase interface ──────────────────────────────────────────────────

    @property
    def out_dim(self) -> int:
        return self._out_dim

    @property
    def _side_map(self) -> dict[str, str]:
        return self.clinical_data.side_map

    def _get(self, *ids) -> torch.Tensor:
        arr = self.clinical_data.vectorize_entity(*ids)
        if self._geom_dim > 0 and self._geom_vectors is not None:
            pid  = int(ids[0])
            eye  = str(ids[1]) if len(ids) > 1 else "OD"
            geom = self._geom_vectors.get(
                (pid, eye),
                np.zeros(self._geom_dim, dtype=np.float32),
            )
            arr = np.concatenate([arr, geom[: self._geom_dim]])
        return torch.from_numpy(arr.astype(np.float32, copy=False))

    # ── nn.Module forward ────────────────────────────────────────────────────

    def forward(self, x) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        h = self.net(x)
        if self.tower_se is not None:
            h, _ = self.tower_se(self.tower_ln(h))
        return h

    # ── Utilities ────────────────────────────────────────────────────────────

    def set_freeze_ratio(self, ratio: float) -> None:
        """Freeze the earliest MLP block proportionally."""
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
