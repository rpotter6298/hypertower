# md_tower.py
import torch
import torch.nn as nn
from classes import ClinicalData
from classes.SE_attention import SEBlock

class MDTower(nn.Module):
    """MLP over ClinicalData.vectorize_row outputs (convert to torch inside tower)."""
    def __init__(self, clinical_data: ClinicalData, hidden_dim: int = 128, dropout: float = 0.1,
                 use_se: bool = False, se_reduction: int = 16, se_pre_norm: bool = True):
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
        self.tower_se = SEBlock(hidden_dim, reduction=se_reduction, residual=True) if use_se else None

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
        """Optionally freeze earliest blocks of the MLP.
        With two blocks, ratio≥0.5 freezes block0; ratio≥1.0 freezes both."""
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
