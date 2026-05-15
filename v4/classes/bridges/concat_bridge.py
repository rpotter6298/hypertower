"""concat_bridge — ConcatBridge: N-input concatenate + linear fusion.

The simplest possible fusion: glue all input embeddings end-to-end and let a
single Linear layer learn the mixing.  No multiplicative interactions, no
zero-collapse risk, no cross-term explosion as N grows.

Useful as a baseline against the multiplicative bridges (FusionBridge,
PairwiseAdditiveBridge): if this gets within noise of them, then multiplicative
fusion isn't actually buying us anything.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from v4.classes.accessory.se_block import SEBlock


class ConcatBridge(nn.Module):
    """Concatenate N input embeddings, project down to fusion_dim.

    Parameters
    ----------
    input_dims   : ordered list of input embedding dims
    fusion_dim   : output dimension after the linear projection
    use_ln       : LayerNorm after the projection (default: True)
    use_se       : SE gate on the fused vector
    se_reduction : SE bottleneck factor
    """

    def __init__(
        self,
        input_dims:   list[int],
        fusion_dim:   int  = 256,
        use_ln:       bool = True,
        use_se:       bool = True,
        se_reduction: int  = 16,
    ):
        super().__init__()
        self.out_dim = fusion_dim
        self.proj    = nn.Linear(sum(input_dims), fusion_dim)
        self.ln      = nn.LayerNorm(fusion_dim) if use_ln else nn.Identity()
        self.se      = SEBlock(fusion_dim, reduction=se_reduction, residual=True) if use_se else None

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        h = torch.cat(embeddings, dim=-1)
        h = self.ln(self.proj(h))
        if self.se is not None:
            h, _ = self.se(h)
        return h

    def set_phase(self, phase: str) -> None:
        enabled = phase not in ("tower_warmup", "cd_warmup")
        for p in self.parameters():
            p.requires_grad_(enabled)
