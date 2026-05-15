"""pairwise_bridge — PairwiseAdditiveBridge.

For N input streams, compute the shifted-multiply ((1+a)(1+b) - 1) fusion for
every pair, then mix them with learnable per-pair scalar weights.

Motivation: the basic additive form (1+a)(1+b)...(1+N) - 1 helps for 2 streams
but regresses for 3+ because the triple-and-higher cross-terms (abc, abcd, …)
have explosive variance.  This bridge keeps only the 2-way interactions —
O(N²) pair-experts rather than O(2^N) cross-terms — and lets the model learn
which pairs matter.

For 2 streams this reduces to a single weighted (1+a)(1+b)-1 → equivalent (up
to the scaling factor) to FusionBridge(additive=True).
"""
from __future__ import annotations

from itertools import combinations

import torch
import torch.nn as nn

from v4.classes.accessory.se_block import SEBlock


class PairwiseAdditiveBridge(nn.Module):
    """Sum of per-pair shifted-multiplies.

    Parameters
    ----------
    input_dims   : ordered list of input embedding dims
    fusion_dim   : projection / output dimension
    use_ln       : LayerNorm after each per-stream projection (default: True)
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
        self.W  = nn.ModuleList([nn.Linear(d, fusion_dim) for d in input_dims])
        self.ln = nn.ModuleList(
            [nn.LayerNorm(fusion_dim) if use_ln else nn.Identity()
             for _ in input_dims]
        )
        n_pairs = max(1, len(input_dims) * (len(input_dims) - 1) // 2)
        # initialise to uniform mixing so each pair contributes equally at start
        self.pair_weights = nn.Parameter(torch.full((n_pairs,), 1.0 / n_pairs))
        self.se = SEBlock(fusion_dim, reduction=se_reduction, residual=True) if use_se else None

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        assert len(embeddings) == len(self.W), (
            f"PairwiseAdditiveBridge expects {len(self.W)} inputs, got {len(embeddings)}"
        )
        projected = [self.ln[i](self.W[i](e)) for i, e in enumerate(embeddings)]
        if len(projected) == 1:
            h = projected[0]
        else:
            pairs = list(combinations(range(len(projected)), 2))
            h     = 0
            for idx, (i, j) in enumerate(pairs):
                pair_fusion = (1.0 + projected[i]) * (1.0 + projected[j]) - 1.0
                h = h + self.pair_weights[idx] * pair_fusion
        if self.se is not None:
            h, _ = self.se(h)
        return h

    def set_phase(self, phase: str) -> None:
        enabled = phase not in ("tower_warmup", "cd_warmup")
        for p in self.parameters():
            p.requires_grad_(enabled)
