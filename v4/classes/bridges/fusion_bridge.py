"""fusion_bridge — FusionBridge: N-input element-wise fusion, embedding output only."""
from __future__ import annotations

import torch
import torch.nn as nn

from v4.classes.accessory.se_block import SEBlock


class FusionBridge(nn.Module):
    """Project N input embeddings to a shared dim, fuse element-wise.

    Pure embedding producer — no classification head.  Attach a head stage in
    the pipeline config to produce logits.

    Two fusion modes:
      * additive=False (Hadamard): h = ∏_i  ln_i(W_i z_i)
                                   Interaction term only.  A near-zero factor
                                   in any stream silences that dimension for
                                   the whole bridge.
      * additive=True  (shifted) : h = ∏_i (1 + ln_i(W_i z_i)) - 1
                                   Expands to Σ_i a_i + cross-terms (sums of
                                   products).  A silent stream (≈0) reduces to
                                   identity on its factor, so other streams'
                                   contributions survive unchanged.

    Parameters
    ----------
    input_dims   : ordered list of input embedding dims
    fusion_dim   : projection / output dimension
    additive     : if True, use the shifted-multiply form (default: False)
    use_se       : SE gate on the fused vector
    se_reduction : SE reduction factor
    se_pre_norm  : LayerNorm before each projection; else Identity
    """

    def __init__(
        self,
        input_dims:   list[int],
        fusion_dim:   int  = 256,
        additive:     bool = False,
        use_se:       bool = True,
        se_reduction: int  = 16,
        se_pre_norm:  bool = True,
    ):
        super().__init__()
        self.out_dim  = fusion_dim
        self.additive = additive
        self.W  = nn.ModuleList([nn.Linear(d, fusion_dim) for d in input_dims])
        self.ln = nn.ModuleList(
            [nn.LayerNorm(fusion_dim) if se_pre_norm else nn.Identity()
             for _ in input_dims]
        )
        self.se = SEBlock(fusion_dim, reduction=se_reduction, residual=True) if use_se else None

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        assert len(embeddings) == len(self.W), (
            f"FusionBridge expects {len(self.W)} inputs, got {len(embeddings)}"
        )
        if self.additive:
            h = 1.0 + self.ln[0](self.W[0](embeddings[0]))
            for i in range(1, len(embeddings)):
                h = h * (1.0 + self.ln[i](self.W[i](embeddings[i])))
            h = h - 1.0
        else:
            h = self.ln[0](self.W[0](embeddings[0]))
            for i in range(1, len(embeddings)):
                h = h * self.ln[i](self.W[i](embeddings[i]))
        if self.se is not None:
            h, _ = self.se(h)
        return h

    def set_phase(self, phase: str) -> None:
        """Freeze bridge during tower_warmup; trainable otherwise."""
        enabled = phase not in ("tower_warmup", "cd_warmup")
        for p in self.parameters():
            p.requires_grad_(enabled)
