"""gated_bridge — GatedAdditiveBridge.

Per-sample, per-stream learned gates determine how much each stream contributes
to the fused embedding.  Each gate is a sigmoid scalar produced by an MLP over
the raw input embeddings, so the gate is conditioned on the actual content of
all streams — when a stream's signal is weak for a particular sample, its gate
can attenuate toward 0; when it's informative, gate goes toward 1.

  h = Σ_i  g_i(x) · LN_i(W_i z_i)
  g_i(x) = σ(MLP_i([z_1, z_2, …, z_N]))

No symmetry-breaking between streams — every tower is treated identically; the
gate network decides per-sample which to amplify.  Gates use sigmoid (not
softmax) so they can be independently small or large; the model isn't forced
into a "pick one" distribution.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from v4.classes.accessory.se_block import SEBlock


class GatedAdditiveBridge(nn.Module):
    """Per-sample sigmoid-gated additive fusion.

    Parameters
    ----------
    input_dims    : ordered list of input embedding dims
    fusion_dim    : projection / output dimension
    gate_hidden   : hidden width of the gating MLP (default: fusion_dim)
    use_ln        : LayerNorm after each per-stream projection (default: True)
    use_se        : SE gate on the fused vector
    se_reduction  : SE bottleneck factor
    """

    def __init__(
        self,
        input_dims:   list[int],
        fusion_dim:   int  = 256,
        gate_hidden:  int  = 128,
        use_ln:       bool = True,
        use_se:       bool = True,
        se_reduction: int  = 16,
    ):
        super().__init__()
        self.out_dim = fusion_dim
        self.n      = len(input_dims)
        self.W  = nn.ModuleList([nn.Linear(d, fusion_dim) for d in input_dims])
        self.ln = nn.ModuleList(
            [nn.LayerNorm(fusion_dim) if use_ln else nn.Identity()
             for _ in input_dims]
        )
        self.gate = nn.Sequential(
            nn.Linear(sum(input_dims), gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, self.n),
            nn.Sigmoid(),
        )
        self.se = SEBlock(fusion_dim, reduction=se_reduction, residual=True) if use_se else None

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        assert len(embeddings) == self.n, (
            f"GatedAdditiveBridge expects {self.n} inputs, got {len(embeddings)}"
        )
        projected = [self.ln[i](self.W[i](e)) for i, e in enumerate(embeddings)]
        gate_in   = torch.cat(embeddings, dim=-1)
        gates     = self.gate(gate_in)              # (B, N) ∈ (0, 1)
        h = 0
        for i in range(self.n):
            h = h + gates[..., i:i+1] * projected[i]
        if self.se is not None:
            h, _ = self.se(h)
        return h

    def set_phase(self, phase: str) -> None:
        enabled = phase not in ("tower_warmup", "cd_warmup")
        for p in self.parameters():
            p.requires_grad_(enabled)
