"""mono_bridge — MonoBridge: passthrough for single-tower fusion stages.

The v4 stage runner always expects tower → bridge → head.  For configs that
have only one tower feeding a head, MonoBridge is the no-op bridge that lets
the architecture be "head sits directly on tower" without any extra projection,
SE, or fusion logic.

Optional LayerNorm is exposed for consistency with FusionBridge but defaults
off to keep the embedding numerically identical to the tower's output.
"""
from __future__ import annotations

import torch
from torch import nn


class MonoBridge(nn.Module):
    """Single-input passthrough bridge.

    Parameters
    ----------
    input_dims : list[int] — must be length 1
    use_ln     : if True, wrap the embedding in a LayerNorm
    """

    def __init__(self, input_dims: list[int], use_ln: bool = False):
        super().__init__()
        if len(input_dims) != 1:
            raise ValueError(
                f"MonoBridge expects exactly 1 input dim, got {len(input_dims)}"
            )
        self.out_dim = input_dims[0]
        self.ln      = nn.LayerNorm(self.out_dim) if use_ln else nn.Identity()

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        if len(embeddings) != 1:
            raise ValueError(
                f"MonoBridge forward expects 1 embedding, got {len(embeddings)}"
            )
        return self.ln(embeddings[0])

    def set_phase(self, phase: str) -> None:
        """Freeze during tower_warmup; trainable otherwise (matches FusionBridge)."""
        enabled = phase not in ("tower_warmup", "cd_warmup")
        for p in self.parameters():
            p.requires_grad_(enabled)
