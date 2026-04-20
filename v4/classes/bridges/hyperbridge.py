"""hyperbridge — HyperBridge: bilateral fusion over paired embeddings, embedding output only."""
from __future__ import annotations

import torch
import torch.nn as nn


class HyperBridge(nn.Module):
    """Fuse side embeddings (e.g. two z_fused vectors) into a single embedding.

    Pure embedding producer — no classification head.  Attach a head stage in
    the pipeline config to produce logits.

    Modes
    -----
    embedding_mlp (default)
        Linear projection of concatenated inputs → hidden_dim embedding.

    classic_bridge
        Per-side projection → Hadamard product → hidden_dim embedding.

    Parameters
    ----------
    input_dims  : {side_key: dim} — e.g. {"a": 256, "b": 256}
    hidden_dim  : output embedding dimension
    mode        : "embedding_mlp" | "classic_bridge"
    """

    def __init__(
        self,
        input_dims: dict[str, int],
        hidden_dim: int = 256,
        mode:       str = "embedding_mlp",
    ):
        super().__init__()
        self.input_names = list(input_dims.keys())
        self.mode        = mode
        self.out_dim     = hidden_dim
        dims             = list(input_dims.values())

        if mode == "embedding_mlp":
            self.proj = nn.Linear(sum(dims), hidden_dim)
        elif mode == "classic_bridge":
            self.W  = nn.ModuleList([nn.Linear(d, hidden_dim) for d in dims])
            self.ln = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in dims])
        else:
            raise ValueError(f"Unknown HyperBridge mode: {mode!r}")

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        ordered = [inputs[name] for name in self.input_names]
        if self.mode == "embedding_mlp":
            return self.proj(torch.cat(ordered, dim=1))
        h = self.ln[0](self.W[0](ordered[0]))
        for i in range(1, len(ordered)):
            h = h * self.ln[i](self.W[i](ordered[i]))
        return h
