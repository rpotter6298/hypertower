"""htlateral — HTLateral: shared encoder over N same-type inputs."""
from __future__ import annotations

import torch
import torch.nn as nn

from v4.classes.vehicles.htbase import HTBase


class HTLateral(HTBase):
    """N same-type inputs through a shared encoder, jointly compressed, then classified.

    All inputs share the same encoder weights (one forward pass per input).
    The joint MLP compresses the concatenated embeddings before classification.

    Aux heads provide per-input logits before the joint MLP — useful for
    BCD-style training.

    Parameters
    ----------
    encoder     : shared encoder module with ``.out_dim``
    input_names : ordered slot names (e.g. ``["od", "os"]``)
    num_classes : output classes
    fusion_dim  : joint MLP hidden dim
    dropout     : dropout in MLP and classifier
    """

    def __init__(
        self,
        encoder:     nn.Module,
        input_names: list[str],
        num_classes: int,
        fusion_dim:  int   = 256,
        dropout:     float = 0.5,
    ):
        super().__init__()
        self.encoder     = encoder
        self.input_names = list(input_names)
        n       = len(input_names)
        in_dim: int = encoder.out_dim  # type: ignore[assignment]

        self.joint = nn.Sequential(
            nn.Linear(n * in_dim, fusion_dim), nn.LayerNorm(fusion_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(fusion_dim, in_dim),
        )
        self.aux_heads = nn.ModuleList([
            nn.Linear(in_dim, num_classes) for _ in range(n)
        ])
        self.head = nn.Sequential(
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(in_dim, num_classes),
        )

    def encode(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return joint embedding (post-MLP, pre-classifier)."""
        zs = [self.encoder(inputs[name]) for name in self.input_names]
        return self.joint(torch.cat(zs, dim=1))

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zs      = [self.encoder(inputs[name]) for name in self.input_names]
        z_joint = self.joint(torch.cat(zs, dim=1))
        logits  = self.head(z_joint)
        aux     = {name: head(z)
                   for name, head, z in zip(self.input_names, self.aux_heads, zs)}
        return logits, aux
