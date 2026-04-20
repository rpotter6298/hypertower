"""htfusion — HTFusion: N named towers fused through a FusionBridge."""
from __future__ import annotations

import torch
import torch.nn as nn

from v4.classes.bridges.fusion_bridge import FusionBridge
from v4.classes.vehicles.htbase import HTBase


class HTFusion(HTBase):
    """General N-tower fusion vehicle.

    Each named encoder is registered as a submodule; the FusionBridge
    projects and Hadamard-fuses their embeddings.

    Parameters
    ----------
    towers      : ordered dict ``{name: encoder}``.  Each encoder must
                  expose ``.out_dim``.
    num_classes : output classes
    fusion_dim  : bridge projection dimensionality
    dropout     : bridge dropout
    use_se      : SE gate on the fused vector

    Forward contract
    ----------------
    ``forward(embeddings)`` takes a ``dict[str, Tensor]`` of pre-computed
    per-tower embeddings and returns ``(logits_fused, aux_dict)`` where
    ``aux_dict`` maps each tower name to its auxiliary head logits.
    """

    def __init__(
        self,
        towers:      dict[str, nn.Module],
        num_classes: int,
        fusion_dim:  int   = 256,
        dropout:     float = 0.5,
        use_se:      bool  = False,
    ):
        super().__init__()
        self.towers = nn.ModuleDict(towers)
        self.bridge = FusionBridge(
            tower_dims=[t.out_dim for t in self.towers.values()],
            num_classes=num_classes,
            fusion_dim=fusion_dim,
            dropout=dropout,
            use_se=use_se,
        )

    def encode(self, embeddings: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return z_fused (pre-classifier) from a dict of per-tower embeddings."""
        return self.bridge.encode([embeddings[name] for name in self.towers])

    def forward(
        self,
        embeddings: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ordered = [embeddings[name] for name in self.towers]
        logits, aux = self.bridge.fuse(ordered)
        return logits, {name: aux[i] for i, name in enumerate(self.towers)}
