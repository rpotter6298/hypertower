"""htmono — HTMono: single tower + ClassificationHead, no bridge."""
from __future__ import annotations

import torch
import torch.nn as nn

from v4.classes.heads.classifier import ClassificationHead
from v4.classes.vehicles.htbase import HTBase


class HTMono(HTBase):
    """Single-tower vehicle: tower embedding fed directly into a ClassificationHead.

    No bridge or projection — the tower's output goes straight to
    ReLU → Dropout → Linear.  Returns ``(logits, {})`` from ``forward``
    to match the HTFusion / HTLateral interface.

    Parameters
    ----------
    tower       : encoder module with ``.out_dim``
    num_classes : output classes
    dropout     : dropout before the output linear layer
    """

    def __init__(
        self,
        tower:       nn.Module,
        num_classes: int,
        dropout:     float = 0.5,
    ):
        super().__init__()
        self.tower = tower
        self.head  = ClassificationHead(tower.out_dim, num_classes, dropout)  # type: ignore[arg-type]

    def encode(self, inputs) -> torch.Tensor:
        """Return tower embedding (pre-classifier)."""
        if isinstance(inputs, dict):
            # single-entry dict from HTDataset eye-level pass
            (z,) = inputs.values()
            return self.tower(z) if torch.is_tensor(z) else self.tower(*z.values())
        return self.tower(inputs)

    def forward(self, inputs) -> tuple[torch.Tensor, dict]:
        return self.head(self.encode(inputs)), {}
