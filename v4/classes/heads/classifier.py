"""classifier — ClassificationHead output head."""
from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Minimal classification head: ReLU → Dropout → Linear(in_dim → num_classes).

    Used as the output stage of bridges and any module that needs a reusable,
    swappable task head producing class logits.
    """

    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)
