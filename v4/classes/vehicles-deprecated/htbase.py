"""htbase — HTBase: abstract base for all v4 vehicle classes."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class HTBase(nn.Module, ABC):
    """Shared interface for all HyperTower vehicles.

    Subclasses must implement ``encode`` and ``forward``.

    ``transform`` walks ``self.towers`` (if present) and returns the transform
    from the first tower that exposes one — used by data loaders.

    ``forward`` contract: returns ``(logits, aux_dict)`` where
    ``aux_dict`` maps a name or index to per-component logits.
    HTMono returns an empty dict to keep the signature uniform.
    """

    @abstractmethod
    def encode(self, inputs) -> torch.Tensor:
        """Return the pre-classifier embedding."""

    @abstractmethod
    def forward(self, inputs) -> tuple[torch.Tensor, dict]:
        """Return (logits, aux_dict)."""

    @property
    def transform(self):
        towers = getattr(self, "towers", None) or {}
        for t in (towers.values() if hasattr(towers, "values") else []):
            if hasattr(t, "transform"):
                return t.transform
        encoder = getattr(self, "encoder", None)
        if encoder is not None:
            return getattr(encoder, "transform", None)
        return None
