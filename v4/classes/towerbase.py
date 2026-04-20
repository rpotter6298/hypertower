"""towerbase — v4 TowerBase ABC.

All v4 towers inherit from TowerBase.  The only required interface is:

    out_dim    : int property — embedding dimensionality
    _side_map  : dict property — generic side keys {"a","b"} → dataset-specific ids
    _get(*ids) : retrieve and transform one sample by entity_id slots

get_sample handles the eye-level / patient-level dispatch automatically:
  len(entity_id) > 1  — full key; calls _get(*entity_id) directly
  len(entity_id) == 1 — patient key; builds {"a": _get(...), "b": _get(...)}
                        using _side_map to expand the missing slot

Towers may optionally implement early_pass(context) for cross-tower
communication before loaders are built.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from v4.classes.dataset import ShellEntry


class TowerBase(nn.Module, ABC):

    @property
    @abstractmethod
    def out_dim(self) -> int: ...

    @property
    @abstractmethod
    def _side_map(self) -> dict[str, str]: ...

    @abstractmethod
    def _get(self, *ids) -> torch.Tensor: ...

    def get_sample(self, entry: ShellEntry) -> torch.Tensor | dict[str, torch.Tensor]:
        eid = entry.entity_id
        if len(eid) > 1:
            return self._get(*eid)
        return {key: self._get(eid[0], id_1) for key, id_1 in self._side_map.items()}

    def set_phase(self, phase: str) -> None:
        """Called by the orchestrator at the start of each training epoch.

        Default: freeze all parameters during fused_warmup, train otherwise.
        Override to implement tower-specific phase behaviour.
        """
        trainable = phase != "fused_warmup"
        for p in self.parameters():
            p.requires_grad_(trainable)

    def early_pass(self, context) -> None:
        pass
