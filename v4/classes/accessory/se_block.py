"""se_block — Squeeze-and-Excitation channel gating."""
from __future__ import annotations

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """SE-style channel gating for vectors, feature maps, and sequences.

    Input shapes:
      [N, C]       (vector)     — squeeze is identity
      [N, C, H, W] (image map)  — squeeze over H, W
      [N, T, C]    (sequence)   — squeeze over T

    Gate modes:
      residual (default): gate = 1 + tanh(MLP(s))  in (0, 2), identity at init
      plain:              gate = sigmoid(MLP(s))    in (0, 1)
    """

    def __init__(
        self,
        dim: int,
        reduction: int = 16,
        residual: bool = True,
        identity_init: bool = True,
    ):
        super().__init__()
        hid = max(1, dim // max(1, reduction))
        self.fc1      = nn.Linear(dim, hid, bias=True)
        self.act      = nn.ReLU(inplace=True)
        self.fc2      = nn.Linear(hid, dim, bias=True)
        self.residual = residual

        if residual and identity_init:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def _squeeze(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x
        if x.dim() == 4:
            return x.mean(dim=(2, 3))
        if x.dim() == 3:
            return x.mean(dim=1)
        return x.view(x.size(0), -1)

    def _broadcast(self, gate: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        if like.dim() == 2:
            return gate
        if like.dim() == 3:
            return gate.unsqueeze(1)
        if like.dim() == 4:
            return gate.unsqueeze(-1).unsqueeze(-1)
        return gate.view_as(like)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = self._squeeze(x)
        u = self.fc2(self.act(self.fc1(s)))
        gate = (1.0 + torch.tanh(u)) if self.residual else torch.sigmoid(u)
        return x * self._broadcast(gate, x), gate


class SEGateLogger:
    """Running stats over SE gate activations across batches."""

    def __init__(
        self,
        enabled: bool = True,
        track_channels: bool = False,
        dim: int | None = None,
    ):
        self.enabled        = enabled
        self.track_channels = track_channels
        self.dim            = dim
        self.reset()

    def reset(self) -> None:
        self._n      = 0
        self._sum    = 0.0
        self._sum2   = 0.0
        self._lt02   = 0
        self._gt08   = 0
        self._ch_sum   = None
        self._ch_count = 0
        if self.track_channels and self.dim is not None:
            self._ch_sum = torch.zeros(self.dim, dtype=torch.float32)

    @torch.no_grad()
    def accumulate(self, gates: torch.Tensor) -> None:
        if not self.enabled:
            return
        if gates.dim() == 4:
            g = gates.mean(dim=(2, 3))
        elif gates.dim() == 3:
            g = gates.mean(dim=1)
        elif gates.dim() == 2:
            g = gates
        else:
            g = gates.view(gates.size(0), -1)
        g = g.detach()
        self._n    += g.numel()
        self._sum  += g.sum().item()
        self._sum2 += (g * g).sum().item()
        self._lt02 += (g < 0.2).sum().item()
        self._gt08 += (g > 0.8).sum().item()
        if self._ch_sum is not None:
            self._ch_sum  += g.sum(dim=0).cpu()
            self._ch_count += g.size(0)

    def get(self, reset: bool = True) -> dict | None:
        if self._n == 0:
            return None
        mean = self._sum / self._n
        var  = max(0.0, self._sum2 / self._n - mean * mean)
        out  = {
            "mean":        mean,
            "std":         var ** 0.5,
            "pct_lt_0.2":  self._lt02 / self._n,
            "pct_gt_0.8":  self._gt08 / self._n,
        }
        if self._ch_sum is not None and self._ch_count > 0:
            out["channel_mean"] = (self._ch_sum / float(self._ch_count)).tolist()
        if reset:
            self.reset()
        return out
