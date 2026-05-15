"""ortho_bridge — OrthoBridge: wraps any underlying bridge with cross-tower
orthogonality regularization.

Forces each tower to encode information that the other towers DON'T encode by
penalising cross-tower representational similarity.  The penalty is added to
the main training loss via the `modify_loss` hook, so the fusion stage runner
sees it transparently — no other code changes required.

How it works
------------
1. Inner bridge fuses the embeddings into the usual single tensor (forward).
2. While forward is running, we compute pairwise linear-CKA between every
   pair of input embeddings and stash the average value as `self._stashed`.
3. `modify_loss(loss)` returns `loss + ortho_weight * stashed`.  Gradients
   from the penalty flow back through the embeddings into the tower weights,
   pushing each tower's representations apart.

CKA reference: Kornblith et al., "Similarity of Neural Network Representations
Revisited" (ICML 2019).  Linear CKA on centred embeddings is in [0, 1]:
    0  = orthogonal (uncorrelated) representations
    1  = identical (up to linear transform)

Config example:
    {
      "name":   "nt",
      "type":   "fusion",
      "module": "v4.classes.bridges.ortho_bridge",
      "class":  "OrthoBridge",
      "args": {
        "ortho_weight":  0.1,
        "inner_module": "v4.classes.bridges.fusion_bridge",
        "inner_class":  "FusionBridge",
        "inner_args":   { "fusion_dim": 256, "use_se": true }
      }
    }
"""
from __future__ import annotations

import importlib
from itertools import combinations
from typing import Any

import torch
import torch.nn as nn


def _linear_cka(z_a: torch.Tensor, z_b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Linear CKA between two (B, D_a) and (B, D_b) embedding batches.

    Centred Gram-matrix similarity, normalised to [0, 1].
    """
    a = z_a - z_a.mean(dim=0, keepdim=True)
    b = z_b - z_b.mean(dim=0, keepdim=True)
    Ga = a @ a.T
    Gb = b @ b.T
    num = (Ga * Gb).sum()
    den = torch.sqrt((Ga * Ga).sum() * (Gb * Gb).sum() + eps)
    return num / den


class OrthoBridge(nn.Module):
    """Wrap any underlying bridge and add cross-tower orthogonality.

    Parameters
    ----------
    input_dims    : ordered list of input embedding dims  (passed to inner bridge)
    fusion_dim    : convenience alias passed to inner bridge if it accepts it
    ortho_weight  : λ on the orthogonality penalty (default 0.1)
    inner_module  : import path of the inner bridge class
    inner_class   : class name within `inner_module`
    inner_args    : kwargs forwarded to the inner bridge constructor

    The OrthoBridge does NOT touch the inner bridge's forward output — it only
    computes and stashes a penalty during forward, then exposes it via the
    `modify_loss` hook.
    """

    def __init__(
        self,
        input_dims:   list[int],
        fusion_dim:   int   = 256,
        ortho_weight: float = 0.1,
        inner_module: str   = "v4.classes.bridges.fusion_bridge",
        inner_class:  str   = "FusionBridge",
        inner_args:   dict | None = None,
    ):
        super().__init__()
        self.ortho_weight = float(ortho_weight)

        # Build the inner bridge.  Pass fusion_dim through unless the caller's
        # inner_args overrides it.
        inner_kwargs: dict[str, Any] = dict(inner_args or {})
        inner_kwargs.setdefault("fusion_dim", fusion_dim)
        mod = importlib.import_module(inner_module)
        cls = getattr(mod, inner_class)
        self.inner = cls(input_dims, **inner_kwargs)
        self.out_dim = self.inner.out_dim

        # Stash penalty here on every forward; modify_loss reads from it.
        self.register_buffer("_stashed", torch.zeros(()), persistent=False)
        self._last_pairs_cka: list[float] = []

    # ── core ────────────────────────────────────────────────────────────────

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        # Compute pairwise linear-CKA across the raw tower embeddings.
        # We use the RAW per-tower embeddings (not the inner bridge's projected
        # versions) because we want to push the TOWERS apart, not the bridge's
        # internal projections.
        if len(embeddings) >= 2:
            ckas = []
            for i, j in combinations(range(len(embeddings)), 2):
                ckas.append(_linear_cka(embeddings[i], embeddings[j]))
            penalty = torch.stack(ckas).mean()
            self._stashed = penalty
            self._last_pairs_cka = [float(c.detach().cpu()) for c in ckas]
        else:
            self._stashed = torch.zeros((), device=embeddings[0].device)
            self._last_pairs_cka = []

        return self.inner(embeddings)

    def modify_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return loss + self.ortho_weight * self._stashed

    # ── delegate to inner ───────────────────────────────────────────────────

    def set_phase(self, phase: str) -> None:
        if hasattr(self.inner, "set_phase"):
            self.inner.set_phase(phase)

    # ── introspection ───────────────────────────────────────────────────────

    @property
    def last_cka(self) -> float:
        """Mean cross-tower linear CKA from the most recent forward pass.

        Useful for logging — should DECREASE during training if the penalty
        is doing its job.
        """
        return float(self._stashed.detach().cpu()) if self._stashed.numel() else 0.0
