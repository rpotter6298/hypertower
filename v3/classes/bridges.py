from __future__ import annotations


import torch
import torch.nn as nn

from v3.classes.SE_attention import SEBlock, SEGateLogger


# ---------------------------------------------------------------------------
# HTClassifier — standalone classification head
# ---------------------------------------------------------------------------

class HTClassifier(nn.Module):
    """Minimal classification head: ReLU → Dropout → Linear(in_dim → num_classes).

    Used as the output stage of Bridge, HyperBridge, and any vehicle that needs
    a reusable, identifiable classifier type.
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


# ---------------------------------------------------------------------------
# Bridge — N-tower fusion
# ---------------------------------------------------------------------------

class Bridge(nn.Module):
    """
    N-tower fusion bridge.

    Takes a list of tower embeddings, projects each to a common ``fusion_dim``,
    element-wise multiplies all projections, optionally applies an SE gate, then
    classifies the fused representation via an HTClassifier.

    Each tower also gets an auxiliary classification head (used for BCD training).

    Construction
    ------------
    ``tower_dims`` is an ordered list of embedding dimensionalities — one entry per
    embedding slot that will be passed to ``fuse()`` or ``forward()``.

    Tower slots are accessed by index: ``W[i]``, ``ln[i]``, ``aux_heads[i]``.
    The bridge has no knowledge of what modality each slot carries.
    """

    def __init__(
        self,
        tower_dims: list[int],
        num_classes: int,
        fusion_dim: int = 256,
        mode: str = "fused",
        dropout: float = 0.5,
        use_se: bool = True,
        se_reduction: int = 16,
        se_pre_norm: bool = True,
    ):
        super().__init__()
        self.mode = mode
        self.use_se = use_se
        self.tower_dims = list(tower_dims)

        # Per-tower projection heads: each projects dim_i → fusion_dim
        self.W = nn.ModuleList([nn.Linear(d, fusion_dim) for d in tower_dims])
        self.ln = nn.ModuleList(
            [nn.LayerNorm(fusion_dim) if se_pre_norm else nn.Identity()
             for _ in tower_dims]
        )

        # Per-tower auxiliary classifiers (for BCD training)
        self.aux_heads = nn.ModuleList([nn.Linear(d, num_classes) for d in tower_dims])

        # SE gate on the fused vector
        self.se = SEBlock(fusion_dim, reduction=se_reduction, residual=True) if use_se else None
        self.se_log = SEGateLogger(enabled=use_se, track_channels=False, dim=fusion_dim)

        # Fused classifier head
        self.classifier_fused = HTClassifier(fusion_dim, num_classes, dropout)

    # ------------------------------------------------------------------
    # SE helpers
    # ------------------------------------------------------------------

    def reset_se_stats(self) -> None:
        """Call at epoch start."""
        if getattr(self, "se_log", None):
            self.se_log.reset()

    def get_se_stats(self, reset: bool = True):
        """Call after eval. Returns dict or None."""
        if getattr(self, "se_log", None) and self.se_log.enabled:
            return self.se_log.get(reset=reset)
        return None

    # ------------------------------------------------------------------
    # Core fusion
    # ------------------------------------------------------------------

    def _compute_fused(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        """Return z_fused embedding (before classifier_fused)."""
        assert len(embeddings) == len(self.W), (
            f"Bridge expects {len(self.W)} embeddings, got {len(embeddings)}"
        )
        h = self.ln[0](self.W[0](embeddings[0]))
        for i in range(1, len(embeddings)):
            h = h * self.ln[i](self.W[i](embeddings[i]))
        if self.se is not None:
            h, gates = self.se(h)
            if self.se_log.enabled:
                self.se_log.accumulate(gates)
        return h

    # ------------------------------------------------------------------
    # N-tower API
    # ------------------------------------------------------------------

    def fuse(
        self, embeddings: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        N-tower forward pass.

        Parameters
        ----------
        embeddings : list of Tensor — one per tower slot (same order as tower_dims).

        Returns
        -------
        logits_fused : Tensor  [B, num_classes]
        aux_logits   : list of Tensor  — one per tower slot, each [B, num_classes]
        """
        z_fused = self._compute_fused(embeddings)
        logits_fused = self.classifier_fused(z_fused)
        aux = [head(e) for head, e in zip(self.aux_heads, embeddings)]
        return logits_fused, aux

    def encode(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        """Return z_fused without applying the classifier head."""
        return self._compute_fused(embeddings)

    def set_phase(self, phase: str) -> None:
        """
        Set requires_grad on bridge sub-modules according to training phase.

          - ``cd_warmup``    — freeze everything in the bridge
          - ``tower_warmup`` — aux_heads trainable, projections + fused head frozen
          - ``fused_warmup`` — projections + fused head trainable, aux_heads frozen
          - ``main`` / other — everything trainable
        """
        def _rg(module, enabled):
            for p in module.parameters():
                p.requires_grad = enabled

        if phase == "cd_warmup":
            _rg(self, False)
            return
        if phase == "tower_warmup":
            for head in self.aux_heads:
                _rg(head, True)
            for W_i in self.W:
                _rg(W_i, False)
            for ln_i in self.ln:
                _rg(ln_i, False)
            _rg(self.classifier_fused, False)
            if self.se is not None:
                _rg(self.se, False)
            return
        if phase == "fused_warmup":
            for head in self.aux_heads:
                _rg(head, False)
            for W_i in self.W:
                _rg(W_i, True)
            for ln_i in self.ln:
                _rg(ln_i, True)
            _rg(self.classifier_fused, True)
            if self.se is not None:
                _rg(self.se, True)
            return
        _rg(self, True)

    def forward(
        self, embeddings: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """N-tower forward. Delegates to fuse()."""
        return self.fuse(embeddings)


# ---------------------------------------------------------------------------
# HyperBridge — higher-order bridge over HT module outputs
# ---------------------------------------------------------------------------

class HyperBridge(nn.Module):
    """Higher-order bridge that fuses z_fused embeddings from multiple HT modules.

    Operates at the HT output level (z_fused from each HT's Bridge.encode())
    rather than raw tower embedding level.

    Modes
    -----
    embedding_mlp (default)
        Concatenate all z_fused inputs → MLP → logits.
        Analogous to EmbeddingMLPEnsembleHT, generalised to N inputs.
        ``Linear(N*fusion_dim → hidden_dim) → ReLU → Dropout → Linear(hidden_dim → num_classes)``

    classic_bridge
        Project each input to ``hidden_dim``, Hadamard product, HTClassifier.
        Analogous to Bridge operating at the HT level — handles inputs of
        differing dims via per-input projection layers.
        ``W[i](z_i) → LayerNorm → Hadamard → ReLU → Dropout → Linear(hidden_dim → num_classes)``

    Both modes expose per-input auxiliary HTClassifier heads for BCD-style training.

    Parameters
    ----------
    input_dims  : ordered dict {name: dim} for each HT input.
                  In embedding_mlp mode, dims may differ.
                  In classic_bridge mode, all dims must be equal (shared space).
    num_classes : output classes
    hidden_dim  : hidden dim for the embedding_mlp MLP head
    mode        : "embedding_mlp" | "classic_bridge"
    dropout     : dropout throughout
    """

    def __init__(
        self,
        input_dims: dict[str, int],
        num_classes: int,
        hidden_dim: int = 256,
        mode: str = "embedding_mlp",
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_names = list(input_dims.keys())
        self.mode = mode
        dims = list(input_dims.values())

        if mode == "embedding_mlp":
            total_dim = sum(dims)
            self.head = nn.Sequential(
                nn.Linear(total_dim, hidden_dim), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes),
            )
        elif mode == "classic_bridge":
            # Project each input to shared fusion_dim space, then Hadamard
            self.W  = nn.ModuleList([nn.Linear(d, hidden_dim) for d in dims])
            self.ln = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in dims])
            self.head = HTClassifier(hidden_dim, num_classes, dropout)
        else:
            raise ValueError(f"Unknown HyperBridge mode: {mode!r}")

        # Per-input auxiliary classifiers (both modes)
        self.aux_heads = nn.ModuleList([
            HTClassifier(d, num_classes, dropout) for d in dims
        ])

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Fuse HT-level embeddings.

        Parameters
        ----------
        inputs : {name: z_fused [B, dim]} — z_fused from each HT's encode()

        Returns
        -------
        logits   : [B, num_classes]
        aux_dict : {name: [B, num_classes]} — per-input aux head logits
        """
        ordered = [inputs[name] for name in self.input_names]

        if self.mode == "embedding_mlp":
            logits = self.head(torch.cat(ordered, dim=1))
        else:  # classic_bridge: project → Hadamard → classify
            h = self.ln[0](self.W[0](ordered[0]))
            for i in range(1, len(ordered)):
                h = h * self.ln[i](self.W[i](ordered[i]))
            logits = self.head(h)

        aux = {name: head(z) for name, head, z
               in zip(self.input_names, self.aux_heads, ordered)}
        return logits, aux


# ---------------------------------------------------------------------------
# VoteBridge
# ---------------------------------------------------------------------------

class VoteBridge(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vote_combiner = nn.Linear(num_classes * 2, num_classes)

    def forward(self, out_img, out_md):
        votes = torch.cat([out_img, out_md], dim=1)
        return self.vote_combiner(votes)
