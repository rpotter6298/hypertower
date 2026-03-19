from __future__ import annotations

import torch
import torch.nn as nn

from v3.classes.SE_attention import SEBlock, SEGateLogger


class Bridge(nn.Module):
    def __init__(
        self,
        img_dim,
        meta_dim,
        num_classes,
        fusion_dim=256,
        mode="fused",
        use_se: bool = True,
        se_reduction: int = 16,
        se_pre_norm: bool = True,
    ):
        super().__init__()
        self.mode = mode
        self.use_se = use_se

        # project towers to equal width
        self.W_img = nn.Linear(img_dim, fusion_dim)
        self.W_md = nn.Linear(meta_dim, fusion_dim)

        # optional: layernorm before SE
        self.ln_img = nn.LayerNorm(fusion_dim) if se_pre_norm else nn.Identity()
        self.ln_md = nn.LayerNorm(fusion_dim) if se_pre_norm else nn.Identity()

        # SE gate on the fused vector
        self.se = SEBlock(fusion_dim, reduction=se_reduction, residual=True) if use_se else None
        self.se_log = SEGateLogger(enabled=use_se, track_channels=False, dim=fusion_dim)

        # heads
        self.classifier_fused = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fusion_dim, num_classes),
        )
        self.classifier_img = nn.Linear(img_dim, num_classes)
        self.classifier_cd = nn.Linear(meta_dim, num_classes)

    def reset_se_stats(self):
        """Call at epoch start."""
        if getattr(self, "se_log", None):
            self.se_log.reset()

    def get_se_stats(self, reset: bool = True):
        """Call after eval. Returns dict or None."""
        if getattr(self, "se_log", None) and self.se_log.enabled:
            return self.se_log.get(reset=reset)
        return None

    def forward(self, img_feats, md_feats):
        out_img = None if self.mode == "clinical_only" else self.classifier_img(img_feats)
        out_md = None if self.mode == "image_only" else self.classifier_cd(md_feats)

        if self.mode == "fused":
            hi = self.ln_img(self.W_img(img_feats))  # image features
            hm = self.ln_md(self.W_md(md_feats))  # clinical data features
            fused = hi * hm  # elementwise product
            # apply SE gates
            if self.se is not None:
                fused, gates = self.se(fused)
                if self.se_log.enabled:
                    self.se_log.accumulate(gates)

            if self.se is not None and self.training and self.se_log.enabled:
                if not hasattr(self, "_dbg_seen"):
                    self._dbg_seen = 0
                if self._dbg_seen < 3:  # print only a few times
                    print("[SE] gate mean this batch:", gates.mean().item())
                    self._dbg_seen += 1
            out_f = self.classifier_fused(fused)
            return out_f, out_img, out_md
        # if ablation modes:
        if self.mode == "image_only":
            return out_img, out_img, None
        if self.mode == "clinical_only":
            return out_md, None, out_md


class VoteBridge(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vote_combiner = nn.Linear(num_classes * 2, num_classes)  # two sets of logits

    def forward(self, out_img, out_md):
        votes = torch.cat([out_img, out_md], dim=1)
        return self.vote_combiner(votes)
