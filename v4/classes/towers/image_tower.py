"""image_tower — ImageEncoder for v4.

Self-contained: no v3 dependencies.
Inherits get_sample dispatch from TowerBase.

Geometry injection (EPC supply)
--------------------------------
When geometry_source is set, ImageEncoder asks the image_data view for a loader
via image_data.build_geometry_loader(source, **kwargs).  The view is responsible
for understanding what that source means for its specific domain (fundus contours,
U-Net segmentations, cat ear landmarks, etc.).

During early_pass the loader pre-computes all per-entity geometry vectors and
publishes them to the EarlyPassContext under the key "geometry_vectors"
({(entity_id...): np.ndarray of length geom_dim}).  ClinicalEncoder (or any
other tower with epc_requests: ["geometry_vectors"]) can then consume them.

The tower reads feature_dim and feature_names from the loader instance, so it
can log geometry info without knowing anything about CDR, disc masks, or other
domain-specific concepts.

Config example:
{
  "name":         "img",
  "module":       "v4.classes.towers.image_tower",
  "class":        "ImageEncoder",
  "data_source":  "image",
  "epc_supplies": ["geometry_vectors"],
  "args": {
    "backbone":        "refugelike",
    "augment":         true,
    "geometry_source": "gt",
    "contour_dir":     "Papila/ExpertsSegmentations/Contours"
  }
}
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from v4.classes.towerbase import TowerBase
from v4.classes.accessory.backbones import build_backbone
from v4.classes.accessory.se_block import SEBlock
from v4.classes.accessory.transforms import (
    build_backbone_transform, build_eval_transform, build_split_transforms,
)


class ImageEncoder(TowerBase):
    """Vision backbone → pooled feature vector.

    image_data        : ImageDataView — provides load_image(*ids) and side_map.
                        Must implement build_geometry_loader(source, **kwargs)
                        if geometry_source is set.
    backbone          : backbone key (see accessory/backbones.py)
    freeze_ratio      : fraction of early blocks to freeze in [0, 1]
    use_se            : apply SE attention over the pooled feature vector
    augment           : include random flip/rotation/jitter in the train transform
    cache_transformed : if True, cache resized + ToTensor'd float32 [0, 1] CHW
                        tensors per fold.  Per-batch cost drops to augment +
                        Normalize on tensors only (no PIL, no Resize, no decode).
                        Memory: ~3 × crop_size² × 4B per cached image.
                        Cache is rebuilt at the start of every fold via early_pass.
    geometry_source   : source key passed to image_data.build_geometry_loader()
                        (e.g. "gt", "unet").  None = geometry disabled.
    **geom_kwargs     : forwarded verbatim to build_geometry_loader() — e.g.
                        contour_dir="Papila/ExpertsSegmentations/Contours"
    """

    EPC_GEOMETRY_KEY = "geometry_vectors"

    def __init__(
        self,
        image_data,
        backbone:          str        = "efficientnet_b0",
        freeze_ratio:      float      = 0.0,
        use_se:            bool       = False,
        se_reduction:      int        = 16,
        se_pre_norm:       bool       = True,
        augment:           bool       = True,
        cache_transformed: bool       = False,
        geometry_source:   str | None = None,
        **geom_kwargs: Any,
    ):
        super().__init__()
        self.image_data     = image_data
        self._name          = backbone
        self.backbone, self._base_dim, self._blocks = build_backbone(backbone, freeze_ratio)

        self._cache_transformed = cache_transformed
        if cache_transformed:
            self._precache_tf, self._post_train_tf = build_split_transforms(backbone, augment=augment)
            _,                 self._post_eval_tf  = build_split_transforms(backbone, augment=False)
            self._tensor_cache: dict[tuple, torch.Tensor] = {}
        else:
            self.transform      = build_backbone_transform(backbone, augment=augment)
            self.eval_transform = build_eval_transform(backbone)

        self.tower_ln = nn.LayerNorm(self._base_dim) if se_pre_norm else nn.Identity()
        self.tower_se = SEBlock(self._base_dim, reduction=se_reduction, residual=True) if use_se else None

        self._geom_loader = None
        if geometry_source is not None:
            if not hasattr(image_data, "build_geometry_loader"):
                raise TypeError(
                    f"ImageEncoder geometry_source={geometry_source!r} requires "
                    f"image_data to implement build_geometry_loader(), "
                    f"but {type(image_data).__name__} does not."
                )
            self._geom_loader = image_data.build_geometry_loader(geometry_source, **geom_kwargs)
            print(
                f"[ImageEncoder] geometry_source={geometry_source!r}  "
                f"features={self._geom_loader.feature_names}",
                flush=True,
            )

    # ── TowerBase interface ──────────────────────────────────────────────────

    @property
    def out_dim(self) -> int:
        return self._base_dim

    @property
    def _side_map(self) -> dict[str, str]:
        return self.image_data.side_map

    def _get(self, *ids) -> torch.Tensor:
        if self._cache_transformed:
            key = tuple(ids)
            cached = self._tensor_cache.get(key)
            if cached is None:
                cached = self._precache_tf(self.image_data.load_image(*ids))
                self._tensor_cache[key] = cached
            tail = self._post_train_tf if self.training else self._post_eval_tf
            return tail(cached)
        img = self.image_data.load_image(*ids)
        t   = self.transform if self.training else self.eval_transform
        return t(img)

    # ── EPC early_pass ───────────────────────────────────────────────────────

    def early_pass(self, context) -> None:
        """Per-fold setup: warm tensor cache (if enabled), publish geometry vectors."""
        data  = context.require("data")
        split = context.require("split")

        if self._cache_transformed:
            self._tensor_cache.clear()
            n = self._warm_tensor_cache(data, split)
            print(
                f"[ImageEncoder] warmed transformed-tensor cache for {n} entries "
                f"({self._name})",
                flush=True,
            )

        if self._geom_loader is None:
            return

        train_samples = data.collect_samples(split.train)
        all_samples   = train_samples + data.collect_samples(split.val)
        if split.test is not None:
            all_samples += data.collect_samples(split.test)

        if hasattr(self._geom_loader, "reset_cache"):
            self._geom_loader.reset_cache()
        if hasattr(self._geom_loader, "reset_weights"):
            self._geom_loader.reset_weights()
        if hasattr(self._geom_loader, "finetune"):
            self._geom_loader.finetune(train_samples)

        self._geom_loader.precompute(all_samples)
        vecs = self._geom_loader.all_vectors()
        context.put(self.EPC_GEOMETRY_KEY, vecs)
        print(
            f"[ImageEncoder] published {len(vecs)} geometry vectors "
            f"(dim={self._geom_loader.feature_dim}) to EPC key '{self.EPC_GEOMETRY_KEY}'",
            flush=True,
        )

    def _warm_tensor_cache(self, data, split) -> int:
        """Pre-fill the per-tower tensor cache for all entries in this fold's splits."""
        seen: set[tuple] = set()
        for df in (split.train, split.val, split.test):
            if df is None or len(df) == 0:
                continue
            pc = data.patient_col
            for _, row in df.iterrows():
                pid = int(row[pc])
                eye = str(row.get("eyeID", "OD"))
                key = (pid, eye)
                if key in self._tensor_cache or key in seen:
                    continue
                self._tensor_cache[key] = self._precache_tf(self.image_data.load_image(pid, eye))
                seen.add(key)
        return len(self._tensor_cache)

    # ── nn.Module forward ────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.backbone(x)
        if self.tower_se is not None:
            y, _ = self.tower_se(self.tower_ln(y))
        return y

    # ── Utilities ────────────────────────────────────────────────────────────

    def set_freeze_ratio(self, ratio: float) -> None:
        """Dynamically freeze the earliest floor(N * ratio) backbone blocks."""
        r        = max(0.0, min(1.0, float(ratio)))
        n_freeze = int(math.floor(len(self._blocks) * r))
        for b in self._blocks:
            for p in b.parameters():
                p.requires_grad = True
        for b in self._blocks[:n_freeze]:
            for p in b.parameters():
                p.requires_grad = False
