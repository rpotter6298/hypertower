from __future__ import annotations

from typing import Any, Callable, Optional

from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .image_loader import CachedImageLoader
from .profiles.base import SlotDescriptor


def slot_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        return {}
    keys = batch[0].keys()
    out: dict[str, Any] = {}
    for key in keys:
        vals = [item.get(key) for item in batch]
        if all(isinstance(v, torch.Tensor) for v in vals):
            try:
                out[key] = torch.stack(vals, dim=0)
            except Exception:
                out[key] = vals
        else:
            out[key] = vals
    return out


class SlotDataset(Dataset):
    """
    Dataset that yields dicts of slot-keyed values.

    Sample records are expected to be dicts with keys matching slot descriptors.
    Image slots accept filesystem paths; matrix slots accept array-like values.
    """

    def __init__(
        self,
        samples: list[dict[str, Any]],
        slot_descriptors: dict[str, SlotDescriptor],
        *,
        image_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        matrix_transform: Optional[Callable[[Any], torch.Tensor]] = None,
        image_preprocessor: Optional[Callable[..., Image.Image]] = None,
        image_cache: Optional[CachedImageLoader] = None,
    ) -> None:
        self.samples = samples
        self.slot_descriptors = slot_descriptors
        self.image_transform = image_transform or transforms.ToTensor()
        self.matrix_transform = matrix_transform or self._default_matrix_transform
        self.image_preprocessor = image_preprocessor
        self.image_cache = image_cache

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.samples[idx]
        out: dict[str, Any] = {}
        for key, desc in self.slot_descriptors.items():
            val = record.get(key)
            if desc.kind == "image":
                out[key] = self._load_image(val, required=desc.required)
            elif desc.kind == "matrix":
                out[key] = self._load_matrix(val, required=desc.required)
            else:
                out[key] = val
        return out

    def _load_image(self, value: Any, *, required: bool) -> Optional[torch.Tensor]:
        if value is None:
            if required:
                raise ValueError("Missing required image slot")
            return None
        if self.image_cache is not None:
            img = self.image_cache.load(value, preprocessor=self.image_preprocessor)
        else:
            from .image_loader import _call_preprocessor
            img = Image.open(value).convert("RGB")
            if self.image_preprocessor is not None:
                img = _call_preprocessor(self.image_preprocessor, img, Path(value))
        return self.image_transform(img)

    def prebuild_image_cache(self, cache_workers: int = 0) -> None:
        """Pre-populate the CachedImageLoader for all image paths in this dataset."""
        if self.image_cache is None:
            return
        paths = list({
            str(record[key])
            for record in self.samples
            for key, desc in self.slot_descriptors.items()
            if desc.kind == "image" and record.get(key) is not None
        })
        self.image_cache._workers = cache_workers or self.image_cache._workers
        self.image_cache.warm(paths, preprocessor=self.image_preprocessor)

    def _load_matrix(self, value: Any, *, required: bool) -> Optional[torch.Tensor]:
        if value is None:
            if required:
                raise ValueError("Missing required matrix slot")
            return None
        return self.matrix_transform(value)

    @staticmethod
    def _default_matrix_transform(value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.float()
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value.astype(np.float32, copy=False))
        return torch.as_tensor(value, dtype=torch.float32)
