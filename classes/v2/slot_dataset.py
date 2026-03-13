from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional

from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

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
        image_cache: Optional[dict[str, np.ndarray]] = None,
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
        path = Path(value)
        cache_key = str(value)

        if self.image_cache is not None:
            cached = self.image_cache.get(cache_key)
            if cached is not None:
                return self.image_transform(Image.fromarray(cached, mode="RGB"))

        img = Image.open(path).convert("RGB")
        if self.image_preprocessor is not None:
            try:
                img = self.image_preprocessor(img, path)
            except TypeError:
                img = self.image_preprocessor(img)

        if self.image_cache is not None:
            self.image_cache[cache_key] = np.asarray(img, dtype=np.uint8)

        return self.image_transform(img)

    def prebuild_image_cache(self, cache_workers: int = 0) -> None:
        """Pre-populate image_cache for all samples in this dataset."""
        if self.image_cache is None:
            return
        paths = list({
            str(record[key])
            for record in self.samples
            for key, desc in self.slot_descriptors.items()
            if desc.kind == "image" and record.get(key) is not None
        })
        to_warm = [p for p in paths if p not in self.image_cache]
        if not to_warm:
            return
        print(
            f"[image_cache] warming {len(to_warm)} images "
            f"({len(paths) - len(to_warm)} already cached)",
            flush=True,
        )

        def _warm_one(path_str: str) -> None:
            if path_str in self.image_cache:
                return
            p = Path(path_str)
            img = Image.open(p).convert("RGB")
            if self.image_preprocessor is not None:
                try:
                    img = self.image_preprocessor(img, p)
                except TypeError:
                    img = self.image_preprocessor(img)
            self.image_cache[path_str] = np.asarray(img, dtype=np.uint8)

        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        if cache_workers <= 1:
            it = tqdm(to_warm, desc="Warm image cache", unit="img") if tqdm else to_warm
            for path_str in it:
                _warm_one(path_str)
        else:
            with ThreadPoolExecutor(max_workers=cache_workers) as ex:
                futures = {ex.submit(_warm_one, p): p for p in to_warm}
                it = tqdm(as_completed(futures), total=len(futures), desc="Warm image cache", unit="img") if tqdm else as_completed(futures)
                for fut in it:
                    fut.result()

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
