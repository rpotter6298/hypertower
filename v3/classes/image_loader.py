"""
CachedImageLoader — shared image loading with optional in-memory cache.

A single instance can be passed to any dataset (SlotDataset, EyeDataset, etc.)
so that images are decoded from disk at most once per run, regardless of how
many folds or loaders reference the same file.

Usage:
    loader = CachedImageLoader(enabled=True, workers=4)
    loader.warm(paths, preprocessor=my_crop_fn)   # optional: parallel pre-fill
    img = loader.load(path, preprocessor=my_crop_fn)  # returns PIL Image
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
from PIL import Image


class CachedImageLoader:
    """
    Loads PIL Images from disk with an optional shared in-memory cache.

    The cache stores decoded, pre-preprocessed images as uint8 numpy arrays
    (RGB, HWC).  Storing after preprocessing means the preprocessor (e.g.
    cropper) runs only once per image across all folds and epochs.

    Parameters
    ----------
    enabled : bool
        When False the cache is disabled and every call hits disk.
    workers : int
        Number of threads used by warm().  0 or 1 = single-threaded.
    """

    def __init__(self, *, enabled: bool = True, workers: int = 4) -> None:
        self._cache: dict[str, np.ndarray] | None = {} if enabled else None
        self._workers = workers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._cache is not None

    def __len__(self) -> int:
        return len(self._cache) if self._cache is not None else 0

    def load(
        self,
        path: str | Path,
        preprocessor: Optional[Callable[..., Image.Image]] = None,
    ) -> Image.Image:
        """
        Return a PIL Image for *path*.

        If the cache is enabled the image is stored after preprocessing so
        the preprocessor only runs once.  Subsequent calls return a copy
        reconstructed from the cached array.
        """
        key = str(path)

        if self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                return Image.fromarray(cached, mode="RGB")

        img = Image.open(path).convert("RGB")
        if preprocessor is not None:
            img = _call_preprocessor(preprocessor, img, path)

        if self._cache is not None:
            self._cache[key] = np.asarray(img, dtype=np.uint8)

        return img

    def warm(
        self,
        paths: Iterable[str | Path],
        preprocessor: Optional[Callable[..., Image.Image]] = None,
    ) -> None:
        """
        Pre-populate the cache for all *paths* (no-op when disabled).

        Already-cached paths are skipped so calling warm() multiple times
        (e.g. once per fold) is safe and only loads new images.
        """
        if self._cache is None:
            return

        to_warm = [str(p) for p in paths if str(p) not in self._cache]
        if not to_warm:
            return

        already = len(paths if isinstance(paths, (list, tuple)) else list(paths)) - len(to_warm)
        print(
            f"[image_cache] warming {len(to_warm)} images"
            + (f" ({already} already cached)" if already else ""),
            flush=True,
        )

        def _warm_one(path_str: str) -> None:
            if path_str in self._cache:   # guard against races
                return
            img = Image.open(path_str).convert("RGB")
            if preprocessor is not None:
                img = _call_preprocessor(preprocessor, img, Path(path_str))
            self._cache[path_str] = np.asarray(img, dtype=np.uint8)

        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        if self._workers <= 1:
            it = tqdm(to_warm, desc="Warm image cache", unit="img") if tqdm else to_warm
            for p in it:
                _warm_one(p)
        else:
            with ThreadPoolExecutor(max_workers=self._workers) as ex:
                futures = {ex.submit(_warm_one, p): p for p in to_warm}
                it = (
                    tqdm(as_completed(futures), total=len(futures),
                         desc="Warm image cache", unit="img")
                    if tqdm else as_completed(futures)
                )
                for fut in it:
                    fut.result()


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _call_preprocessor(
    fn: Callable[..., Image.Image],
    img: Image.Image,
    path: Path,
) -> Image.Image:
    """Call preprocessor with (img, path) or just (img) depending on arity."""
    try:
        return fn(img, path)
    except TypeError:
        return fn(img)
