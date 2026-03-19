"""
portable_versions/image_loader.py
==================================
A self-contained image loader with in-memory caching, an optional
preprocessing pipeline (e.g. disc cropping), and composable augmentations.

Returns plain NumPy arrays — works with PyTorch, TensorFlow, JAX, or
anything else that can consume an ndarray.

Dependencies: Pillow, numpy  (nothing else)

Quickstart
----------
    from portable_versions.image_loader import ImageLoader, RandomHorizontalFlip, RandomRotation, ColorJitter

    # 1. Build the loader (once per run)
    loader = ImageLoader(
        target_size=(200, 200),
        normalize=True,           # float32 in [0, 1] with ImageNet mean/std
        cache=True,               # each image decoded from disk only once
        workers=4,                # parallel cache warm-up threads
        preprocessor=my_crop_fn, # optional callable(PIL.Image) -> PIL.Image
    )

    # 2. Attach augmentations (applied randomly and independently per call)
    loader.augmentation = [
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=15),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    ]

    # 3. Warm the cache up front (optional but fast)
    loader.warm(all_paths)

    # 4. Fetch images by path list — call as many times as you like
    #    Returns ndarray of shape (N, H, W, 3), dtype float32
    imgs = loader.get_img(train_paths)

    # For TensorFlow:
    import tensorflow as tf
    tensor = tf.constant(imgs)          # (N, H, W, 3)

    # For PyTorch:
    import torch
    tensor = torch.from_numpy(imgs).permute(0, 3, 1, 2)  # (N, C, H, W)


Augmentations reference
-----------------------
All augmentation classes live in this file and depend only on PIL + numpy.

    RandomHorizontalFlip(p=0.5)
    RandomVerticalFlip(p=0.5)
    RandomRotation(degrees=15)
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)
    RandomGrayscale(p=0.1)

You can also pass any callable(PIL.Image.Image) -> PIL.Image.Image as an
augmentation step.
"""
from __future__ import annotations

import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

# ImageNet channel statistics (RGB)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

PathLike = Union[str, Path]


def _call_preprocessor(
    fn: Callable[..., Image.Image],
    img: Image.Image,
    path: Path,
) -> Image.Image:
    """Call preprocessor as fn(img, path) if it accepts two args, else fn(img)."""
    try:
        return fn(img, path)
    except TypeError:
        return fn(img)


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

class ImageLoader:
    """
    Preprocessing pipeline + in-memory cache + augmentation, returning NumPy.

    Parameters
    ----------
    target_size : (height, width)
        Output spatial dimensions.  Applied after ``preprocessor`` (if any).
        Ignored when a ``preprocessor`` already resizes to the right size.
    normalize : bool
        When True, output is float32 with ImageNet mean/std subtraction.
        When False, output is uint8 in [0, 255].
    cache : bool
        Store decoded+preprocessed images in RAM so each file is read from
        disk at most once.  The cache persists across ``get_img`` calls.
    workers : int
        Thread count for ``warm()``.  0 or 1 = single-threaded.
    preprocessor : callable, optional
        Called as ``preprocessor(img: PIL.Image) -> PIL.Image`` before
        resizing and caching.  Use this for disc cropping, padding, etc.
    augmentation : list of callables
        Each element is called as ``fn(img: PIL.Image) -> PIL.Image``.
        Applied **after** cache retrieval, so augmentations are NOT cached —
        they are re-sampled independently on every ``get_img`` call.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        *,
        normalize: bool = True,
        cache: bool = True,
        workers: int = 4,
        preprocessor: Optional[Callable[[Image.Image], Image.Image]] = None,
    ) -> None:
        self.target_size  = target_size
        self.normalize    = normalize
        self.workers      = workers
        self.preprocessor = preprocessor
        self.augmentation: List[Callable[[Image.Image], Image.Image]] = []

        self._cache: Optional[dict[str, np.ndarray]] = {} if cache else None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def warm(self, paths: Iterable[PathLike]) -> None:
        """
        Pre-load all *paths* into the cache in parallel.

        Already-cached paths are skipped, so calling ``warm`` multiple
        times (e.g. once per fold) is safe and only loads new images.
        """
        if self._cache is None:
            return

        paths = [str(p) for p in paths]
        to_warm = [p for p in paths if p not in self._cache]
        if not to_warm:
            return

        already = len(paths) - len(to_warm)
        print(
            f"[ImageLoader] warming {len(to_warm)} images"
            + (f" ({already} already cached)" if already else ""),
            flush=True,
        )

        def _load_one(path_str: str) -> None:
            arr = self._decode(path_str)
            with self._lock:
                self._cache.setdefault(path_str, arr)

        if self.workers <= 1:
            for p in to_warm:
                _load_one(p)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                futures = {ex.submit(_load_one, p): p for p in to_warm}
                for fut in as_completed(futures):
                    fut.result()

    def get_img(
        self,
        paths: Iterable[PathLike],
        augment: bool = True,
    ) -> np.ndarray:
        """
        Return images for the given paths as a single NumPy array.

        Parameters
        ----------
        paths : iterable of path-like
            File paths to load.  If the cache is enabled and a path has
            been warmed (or loaded before), it is served from RAM.
        augment : bool
            Apply ``self.augmentation`` pipeline.  Set to False at eval time.

        Returns
        -------
        np.ndarray, shape (N, H, W, 3)
            float32 in [0, 1] (or normalised) if ``self.normalize`` is True,
            otherwise uint8 in [0, 255].
        """
        imgs = []
        for p in paths:
            img = self._get_one(str(p), augment=augment)
            imgs.append(img)
        return np.stack(imgs, axis=0)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _decode(self, path_str: str) -> np.ndarray:
        """Open, preprocess, and resize → uint8 HWC ndarray (for the cache)."""
        img = Image.open(path_str).convert("RGB")
        if self.preprocessor is not None:
            img = _call_preprocessor(self.preprocessor, img, Path(path_str))
        # Only resize here if preprocessor didn't already produce target_size
        if img.size != (self.target_size[1], self.target_size[0]):
            img = img.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)

    def _get_one(self, path_str: str, augment: bool) -> np.ndarray:
        if self._cache is not None:
            arr = self._cache.get(path_str)
            if arr is None:
                arr = self._decode(path_str)
                with self._lock:
                    self._cache.setdefault(path_str, arr)
            img = Image.fromarray(arr, mode="RGB")
        else:
            img = Image.open(path_str).convert("RGB")
            if self.preprocessor is not None:
                img = self.preprocessor(img)
            if img.size != (self.target_size[1], self.target_size[0]):
                img = img.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)

        if augment and self.augmentation:
            for fn in self.augmentation:
                img = fn(img)

        arr = np.asarray(img, dtype=np.float32) / 255.0
        if self.normalize:
            arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
        else:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return arr

    def __len__(self) -> int:
        """Number of images currently in the cache."""
        return len(self._cache) if self._cache is not None else 0

    def __repr__(self) -> str:
        return (
            f"ImageLoader(target_size={self.target_size}, "
            f"normalize={self.normalize}, "
            f"cached={len(self)}, "
            f"augmentations={len(self.augmentation)})"
        )


# ---------------------------------------------------------------------------
# Augmentation primitives  (PIL-only, no torch/tf dependencies)
# ---------------------------------------------------------------------------

class RandomHorizontalFlip:
    """Flip image left-right with probability *p*."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        return ImageOps.mirror(img) if random.random() < self.p else img


class RandomVerticalFlip:
    """Flip image top-bottom with probability *p*."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        return ImageOps.flip(img) if random.random() < self.p else img


class RandomRotation:
    """Rotate by a uniformly-sampled angle in [-degrees, +degrees]."""
    def __init__(self, degrees: float = 15):
        self.degrees = degrees

    def __call__(self, img: Image.Image) -> Image.Image:
        angle = random.uniform(-self.degrees, self.degrees)
        return img.rotate(angle, resample=Image.BILINEAR, expand=False)


class ColorJitter:
    """
    Randomly jitter brightness, contrast, saturation, and hue.

    Each factor is sampled uniformly from [1 - amount, 1 + amount].
    Hue shift is sampled from [-hue, +hue] (range 0–0.5).
    Pass 0 for any channel to leave it unchanged.
    """
    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float   = 0.2,
        saturation: float = 0.1,
        hue: float        = 0.05,
    ):
        self.brightness = brightness
        self.contrast   = contrast
        self.saturation = saturation
        self.hue        = hue

    def __call__(self, img: Image.Image) -> Image.Image:
        ops = []
        if self.brightness:
            ops.append(("brightness", self.brightness))
        if self.contrast:
            ops.append(("contrast",   self.contrast))
        if self.saturation:
            ops.append(("saturation", self.saturation))
        if self.hue:
            ops.append(("hue",        self.hue))
        random.shuffle(ops)

        for kind, amount in ops:
            factor = random.uniform(1 - amount, 1 + amount)
            if kind == "brightness":
                img = ImageEnhance.Brightness(img).enhance(factor)
            elif kind == "contrast":
                img = ImageEnhance.Contrast(img).enhance(factor)
            elif kind == "saturation":
                img = ImageEnhance.Color(img).enhance(factor)
            elif kind == "hue":
                # PIL has no direct hue enhancer — shift via HSV in numpy
                arr = np.asarray(img.convert("HSV"), dtype=np.int16)
                shift = int(random.uniform(-self.hue, self.hue) * 255)
                arr[:, :, 0] = (arr[:, :, 0] + shift) % 256
                img = Image.fromarray(arr.astype(np.uint8), mode="HSV").convert("RGB")
        return img


class RandomGrayscale:
    """Convert to grayscale (keeping 3 channels) with probability *p*."""
    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            img = ImageOps.grayscale(img).convert("RGB")
        return img


