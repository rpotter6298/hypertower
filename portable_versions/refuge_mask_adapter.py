"""
portable_versions/refuge_mask_adapter.py
=========================================
Optic-disc cropper for REFUGE (and REFUGE2) fundus images, designed as a
drop-in ``preprocessor`` for ``ImageLoader``.

Given an image and its corresponding segmentation mask, it:
  1. Extracts the optic disc region from the mask
  2. Computes a padded bounding box around it
  3. Crops and resizes the original image

Dependencies: Pillow, numpy  (nothing else)

Quickstart
----------
    from portable_versions.image_loader import ImageLoader, RandomHorizontalFlip, RandomRotation, ColorJitter
    from portable_versions.refuge_mask_adapter import RefugeMaskCropper

    cropper = RefugeMaskCropper(
        mask_dir="REFUGE/Annotations/Training400/Disc_Cup_Masks",
        scale=1.5,              # context around disc (1.0 = tight, 2.0 = lots of context)
        target_size=(200, 200), # output size — should match ImageLoader target_size
        mask_suffix=".bmp",     # REFUGE1 uses .bmp; REFUGE2 uses .png
    )

    loader = ImageLoader(
        target_size=(200, 200),
        normalize=True,
        preprocessor=cropper,
    )
    loader.augmentation = [
        RandomHorizontalFlip(),
        RandomRotation(15),
        ColorJitter(0.2, 0.2, 0.1, 0.05),
    ]

    imgs = loader.get_img(image_paths, augment=True)  # (N, 200, 200, 3)

REFUGE mask formats
-------------------
REFUGE1  Grayscale BMP: background=128, disc=255, cup=0
REFUGE2  RGB PNG:       background detected from image borders, disc/cup by colour

Both are handled automatically.

Directory structure assumption
------------------------------
The cropper looks for the mask with the same stem as the image file, inside
``mask_dir``.  If your layout differs, pass a custom ``mask_path_fn``:

    cropper = RefugeMaskCropper(
        mask_path_fn=lambda img_path: img_path.with_suffix(".bmp"),
        scale=1.5,
        target_size=(200, 200),
    )
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image


_MASK_DIR_NAMES = {"Disc_Cup_Masks", "Disc_Masks", "Disc_Mask"}
_MASK_SUFFIXES  = {".bmp", ".png"}


class RefugeMaskCropper:
    """
    Crop a fundus image to the optic disc region using its segmentation mask.

    Pass the REFUGE root directory and the cropper will automatically index
    all masks underneath it — no need to specify which subdirectory or
    file extension.

        cropper = RefugeMaskCropper("REFUGE/", scale=1.5)
        loader  = ImageLoader(target_size=(200, 200), preprocessor=cropper)
        imgs    = loader.get_img(test_set)   # test_set = any list of image paths

    Parameters
    ----------
    refuge_root : str or Path
        Top-level REFUGE directory.  All mask files under directories named
        ``Disc_Cup_Masks``, ``Disc_Masks``, or ``Disc_Mask`` are indexed
        automatically (supports both .bmp and .png).
    scale : float
        Padding multiplier applied to the disc radius.
        1.0 = tight crop, 1.5 = moderate context, 2.5 = lots of context.
    target_size : (height, width)
        Output size after cropping.  Should match ``ImageLoader.target_size``.
    """

    def __init__(
        self,
        refuge_root: str | Path,
        *,
        scale: float = 1.5,
        target_size: Tuple[int, int] = (200, 200),
    ) -> None:
        self.refuge_root = Path(refuge_root)
        self.scale       = scale
        self.target_size = target_size
        self._index: dict[str, list[Path]] = {}
        self._build_index()

    def _build_index(self) -> None:
        """Walk refuge_root and index all mask files by stem (stem → [paths])."""
        for mask_dir in self.refuge_root.rglob("*"):
            if mask_dir.is_dir() and mask_dir.name in _MASK_DIR_NAMES:
                for f in mask_dir.rglob("*"):
                    if f.is_file() and f.suffix.lower() in _MASK_SUFFIXES:
                        self._index.setdefault(f.stem, []).append(f)
        if not self._index:
            raise FileNotFoundError(
                f"No mask files found under {self.refuge_root!r}. "
                f"Expected directories named: {_MASK_DIR_NAMES}"
            )
        n_masks = sum(len(v) for v in self._index.values())
        print(f"[RefugeMaskCropper] indexed {n_masks} masks ({len(self._index)} unique stems)", flush=True)

    # ------------------------------------------------------------------
    # Callable interface — drop-in preprocessor for ImageLoader
    # ------------------------------------------------------------------

    def __call__(
        self,
        img: Image.Image,
        img_path: Optional[str | Path] = None,
    ) -> Image.Image:
        stem      = Path(img_path).stem if img_path else None
        mask_path = self._lookup(stem, img_path)
        disc_mask = _load_disc_mask(mask_path, img.size)
        box       = _mask_to_crop_box(disc_mask, scale=self.scale, img_size=img.size)
        cropped   = img.crop(box)
        return cropped.resize(
            (self.target_size[1], self.target_size[0]), Image.Resampling.BILINEAR
        )

    def _lookup(self, stem: Optional[str], img_path: Optional[str | Path] = None) -> Path:
        if stem is None:
            raise ValueError("img_path is required to match the mask.")
        candidates = self._index.get(stem)
        if not candidates:
            raise KeyError(
                f"No mask found for image stem {stem!r}. "
                f"Available stems (sample): {list(self._index)[:5]}"
            )
        if len(candidates) == 1:
            return candidates[0]
        # Pick the mask whose directory components best overlap with img_path
        # (ignores the filename itself to handle extension differences)
        img_parts = set(Path(img_path).parent.parts) if img_path else set()
        return max(candidates, key=lambda m: len(set(m.parent.parts) & img_parts))

    def __repr__(self) -> str:
        return (
            f"RefugeMaskCropper(refuge_root={str(self.refuge_root)!r}, "
            f"scale={self.scale}, target_size={self.target_size}, "
            f"masks_indexed={len(self._index)})"
        )


# ---------------------------------------------------------------------------
# Mask parsing
# ---------------------------------------------------------------------------

def _load_disc_mask(mask_path: Path, img_size: Tuple[int, int]) -> np.ndarray:
    """
    Return a binary disc mask (uint8, 1=disc) from a REFUGE mask file.

    Handles:
      - Grayscale BMP (REFUGE1):  background≈128, disc=255, cup=0
      - RGB PNG (REFUGE2):        background detected from image borders
    """
    mask_img = Image.open(mask_path)

    if mask_img.mode == "L" or mask_img.mode == "P":
        arr = np.asarray(mask_img.convert("L"), dtype=np.uint8)
        bg  = _border_mode(arr)
        disc_mask = (arr != bg).astype(np.uint8)
    else:
        arr = np.asarray(mask_img.convert("RGB"), dtype=np.uint8)
        bg  = _border_mode_rgb(arr)
        # disc = any non-background pixel
        bg_mask   = np.all(arr == bg, axis=2)
        disc_mask = (~bg_mask).astype(np.uint8)

    # Ensure mask matches image spatial size
    mh, mw = disc_mask.shape
    iw, ih = img_size
    if (mw, mh) != (iw, ih):
        disc_img  = Image.fromarray(disc_mask * 255).resize((iw, ih), Image.NEAREST)
        disc_mask = (np.asarray(disc_img) > 0).astype(np.uint8)

    return disc_mask


def _border_mode(arr: np.ndarray, border: int = 5) -> int:
    """Most common pixel value along the image border (grayscale)."""
    h, w   = arr.shape
    border_pixels = np.concatenate([
        arr[:border, :].ravel(),
        arr[-border:, :].ravel(),
        arr[:, :border].ravel(),
        arr[:, -border:].ravel(),
    ])
    return int(Counter(border_pixels.tolist()).most_common(1)[0][0])


def _border_mode_rgb(arr: np.ndarray, border: int = 5) -> np.ndarray:
    """Most common RGB colour along the image border."""
    h, w, _ = arr.shape
    border_pixels = np.concatenate([
        arr[:border, :].reshape(-1, 3),
        arr[-border:, :].reshape(-1, 3),
        arr[:, :border].reshape(-1, 3),
        arr[:, -border:].reshape(-1, 3),
    ], axis=0)
    tuples  = [tuple(row) for row in border_pixels.tolist()]
    most_common = Counter(tuples).most_common(1)[0][0]
    return np.array(most_common, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Bounding box from mask
# ---------------------------------------------------------------------------

def _mask_to_crop_box(
    disc_mask: np.ndarray,
    scale: float,
    img_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """
    Compute a square crop box centred on the disc with padding = scale * radius.

    Returns (left, upper, right, lower) — ready for PIL Image.crop().
    Falls back to the full image if no disc pixels are found.
    """
    coords = np.argwhere(disc_mask > 0)   # (N, 2) in (row, col) order
    if coords.size == 0:
        w, h = img_size
        return (0, 0, w, h)

    ys, xs       = coords[:, 0], coords[:, 1]
    centre_x     = float(xs.mean())
    centre_y     = float(ys.mean())
    radius       = max(float(xs.max() - xs.min()), float(ys.max() - ys.min())) / 2.0
    crop_radius  = radius * scale

    iw, ih = img_size
    left   = int(max(0,  centre_x - crop_radius))
    upper  = int(max(0,  centre_y - crop_radius))
    right  = int(min(iw, centre_x + crop_radius))
    lower  = int(min(ih, centre_y + crop_radius))

    # Make square by expanding the shorter side
    cw, ch = right - left, lower - upper
    if cw < ch:
        diff  = ch - cw
        left  = max(0,  left  - diff // 2)
        right = min(iw, right + diff // 2)
    elif ch < cw:
        diff  = cw - ch
        upper = max(0,  upper - diff // 2)
        lower = min(ih, lower + diff // 2)

    return (left, upper, right, lower)
