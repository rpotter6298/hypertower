"""Shared helpers for deriving disc/cup geometry features."""

from __future__ import annotations

from collections import Counter
from typing import Tuple

import numpy as np
from PIL import Image

EPS = 1e-6
FEATURE_DIM = 5


def disc_cup_from_mask_image(mask_img: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    """Return binary disc/cup masks from a REFUGE-style annotation image."""
    arr = np.asarray(mask_img)
    if arr.ndim == 3:
        h, w, c = arr.shape
        border = np.concatenate(
            [arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]],
            axis=0,
        )
        border_counts = Counter(map(tuple, border))
        bg_color = border_counts.most_common(1)[0][0]
        flat = arr.reshape(-1, c)
        colors = Counter(map(tuple, flat))
        colors.pop(bg_color, None)
        disc = (~np.all(arr == bg_color, axis=-1)).astype(np.uint8)
        if colors:
            cup_color = min(colors.keys(), key=lambda col: sum(col))
            cup = np.all(arr == cup_color, axis=-1).astype(np.uint8)
        else:
            cup = np.zeros((h, w), dtype=np.uint8)
    else:
        border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
        counts = Counter(border.tolist())
        bg_value = counts.most_common(1)[0][0]
        disc = (arr != bg_value).astype(np.uint8)
        fg = arr[arr != bg_value]
        if fg.size > 0:
            cup_value = int(np.min(fg))
            cup = (arr == cup_value).astype(np.uint8)
        else:
            cup = np.zeros_like(arr, dtype=np.uint8)
    cup = (cup > 0) & (disc > 0)
    return disc.astype(np.uint8), cup.astype(np.uint8)


def compute_geometry_features(disc_mask: np.ndarray, cup_mask: np.ndarray) -> np.ndarray:
    """Compute cup/disc geometry descriptors (area, rim, diameter ratios, centre shift)."""
    disc = (disc_mask > 0).astype(np.float32)
    cup = (cup_mask > 0).astype(np.float32)

    disc_area = disc.sum()
    cup_area = cup.sum()
    area_ratio = cup_area / (disc_area + EPS)
    rim_ratio = (disc_area - cup_area) / (disc_area + EPS)

    disc_rows = np.any(disc > 0, axis=1)
    cup_rows = np.any(cup > 0, axis=1)
    disc_cols = np.any(disc > 0, axis=0)
    cup_cols = np.any(cup > 0, axis=0)

    disc_height = float(disc_rows.sum())
    cup_height = float(cup_rows.sum())
    disc_width = float(disc_cols.sum())
    cup_width = float(cup_cols.sum())

    vertical_ratio = cup_height / (disc_height + EPS)
    horizontal_ratio = cup_width / (disc_width + EPS)

    def _centre(mask: np.ndarray) -> Tuple[float, float]:
        coords = np.argwhere(mask > 0)
        if coords.size == 0:
            return 0.5, 0.5
        ys, xs = coords[:, 0], coords[:, 1]
        return float(xs.mean()) / mask.shape[1], float(ys.mean()) / mask.shape[0]

    disc_cx, disc_cy = _centre(disc)
    cup_cx, cup_cy = _centre(cup)
    centre_shift = float(np.hypot(cup_cx - disc_cx, cup_cy - disc_cy))

    return np.array(
        [area_ratio, rim_ratio, vertical_ratio, horizontal_ratio, centre_shift],
        dtype=np.float32,
    )
