"""Generate ground-truth mask overlays for REFUGE and Papila samples."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
from collections import Counter
from PIL import Image
from PIL.Image import Resampling

from classes.unet_segmenter import UNetSegmenter

REFUGE_ROOT = Path("REFUGE")
DEFAULT_MANIFEST = Path("manifest.csv")
OUTPUT_DIR = Path("temp/gt_test")


def to_mask_colors(disc: np.ndarray, cup: np.ndarray) -> Image.Image:
    h, w = disc.shape
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    disc_mask = disc.astype(bool)
    cup_mask = cup.astype(bool)
    canvas[disc_mask] = [128, 128, 128]
    canvas[cup_mask] = [0, 0, 0]
    return Image.fromarray(canvas)


def overlay(
    original: Image.Image, mask_rgb: Image.Image, alpha: float = 0.6
) -> Image.Image:
    mask_rgba = mask_rgb.convert("RGBA")
    updates = np.array(mask_rgba, dtype=np.float32)
    updates[..., 3] = alpha * 255 * (updates[..., :3] != 255).any(axis=-1)
    base = original.convert("RGBA")
    return Image.alpha_composite(
        base, Image.fromarray(updates.astype(np.uint8))
    ).convert("RGB")


def original_mask_to_rgb(mask_path: Path) -> Image.Image:
    mask_img = Image.open(mask_path)
    arr = np.asarray(mask_img)
    h, w = arr.shape[:2]
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

    if arr.ndim == 2:
        border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
        bg_value = Counter(border.tolist()).most_common(1)[0][0]
        disc_mask = arr != bg_value
        fg_counts = Counter(arr[arr != bg_value].flatten())
        if fg_counts:
            # For REFUGE-style masks: cup should be the darkest (minimum value)
            cup_value = min(fg_counts.keys())
            cup_mask = arr == cup_value
        else:
            cup_mask = np.zeros_like(arr, dtype=bool)
    else:
        edges = np.concatenate(
            [arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]], axis=0
        )
        bg_color = Counter(map(tuple, edges)).most_common(1)[0][0]
        disc_mask = ~np.all(arr == bg_color, axis=-1)
        color_counts = Counter(map(tuple, arr.reshape(-1, arr.shape[2])))
        cup_mask = np.zeros((h, w), dtype=bool)
        candidates = {}
        for color, count in color_counts.items():
            if color == bg_color:
                continue
            mask = np.all(arr == color, axis=-1)
            candidates[color] = mask
        if candidates:
            # For REFUGE-style masks: cup should be the darkest color (closest to black)
            cup_color = min(candidates.keys(), key=lambda color: sum(color))
            cup_mask = candidates[cup_color]
        disc_mask = disc_mask.astype(bool)
    cup_mask = cup_mask & disc_mask

    canvas[disc_mask] = [128, 128, 128]
    canvas[cup_mask] = [0, 0, 0]
    return Image.fromarray(canvas)


def process_entries(
    segmenter: UNetSegmenter, entries, prefix: str, count: int, dest: Path
) -> None:
    for entry in entries[:count]:
        img_path = Path(entry.image_path)
        if not img_path.exists():
            continue
        orig = Image.open(img_path).convert("RGB")
        image = segmenter.preprocess_image(orig)
        disc, cup = segmenter.load_masks(entry)
        disc_coords = segmenter._mask_to_coords(disc)
        cup_coords = segmenter._mask_to_coords(cup)
        print(
            f"{entry.sample_id}: disc coords {disc_coords.shape[0] if disc_coords is not None else 0}, "
            f"cup coords {cup_coords.shape[0] if cup_coords is not None else 0}"
        )
        mask_rgb = to_mask_colors(disc, cup)
        overlay_img = overlay(image, mask_rgb)
        mask_rgb.save(dest / f"{prefix}_{entry.sample_id}_mask.png")
        overlay_img.save(dest / f"{prefix}_{entry.sample_id}_overlay.png")

        if entry.annotation_type_disc == "mask":
            gt_mask_rgb = original_mask_to_rgb(entry.annotation_disc)
            gt_overlay = overlay(
                orig.resize(gt_mask_rgb.size, Resampling.BILINEAR), gt_mask_rgb
            )
            gt_mask_rgb.save(dest / f"{prefix}_{entry.sample_id}_gt_mask.png")
            gt_overlay.save(dest / f"{prefix}_{entry.sample_id}_gt_overlay.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect ground-truth masks for REFUGE and Papila"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--count", type=int, default=10, help="Number of samples per dataset"
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    segmenter = UNetSegmenter(args.manifest)
    refuge_entries = [e for e in segmenter._manifest if e.dataset == "refuge"]
    papila_entries = [e for e in segmenter._manifest if e.dataset == "papila"]

    process_entries(segmenter, refuge_entries, "refuge", args.count, args.output)
    process_entries(segmenter, papila_entries, "papila", args.count, args.output)
    print(f"Saved overlays to {args.output}")


if __name__ == "__main__":
    main()
