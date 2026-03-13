#!/usr/bin/env python3
"""Compare cached crop bounds/features vs GT-derived crops from the manifest."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classes.hypertower import ManifestImageCropper, UNetImageCropper
from classes.refuge_segmentation import UNet as RefugeUNet


# ---------------------------
# Config (edit in IDE)
# ---------------------------
CACHE_DIR = Path("analysis_data/hypertower_crops")
MANIFEST_PATH = Path("manifest.csv")
IMAGE_DIR = Path("Papila/FundusImages")
SCALE = 2.5
MAX_SAMPLES = 200  # set None to scan all
TOL_BOUNDS = 1.0  # pixels
TOL_FEATURES = 1e-3
UNET_VARIANTS = [
    ("norm_imagenet", Path("models/unet_segmenter/norm_imagenet/best.pt"), "imagenet"),
    ("normalize_none", Path("models/unet_segmenter/normalize_none/best.pt"), "none"),
    ("norm_per_image", Path("models/unet_segmenter/norm_per_image/best.pt"), "per_image"),
]
REFUGE_SEG_WEIGHTS = Path("models/refuge/segmentation/refuge_segmentation_best.pt")


def _load_cache(path: Path) -> Optional[Dict[str, np.ndarray]]:
    try:
        data = np.load(path, allow_pickle=False)
    except Exception:
        return None
    return {k: data[k] for k in data.files}


def _parse_stem(path: Path) -> str:
    # expects RET###OS_s250.npz -> RET###OS
    stem = path.stem
    if "_s" in stem:
        stem = stem.split("_s")[0]
    return stem


def _image_path_from_stem(stem: str) -> Optional[Path]:
    cand = IMAGE_DIR / f"{stem}.jpg"
    if cand.exists():
        return cand
    cand = IMAGE_DIR / f"{stem}.png"
    if cand.exists():
        return cand
    return None


def _gt_info(
    cropper: ManifestImageCropper, image_path: Path
) -> Optional[Dict[str, float]]:
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return None
    info = cropper._compute_crop_info(image, image_path)
    return info


def _unet_info(
    cropper: UNetImageCropper, image_path: Path
) -> Optional[Dict[str, float]]:
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return None
    info = cropper._compute_crop_info(image, image_path)
    return info


def _load_refuge_model(device: str) -> Optional[RefugeUNet]:
    if not REFUGE_SEG_WEIGHTS.exists():
        return None
    model = RefugeUNet()
    try:
        state = torch.load(REFUGE_SEG_WEIGHTS, map_location=device)
    except Exception:
        return None
    state_dict = state.get("model", state) if isinstance(state, dict) else state
    try:
        model.load_state_dict(state_dict)
    except Exception:
        return None
    model.to(device)
    model.eval()
    return model


def _refuge_seg_info(
    model: RefugeUNet, device: str, image_path: Path
) -> Optional[Dict[str, float]]:
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return None
    original_size = image.size
    image_resized = image.resize((512, 512), Image.BILINEAR)
    tensor = transforms.ToTensor()(image_resized).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        mask = torch.sigmoid(logits)[0, 0]
    mask_np = (mask.cpu().numpy() > 0.5).astype(np.float32)
    mask_img = Image.fromarray(mask_np)
    mask_img = mask_img.resize(original_size, Image.NEAREST)
    mask_np = np.array(mask_img, dtype=np.float32)
    coords = np.argwhere(mask_np > 0.5)
    if coords.size == 0:
        return None
    ys, xs = coords[:, 0], coords[:, 1]
    centre_x = float(xs.mean())
    centre_y = float(ys.mean())
    width = float(xs.max() - xs.min())
    height = float(ys.max() - ys.min())
    diameter = max(width, height)
    radius = diameter / 2.0
    crop_radius = radius * SCALE
    left = max(0.0, centre_x - crop_radius)
    upper = max(0.0, centre_y - crop_radius)
    right = min(float(image.width), centre_x + crop_radius)
    lower = min(float(image.height), centre_y + crop_radius)
    return {
        "left": left,
        "upper": upper,
        "right": right,
        "lower": lower,
    }


def _diff_bounds(cache: Dict[str, np.ndarray], gt: Dict[str, float]) -> Optional[float]:
    keys = ("left", "upper", "right", "lower")
    if not all(k in cache for k in keys):
        return None
    diffs = [abs(float(cache[k]) - float(gt[k])) for k in keys]
    return float(max(diffs))


def _diff_features(
    cache: Dict[str, np.ndarray], gt: Dict[str, float]
) -> Optional[float]:
    if "features" not in cache or "features" not in gt:
        return None
    cf = np.asarray(cache["features"], dtype=float).ravel()
    gf = np.asarray(gt["features"], dtype=float).ravel()
    if cf.shape != gf.shape:
        return None
    return float(np.max(np.abs(cf - gf)))


def main() -> None:
    if not CACHE_DIR.exists():
        raise SystemExit(f"Cache dir not found: {CACHE_DIR}")
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"Manifest not found: {MANIFEST_PATH}")

    cache_files = sorted(CACHE_DIR.glob(f"*_s{int(SCALE * 100)}.npz"))
    if MAX_SAMPLES is not None:
        cache_files = cache_files[:MAX_SAMPLES]
    print(f"[debug] cache files found: {len(cache_files)}")

    try:
        manifest_df = pd.read_csv(MANIFEST_PATH)
    except Exception as exc:
        raise SystemExit(f"Failed to read manifest: {exc}")
    manifest_images = manifest_df.get("image_path")
    if manifest_images is None:
        raise SystemExit("Manifest is missing image_path column.")
    manifest_images = manifest_images.dropna().astype(str)
    manifest_stems = {Path(p).stem for p in manifest_images}
    print(f"[debug] manifest image_path count: {len(manifest_images)}")
    print(f"[debug] manifest unique stems: {len(manifest_stems)}")

    cache_stems = {_parse_stem(p) for p in cache_files}
    overlap = cache_stems & manifest_stems
    print(
        f"[debug] cache stems: {len(cache_stems)} overlap with manifest stems: {len(overlap)}"
    )
    if cache_files:
        print(f"[debug] example cache stems: {sorted(list(cache_stems))[:5]}")
    if manifest_stems:
        print(f"[debug] example manifest stems: {sorted(list(manifest_stems))[:5]}")

    cropper = ManifestImageCropper(
        manifest_path=MANIFEST_PATH,
        scale=SCALE,
        target_size=224,
        cache_dir=None,
    )

    rows: List[Dict[str, object]] = []
    for cache_path in cache_files:
        cache = _load_cache(cache_path)
        if cache is None:
            continue
        stem = _parse_stem(cache_path)
        image_path = _image_path_from_stem(stem)
        if image_path is None:
            continue

        gt = _gt_info(cropper, image_path)
        if gt is None:
            continue

        bounds_diff = _diff_bounds(cache, gt)
        feat_diff = _diff_features(cache, gt)

        rows.append(
            {
                "file": cache_path.name,
                "bounds_diff": bounds_diff,
                "features_diff": feat_diff,
                "bounds_match": bounds_diff is not None and bounds_diff <= TOL_BOUNDS,
                "features_match": feat_diff is not None and feat_diff <= TOL_FEATURES,
            }
        )

    if not rows:
        print("[warn] No cache entries matched GT manifest entries.")
    else:
        df = pd.DataFrame(rows)
        print(df.head(10).to_string(index=False))
        print("\nSummary:")
        print(df[["bounds_diff", "features_diff"]].describe().to_string())
        if df["bounds_match"].notna().any():
            match_rate = df["bounds_match"].mean()
            print(f"\nBounds match rate (<= {TOL_BOUNDS}px): {match_rate:.3f}")
        if df["features_match"].notna().any():
            match_rate = df["features_match"].mean()
            print(f"Features match rate (<= {TOL_FEATURES}): {match_rate:.3f}")

    print("\nUNet variant comparisons (no cache writes):")
    for name, weights, normalize in UNET_VARIANTS:
        if not weights.exists():
            print(f"[warn] {name}: weights not found at {weights}")
            continue

        unet = UNetImageCropper(
            manifest_path=MANIFEST_PATH,
            weights_path=weights,
            normalize=normalize,
            threshold=0.5,
            tta=False,
            scale=SCALE,
            target_size=224,
            cache_dir=None,  # ensure no cache writes
        )

        u_rows: List[Dict[str, object]] = []
        missing_images = 0
        unet_none = 0
        cache_missing = 0
        exceptions = 0
        for cache_path in cache_files:
            cache = _load_cache(cache_path)
            if cache is None:
                cache_missing += 1
                continue
            stem = _parse_stem(cache_path)
            image_path = _image_path_from_stem(stem)
            if image_path is None:
                missing_images += 1
                continue
            try:
                info = _unet_info(unet, image_path)
            except Exception:
                exceptions += 1
                continue
            if info is None:
                unet_none += 1
                continue
            bounds_diff = _diff_bounds(cache, info)
            feat_diff = _diff_features(cache, info)
            u_rows.append(
                {
                    "bounds_diff": bounds_diff,
                    "features_diff": feat_diff,
                    "bounds_match": bounds_diff is not None
                    and bounds_diff <= TOL_BOUNDS,
                    "features_match": feat_diff is not None
                    and feat_diff <= TOL_FEATURES,
                }
            )

        if not u_rows:
            print(
                f"[warn] {name}: no comparisons computed "
                f"(cache_missing={cache_missing}, missing_images={missing_images}, "
                f"unet_none={unet_none}, exceptions={exceptions})"
            )
            continue
        u_df = pd.DataFrame(u_rows)
        b_mean = float(u_df["bounds_diff"].mean())
        f_mean = float(u_df["features_diff"].mean())
        b_match = float(u_df["bounds_match"].mean())
        f_match = float(u_df["features_match"].mean())

        print(
            f"{name}: mean bounds diff={b_mean:.3f}, mean feat diff={f_mean:.6f}, "
            f"bounds match rate={b_match:.3f}, features match rate={f_match:.3f}"
        )

    print("\nRefuge segmentation model comparison (bounds only, no cache writes):")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    refuge_model = _load_refuge_model(device)
    if refuge_model is None:
        print(f"[warn] refuge_segmentation_best.pt not found or failed to load at {REFUGE_SEG_WEIGHTS}")
        return

    r_rows: List[Dict[str, object]] = []
    missing_images = 0
    cache_missing = 0
    model_none = 0
    exceptions = 0
    for cache_path in cache_files:
        cache = _load_cache(cache_path)
        if cache is None:
            cache_missing += 1
            continue
        stem = _parse_stem(cache_path)
        image_path = _image_path_from_stem(stem)
        if image_path is None:
            missing_images += 1
            continue
        try:
            info = _refuge_seg_info(refuge_model, device, image_path)
        except Exception:
            exceptions += 1
            continue
        if info is None:
            model_none += 1
            continue
        bounds_diff = _diff_bounds(cache, info)
        r_rows.append(
            {
                "bounds_diff": bounds_diff,
                "bounds_match": bounds_diff is not None and bounds_diff <= TOL_BOUNDS,
            }
        )

    if not r_rows:
        print(
            "[warn] refuge_segmentation_best: no comparisons computed "
            f"(cache_missing={cache_missing}, missing_images={missing_images}, "
            f"model_none={model_none}, exceptions={exceptions})"
        )
        return

    r_df = pd.DataFrame(r_rows)
    b_mean = float(r_df["bounds_diff"].mean())
    b_match = float(r_df["bounds_match"].mean())
    print(
        f"refuge_segmentation_best: mean bounds diff={b_mean:.3f}, "
        f"bounds match rate={b_match:.3f}"
    )


if __name__ == "__main__":
    main()
