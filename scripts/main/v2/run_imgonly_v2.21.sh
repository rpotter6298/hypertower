#!/usr/bin/env bash
set -euo pipefail

# Image-only runs v2.21 (6 total):
#   No crop:    binary | multiclass
#   GT crop:    binary | multiclass
#   UNet crop:  binary | multiclass
#
# Purpose: isolate the effect of ROI cropping at the single-CNN level,
# without any MD tower contribution (bridge-mode=image_only).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

MANIFEST="manifest.csv"
UNET_WEIGHTS="models/v2/refuge/segmentation/per_image/best.pt"

COMMON=(
  --epochs 40
  --n-splits 5
  --batch-size 8
  --backbone refugelike
  --tower-mode single
  --bridge-mode image_only
  --single-warmup-tower-epochs 4
  --single-warmup-fused-epochs 0
  --img-crop-manifest "$MANIFEST"
)

# ── No crop ──────────────────────────────────────────────────────────────────

echo "[1/6] No crop — binary, image-only..."
python3 scripts/main/v2/run_multifold_v2.py \
  "${COMMON[@]}" --eval-mode binary \
  --run-name v2.21_imgonly_binary_nocrop_40ep_5fold

echo "[2/6] No crop — multiclass, image-only..."
python3 scripts/main/v2/run_multifold_v2.py \
  "${COMMON[@]}" --eval-mode multiclass \
  --run-name v2.21_imgonly_multiclass_nocrop_40ep_5fold

# ── GT crop ──────────────────────────────────────────────────────────────────

echo "[3/6] GT crop — binary, image-only..."
python3 scripts/main/v2/run_multifold_v2.py \
  "${COMMON[@]}" --eval-mode binary --img-crop-gt \
  --run-name v2.21_imgonly_binary_gt_40ep_5fold

echo "[4/6] GT crop — multiclass, image-only..."
python3 scripts/main/v2/run_multifold_v2.py \
  "${COMMON[@]}" --eval-mode multiclass --img-crop-gt \
  --run-name v2.21_imgonly_multiclass_gt_40ep_5fold

# ── UNet crop ────────────────────────────────────────────────────────────────

echo "[5/6] UNet crop — binary, image-only..."
python3 scripts/main/v2/run_multifold_v2.py \
  "${COMMON[@]}" --eval-mode binary \
  --img-crop-weights "$UNET_WEIGHTS" \
  --run-name v2.21_imgonly_binary_unet_40ep_5fold

echo "[6/6] UNet crop — multiclass, image-only..."
python3 scripts/main/v2/run_multifold_v2.py \
  "${COMMON[@]}" --eval-mode multiclass \
  --img-crop-weights "$UNET_WEIGHTS" \
  --run-name v2.21_imgonly_multiclass_unet_40ep_5fold

echo "Image-only v2.21 runs complete."
