#!/usr/bin/env bash
set -euo pipefail

# Quick smoke test for ROI mode runs:
#   1) GT masks
#   2) UNet masks
# Uses 1 epoch and 1 fold for fast validation.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

COMMON_ARGS=(
  --eval-modes binary multiclass
  --tower-modes single ensemble bilateral
  --epochs 1
  --n-splits 2
  --folds 1
  --batch-size 8
  --backbone refugelike
  --img-crop-manifest manifest.csv
  --warmup-tower-epochs 0
  --warmup-fused-epochs 0
  --single-warmup-tower-epochs 0
  --single-warmup-fused-epochs 0
  --bilat-warmup-tower-epochs 0
  --bilat-warmup-fused-epochs 0
)

echo "[smoke 1/2] Starting GT ROI run..."
python3 scripts/basic_analysis/compare_hypertower_modes.py \
  "${COMMON_ARGS[@]}" \
  --img-crop-gt \
  --run-name smoke_v2_modes_roi_gt

echo "[smoke 2/2] Starting UNet ROI run..."
python3 scripts/basic_analysis/compare_hypertower_modes.py \
  "${COMMON_ARGS[@]}" \
  --img-crop-weights models/v2/refuge/segmentation/per_image_refuge_build/best.pt \
  --img-crop-normalize per_image \
  --run-name smoke_v2_modes_roi_unet_perimage

echo "Smoke runs complete."
