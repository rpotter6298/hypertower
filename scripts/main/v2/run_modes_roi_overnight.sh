#!/usr/bin/env bash
set -euo pipefail

# Runs two back-to-back Hypertower mode comparisons with ROI cropping:
#   1) GT masks
#   2) UNet masks

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

COMMON_ARGS=(
  --eval-modes binary multiclass
  --tower-modes single ensemble bilateral
  --epochs 40
  --n-splits 5
  --batch-size 8
  --backbone refugelike
  --img-crop-manifest manifest.csv
)

echo "[1/2] Starting GT ROI run..."
python3 scripts/basic_analysis/compare_hypertower_modes.py \
  "${COMMON_ARGS[@]}" \
  --img-crop-gt \
  --run-name v2_modes_full_40ep_5fold_roi_gt_holdout

echo "[2/2] Starting UNet ROI run..."
python3 scripts/basic_analysis/compare_hypertower_modes.py \
  "${COMMON_ARGS[@]}" \
  --img-crop-weights models/v2/refuge/segmentation/per_image_refuge_build/best.pt \
  --img-crop-normalize per_image \
  --run-name v2_modes_full_40ep_5fold_roi_unet_perimage_refugebuild_holdout

echo "All runs complete."
