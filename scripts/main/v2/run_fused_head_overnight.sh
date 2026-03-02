#!/usr/bin/env bash
set -euo pipefail

# Fused-head ensemble runs (2 total) — UNet ROI crop:
#   binary     × ensemble + fused head
#   multiclass × ensemble + fused head

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

CROP_ARGS=(
  --img-crop-manifest manifest.csv
  --img-crop-weights models/v2/refuge/segmentation/per_image_refuge_build/best.pt
  --img-crop-normalize per_image
)

COMMON_ARGS=(
  --epochs 40
  --n-splits 5
  --batch-size 8
  --backbone refugelike
  --tower-mode ensemble
  --fused-head
  --fusion-epochs 10
)

echo "[1/2] UNet ROI — binary, ensemble + fused head..."
python3 scripts/main/v2/run_multifold_v2.py \
  "${COMMON_ARGS[@]}" \
  "${CROP_ARGS[@]}" \
  --eval-mode binary \
  --run-name v2_ensemble_fused_binary_unet_40ep_5fold_v1

echo "[2/2] UNet ROI — multiclass, ensemble + fused head..."
python3 scripts/main/v2/run_multifold_v2.py \
  "${COMMON_ARGS[@]}" \
  "${CROP_ARGS[@]}" \
  --eval-mode multiclass \
  --run-name v2_ensemble_fused_multiclass_unet_40ep_5fold_v1

echo "Fused-head runs complete."
