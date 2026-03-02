#!/usr/bin/env bash
set -euo pipefail

# Bilateral tower runs (3 total):
#   binary     × bilateral
#   multiclass × bilateral
#   multiclass × bilateral + balanced sampling

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

CROP_ARGS=(
  --img-crop-manifest manifest.csv
  --img-crop-gt
)

COMMON_ARGS=(
  --epochs 40
  --n-splits 5
  --batch-size 8
  --backbone refugelike
  --tower-modes bilateral
)

# Runs 1-2: binary + multiclass bilateral (no balanced sampling)
echo "[1/2] GT ROI — binary + multiclass, bilateral..."
python3 scripts/basic_analysis/compare_hypertower_modes.py \
  "${COMMON_ARGS[@]}" \
  "${CROP_ARGS[@]}" \
  --eval-modes binary multiclass \
  --run-name v2_modes_gt_40ep_5fold_bilateral_v2

# Run 3: multiclass bilateral + balanced sampling
# (reuse the crop cache built above)
echo "[2/2] GT ROI — multiclass, bilateral, balanced sampling..."
python3 scripts/basic_analysis/compare_hypertower_modes.py \
  "${COMMON_ARGS[@]}" \
  "${CROP_ARGS[@]}" \
  --eval-modes multiclass \
  --balanced-sampling \
  --persist-img-crop-cache \
  --run-name v2_modes_gt_40ep_5fold_bilateral_balanced_v2

echo "Bilateral runs complete."
