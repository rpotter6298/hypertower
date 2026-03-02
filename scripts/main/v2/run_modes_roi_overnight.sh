#!/usr/bin/env bash
set -euo pipefail

# Six V2HyperTower runs using GT ROI crop:
#   binary  × {ensemble, bilateral}               (runs 1-2)
#   multiclass × {ensemble, bilateral}             (runs 3-4)
#   multiclass × {ensemble, bilateral} + balanced  (runs 5-6)

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
)

# Runs 1-4: binary + multiclass, ensemble + bilateral, no balanced sampling
echo "[1/2] GT ROI — binary + multiclass, ensemble + bilateral..."
python3 scripts/basic_analysis/compare_hypertower_modes.py \
  "${COMMON_ARGS[@]}" \
  "${CROP_ARGS[@]}" \
  --eval-modes binary multiclass \
  --tower-modes ensemble bilateral \
  --run-name v2_modes_gt_40ep_5fold_no_single_v2

# Runs 5-6: multiclass only, ensemble + bilateral, balanced sampling
# (reuse the crop cache built during runs 1-4)
echo "[2/2] GT ROI — multiclass, ensemble + bilateral, balanced sampling..."
python3 scripts/basic_analysis/compare_hypertower_modes.py \
  "${COMMON_ARGS[@]}" \
  "${CROP_ARGS[@]}" \
  --eval-modes multiclass \
  --tower-modes ensemble bilateral \
  --balanced-sampling \
  --persist-img-crop-cache \
  --run-name v2_modes_gt_40ep_5fold_multiclass_balanced_v2

echo "All runs complete."
