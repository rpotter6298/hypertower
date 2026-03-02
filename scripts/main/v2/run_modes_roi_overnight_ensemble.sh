#!/usr/bin/env bash
set -euo pipefail

# Ensemble tower runs (3 total):
#   binary     × ensemble
#   multiclass × ensemble
#   multiclass × ensemble + balanced sampling

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
  --tower-modes ensemble
)

# Runs 1-2: binary + multiclass ensemble (no balanced sampling)
echo "[1/2] GT ROI — binary + multiclass, ensemble..."
python3 scripts/basic_analysis/compare_hypertower_modes.py \
  "${COMMON_ARGS[@]}" \
  "${CROP_ARGS[@]}" \
  --eval-modes binary multiclass \
  --run-name v2_modes_gt_40ep_5fold_ensemble_v2

# Run 3: multiclass ensemble + balanced sampling
# (reuse the crop cache built above)
echo "[2/2] GT ROI — multiclass, ensemble, balanced sampling..."
python3 scripts/basic_analysis/compare_hypertower_modes.py \
  "${COMMON_ARGS[@]}" \
  "${CROP_ARGS[@]}" \
  --eval-modes multiclass \
  --balanced-sampling \
  --persist-img-crop-cache \
  --run-name v2_modes_gt_40ep_5fold_ensemble_balanced_v2

echo "Ensemble runs complete."
