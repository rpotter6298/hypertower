#!/usr/bin/env bash
set -euo pipefail

# Multiclass runs v2.2 (6 total):
#   UNet crop:  single | ensemble | fused head
#   GT crop:    single | ensemble | fused head

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

MANIFEST="manifest.csv"
UNET_WEIGHTS="models/v2/refuge/segmentation/per_image/best.pt"

COMMON=(
  --epochs 40
  --n-splits 5
  --batch-size 8
  --backbone refugelike
  --eval-mode multiclass
  --img-crop-manifest "$MANIFEST"
)

UNET_CROP=(
  --img-crop-weights "$UNET_WEIGHTS"
)

GT_CROP=(
  --img-crop-gt
)

# ── UNet crop ────────────────────────────────────────────────────────────────

echo "[1/6] UNet crop — multiclass, single..."
python3 scripts/main/v2/run_multifold_v2.py \
  "${COMMON[@]}" "${UNET_CROP[@]}" \
  --tower-mode single \
  --run-name v2.2_single_multiclass_unet_40ep_5fold

echo "[2/6] UNet crop — multiclass, ensemble..."
python3 scripts/main/v2/run_multifold_v2.py \
  "${COMMON[@]}" "${UNET_CROP[@]}" \
  --tower-mode ensemble \
  --run-name v2.2_ensemble_multiclass_unet_40ep_5fold

echo "[3/6] UNet crop — multiclass, ensemble + fused head..."
python3 scripts/main/v2/run_multifold_v2.py \
  "${COMMON[@]}" "${UNET_CROP[@]}" \
  --tower-mode ensemble \
  --fused-head --fusion-epochs 10 \
  --run-name v2.2_fused_multiclass_unet_40ep_5fold

# ── GT crop ──────────────────────────────────────────────────────────────────

echo "[4/6] GT crop — multiclass, single..."
python3 scripts/main/v2/run_multifold_v2.py \
  "${COMMON[@]}" "${GT_CROP[@]}" \
  --tower-mode single \
  --run-name v2.2_single_multiclass_gt_40ep_5fold

echo "[5/6] GT crop — multiclass, ensemble..."
python3 scripts/main/v2/run_multifold_v2.py \
  "${COMMON[@]}" "${GT_CROP[@]}" \
  --tower-mode ensemble \
  --run-name v2.2_ensemble_multiclass_gt_40ep_5fold

echo "[6/6] GT crop — multiclass, ensemble + fused head..."
python3 scripts/main/v2/run_multifold_v2.py \
  "${COMMON[@]}" "${GT_CROP[@]}" \
  --tower-mode ensemble \
  --fused-head --fusion-epochs 10 \
  --run-name v2.2_fused_multiclass_gt_40ep_5fold

echo "Multiclass v2.2 runs complete."
