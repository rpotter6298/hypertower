#!/usr/bin/env bash
set -e

echo "=== v2.3 fused binary nocrop ==="
python scripts/basic_analysis/compare_hypertower_modes.py \
  --tower-modes ensemble --eval-modes binary \
  --epochs 40 --n-splits 5 \
  --backbone refugelike \
  --img-crop-manifest analysis_data/unet_manifest.csv \
  --warmup-md-epochs 50 \
  --fused-head \
  --run-name v2.3_fused_binary_nocrop

echo "=== v2.3 fused multiclass nocrop ==="
python scripts/basic_analysis/compare_hypertower_modes.py \
  --tower-modes ensemble --eval-modes multiclass \
  --epochs 40 --n-splits 5 \
  --backbone refugelike \
  --img-crop-manifest analysis_data/unet_manifest.csv \
  --warmup-md-epochs 50 \
  --fused-head \
  --run-name v2.3_fused_multiclass_nocrop

echo "=== done ==="
