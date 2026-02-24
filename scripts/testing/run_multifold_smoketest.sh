#!/usr/bin/env bash
# Quick smoke-test for run_multifold: runs two 1-epoch configs.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MANIFEST="manifest.csv"
IMAGENET_WEIGHTS="models/unet_segmenter/norm_imagenet/best.pt"
if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing $MANIFEST; run scripts/main/refuge/build_manifest.py first." >&2
  exit 1
fi
if [[ ! -f "$IMAGENET_WEIGHTS" ]]; then
  echo "Missing $IMAGENET_WEIGHTS; train the imagenet-normalized UNet first." >&2
  exit 1
fi

COMMON_ARGS=(
  --backbone resnet50
  --fusion-mode fused
  --epochs 1
  --batch-size 4
  --img-crop-manifest "$MANIFEST"
  --img-crop-weights "$IMAGENET_WEIGHTS"
  --img-crop-normalize imagenet
  --shortname smoketest
  --holdout-per-class 12
)

echo "[smoketest] Binary eval, fused head"
python scripts/run_multifold.py \
  "${COMMON_ARGS[@]}" \
  --eval_mode binary \
  --run-id smoketest_binary
echo "→ Results under analysis_data/smoketest/smoketest_binary"

echo "[smoketest] Multiclass eval, fused head"
python scripts/run_multifold.py \
  "${COMMON_ARGS[@]}" \
  --eval_mode multiclass \
  --run-id smoketest_multiclass
echo "→ Results under analysis_data/smoketest/smoketest_multiclass"
