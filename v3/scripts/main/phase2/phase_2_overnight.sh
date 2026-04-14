#!/usr/bin/env bash
# Phase 2 overnight batch — image-only ResNet50, 10x5 rep-CV
# Runs 6 configurations:
#   1. Leaky CV (eye-level splits)
#   2. Proper CV (patient-level, baseline)
#   3. GT crop  scale=1.1  (paper-matched tight crop)
#   4. GT crop  scale=2.5  (default generous crop)
#   5. UNet crop scale=1.1
#   6. UNet crop scale=2.5
set -euo pipefail

SCRIPT="python -m v3.scripts.main.run_cv"
OUTROOT="v3/results/phase2"
MANIFEST="manifest.csv"
UNET_WEIGHTS="models/v2/refuge/segmentation/per_image/best.pt"

BASE="--eval-mode binary \
      --tower-mode single \
      --bridge-mode image_only \
      --backbone resnet50 \
      --epochs 30 \
      --augment \
      --in-memory-cache \
      --reps 10 \
      --rep-seed-start 100 \
      --rep-seed-step 100 \
      --output-root ${OUTROOT}"

echo "============================================================"
echo " Phase 2 overnight batch"
echo " $(date)"
echo "============================================================"

# ----------------------------------------------------------------
# 1. Leaky CV (eye-level splits, no crop)
# ----------------------------------------------------------------
echo ""
echo "=== [1/6] Leaky CV (eye-level) ==="
$SCRIPT $BASE \
    --leaky-cv \
    --run-name imageonly_resnet50_leaky

# ----------------------------------------------------------------
# 2. Proper CV (patient-level, no crop) — baseline
# ----------------------------------------------------------------
echo ""
echo "=== [2/6] Proper CV (patient-level, baseline) ==="
$SCRIPT $BASE \
    --run-name imageonly_resnet50_proper

# ----------------------------------------------------------------
# 3. GT crop, scale=1.1  (paper-matched tight crop)
# ----------------------------------------------------------------
echo ""
echo "=== [3/6] GT crop, scale=1.1 ==="
$SCRIPT $BASE \
    --img-crop-gt \
    --img-crop-manifest ${MANIFEST} \
    --img-crop-scale 1.1 \
    --img-crop-size 200 \
    --run-name imageonly_resnet50_gtcrop_1.1

# ----------------------------------------------------------------
# 4. GT crop, scale=2.5  (default generous crop)
# ----------------------------------------------------------------
echo ""
echo "=== [4/6] GT crop, scale=2.5 ==="
$SCRIPT $BASE \
    --img-crop-gt \
    --img-crop-manifest ${MANIFEST} \
    --img-crop-scale 2.5 \
    --img-crop-size 200 \
    --run-name imageonly_resnet50_gtcrop_2.5

# ----------------------------------------------------------------
# 5. UNet crop, scale=1.1
# ----------------------------------------------------------------
echo ""
echo "=== [5/6] UNet crop, scale=1.1 ==="
$SCRIPT $BASE \
    --img-crop-weights ${UNET_WEIGHTS} \
    --img-crop-manifest ${MANIFEST} \
    --img-crop-scale 1.1 \
    --img-crop-size 200 \
    --run-name imageonly_resnet50_unetcrop_1.1

# ----------------------------------------------------------------
# 6. UNet crop, scale=2.5
# ----------------------------------------------------------------
echo ""
echo "=== [6/6] UNet crop, scale=2.5 ==="
$SCRIPT $BASE \
    --img-crop-weights ${UNET_WEIGHTS} \
    --img-crop-manifest ${MANIFEST} \
    --img-crop-scale 2.5 \
    --img-crop-size 200 \
    --run-name imageonly_resnet50_unetcrop_2.5

# ----------------------------------------------------------------
# 7. Refugelike backbone (proper CV, no crop) — pre-training effect
# ----------------------------------------------------------------
echo ""
echo "=== [7/7] Refugelike backbone (proper CV, no crop) ==="
$SCRIPT $BASE \
    --backbone refugelike \
    --run-name imageonly_refugelike_proper

echo ""
echo "============================================================"
echo " All done — $(date)"
echo "============================================================"
