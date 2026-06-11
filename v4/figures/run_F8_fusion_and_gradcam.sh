#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-/home/rpotter/miniconda3/envs/fundus_imaging/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"

FUSION_SPLIT="${FUSION_SPLIT:-test}"
GRADCAM_GRID="${GRADCAM_GRID:-16}"
GRADCAM_ALPHA="${GRADCAM_ALPHA:-0.45}"

FUSION_SOURCE="${FUSION_SOURCE:-v3}"
GRADCAM_SOURCE="${GRADCAM_SOURCE:-v3}"

echo "Running F8a fusion event panel from ${FUSION_SOURCE} ${FUSION_SPLIT} predictions..."
"${PYTHON_BIN}" -m v4.figures.F8_explainability \
  --only-fusion \
  --fusion-source "${FUSION_SOURCE}" \
  --fusion-split "${FUSION_SPLIT}"

echo "Running oriented GradCAM outputs from ${GRADCAM_SOURCE}..."
"${PYTHON_BIN}" -m v4.figures.F8_explainability \
  --only-gradcam \
  --gradcam-source "${GRADCAM_SOURCE}" \
  --n-grid "${GRADCAM_GRID}" \
  --alpha "${GRADCAM_ALPHA}"
