#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"

/home/rpotter/miniconda3/envs/fundus_imaging/bin/python -m v4.figures.F8_explainability \
  --only-gradcam \
  --n-grid 16 \
  --alpha 0.45
