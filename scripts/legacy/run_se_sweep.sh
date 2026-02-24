#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_se_sweep.sh --epochs 50 --n-splits 5 --batch-size 8 --eval_mode multiclass [extra args]
#
# This will launch a series of runs covering the grid from the slide:
# - Bridge-only R8/R16/R32 (normalized and non-normalized)
# - Tower-only  R8/R16/R32 (normalized and non-normalized)
# - Tower+Bridge R8/R16/R32 (normalized and non-normalized)

ARGS=("$@")

run() {
  local SHORT="$1"; shift
  echo "=== Running: $SHORT ==="
  # Skip if a summary for this shortname already exists
  if ls "analysis_data/${SHORT}_"*.md >/dev/null 2>&1; then
    echo "… skipping ${SHORT} (summary already present)"
    return 0
  fi
  python3 scripts/run_multifold.py \
    --shortname "$SHORT" \
    "$@" \
    "${ARGS[@]}" || true
}

# Bridge-only (normalized + non-normalized)
for R in 8 16 32; do
  run "se_bridge_R${R}_norm" --se-where bridge --se-reduction ${R} --se-pre-norm --checkpoint-best
  run "se_bridge_R${R}_nonorm" --se-where bridge --se-reduction ${R} --no-se-pre-norm --checkpoint-best
done

# Tower-only (normalized + non-normalized)
for R in 8 16 32; do
  run "se_tower_R${R}_norm" \
    --se-where tower --se-reduction-tower ${R} --se-pre-norm-tower \
    --checkpoint-best
  run "se_tower_R${R}_nonorm" \
    --se-where tower --se-reduction-tower ${R} --no-se-pre-norm-tower \
    --checkpoint-best
done

# Tower+Bridge (normalized + non-normalized)
for R in 8 16 32; do
  # normalized (both pre-norm on)
  run "se_tower_bridge_R${R}_norm" \
    --se-where both --se-reduction ${R} --se-reduction-tower ${R} \
    --se-pre-norm --se-pre-norm-tower --checkpoint-best
  # non-normalized (both pre-norm off)
  run "se_tower_bridge_R${R}_nonorm" \
    --se-where both --se-reduction ${R} --se-reduction-tower ${R} \
    --no-se-pre-norm --no-se-pre-norm-tower --checkpoint-best
done

echo "Sweep submitted. Check analysis_data/* and models/* for outputs."
