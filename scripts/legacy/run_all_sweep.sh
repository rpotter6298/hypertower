#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_all_sweep.sh --epochs 25 --n-splits 5 --batch-size 8 [extra args]
#
# Merged sweep: runs the SE attention grid (bridge/tower/both × R=8/16/32),
# skipping tower-only non-normalized variants (tower normalization is a no-op),
# and includes binary eval counterparts for each baseline run. It also submits
# the full gradual-thaw grid (multiclass + binary variants).

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

echo "--- SE Grid (bridge/tower/both × R=8/16/32; tower nonorm skipped) ---"
for R in 8 16 32; do
  # Bridge-only
  run "se_bridge_R${R}_norm"       --se-where bridge --se-reduction ${R} --se-pre-norm       --checkpoint-best
  run "se_bridge_R${R}_norm_bin"   --se-where bridge --se-reduction ${R} --se-pre-norm       --checkpoint-best --eval_mode binary
  run "se_bridge_R${R}_nonorm"     --se-where bridge --se-reduction ${R} --no-se-pre-norm    --checkpoint-best
  run "se_bridge_R${R}_nonorm_bin" --se-where bridge --se-reduction ${R} --no-se-pre-norm    --checkpoint-best --eval_mode binary

  # Tower-only
  run "se_tower_R${R}_norm"     --se-where tower --se-reduction-tower ${R} --se-pre-norm-tower     --checkpoint-best
  run "se_tower_R${R}_norm_bin" --se-where tower --se-reduction-tower ${R} --se-pre-norm-tower     --checkpoint-best --eval_mode binary

  # Tower+Bridge
  run "se_tower_bridge_R${R}_norm" \
      --se-where both --se-reduction ${R} --se-reduction-tower ${R} \
      --se-pre-norm --se-pre-norm-tower --checkpoint-best
  run "se_tower_bridge_R${R}_norm_bin" \
      --se-where both --se-reduction ${R} --se-reduction-tower ${R} \
      --se-pre-norm --se-pre-norm-tower --checkpoint-best --eval_mode binary
  run "se_tower_bridge_R${R}_nonorm" \
      --se-where both --se-reduction ${R} --se-reduction-tower ${R} \
      --no-se-pre-norm --no-se-pre-norm-tower --checkpoint-best
  run "se_tower_bridge_R${R}_nonorm_bin" \
      --se-where both --se-reduction ${R} --se-reduction-tower ${R} \
      --no-se-pre-norm --no-se-pre-norm-tower --checkpoint-best --eval_mode binary
done

THAW_COMMON_ARGS=(
  --gradual-thaw
  --thaw-phase-duration 5
  --thaw-ratio 0.33
  --thaw-start-epoch 5
  --early-stop
  --early-patience 5
)

echo "--- Gradual Thaw Grid (multiclass + binary) ---"

# Bridge-only thaw runs (norm and nonorm)
for R in 8 16 32; do
  for MODE in norm nonorm; do
    if [[ "$MODE" == "norm" ]]; then
      FLAGS=(--se-where bridge --se-reduction "$R" --se-pre-norm --checkpoint-best)
    else
      FLAGS=(--se-where bridge --se-reduction "$R" --no-se-pre-norm --checkpoint-best)
    fi
    run "thaw_se_bridge_R${R}_${MODE}" "${FLAGS[@]}" "${THAW_COMMON_ARGS[@]}"
    run "thawbin_se_bridge_R${R}_${MODE}" "${FLAGS[@]}" "${THAW_COMMON_ARGS[@]}" --eval_mode binary
  done
done

# Tower-only thaw runs (norm and nonorm)
for R in 8 16 32; do
  for MODE in norm nonorm; do
    if [[ "$MODE" == "norm" ]]; then
      FLAGS=(--se-where tower --se-reduction-tower "$R" --se-pre-norm-tower --checkpoint-best)
    else
      FLAGS=(--se-where tower --se-reduction-tower "$R" --no-se-pre-norm-tower --checkpoint-best)
    fi
    run "thaw_se_tower_R${R}_${MODE}" "${FLAGS[@]}" "${THAW_COMMON_ARGS[@]}"
    run "thawbin_se_tower_R${R}_${MODE}" "${FLAGS[@]}" "${THAW_COMMON_ARGS[@]}" --eval_mode binary
  done
done

# Tower+bridge thaw runs (norm and nonorm)
for R in 8 16 32; do
  for MODE in norm nonorm; do
    if [[ "$MODE" == "norm" ]]; then
      FLAGS=(
        --se-where both
        --se-reduction "$R"
        --se-reduction-tower "$R"
        --se-pre-norm
        --se-pre-norm-tower
        --checkpoint-best
      )
    else
      FLAGS=(
        --se-where both
        --se-reduction "$R"
        --se-reduction-tower "$R"
        --no-se-pre-norm
        --no-se-pre-norm-tower
        --checkpoint-best
      )
    fi
    run "thaw_se_tower_bridge_R${R}_${MODE}" "${FLAGS[@]}" "${THAW_COMMON_ARGS[@]}"
    run "thawbin_se_tower_bridge_R${R}_${MODE}" "${FLAGS[@]}" "${THAW_COMMON_ARGS[@]}" --eval_mode binary
  done
done

echo "Merged sweep submitted. Check analysis_data/* and models/* for outputs."
