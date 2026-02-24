#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_gradual_thaw_top5.sh --epochs 20 --n-splits 5 --batch-size 8 [extra args]
#
# Runs the full gradual-thaw grid aligned with the SE sweep (bridge/tower/both × R=8/16/32 × norm vs nonorm).

ARGS=("$@")

run() {
  local SHORT="$1"; shift
  echo "=== Running: $SHORT ==="
  if ls "analysis_data/${SHORT}_"*.md >/dev/null 2>&1; then
    echo "… skipping ${SHORT} (summary already present)"
    return 0
  fi
  python3 scripts/run_multifold.py \
    --shortname "$SHORT" \
    --gradual-thaw --thaw-phase-duration 5 --thaw-ratio 0.33 --thaw-start-epoch 5 \
    --early-stop --early-patience 5 \
    "$@" \
    "${ARGS[@]}" || true
}

# Bridge-only thaw runs
for R in 8 16 32; do
  for MODE in norm nonorm; do
    if [[ "$MODE" == "norm" ]]; then
      FLAGS=(--se-where bridge --se-reduction "$R" --se-pre-norm --checkpoint-best)
    else
      FLAGS=(--se-where bridge --se-reduction "$R" --no-se-pre-norm --checkpoint-best)
    fi
    run "thaw_se_bridge_R${R}_${MODE}" "${FLAGS[@]}"
    run "thawbin_se_bridge_R${R}_${MODE}" "${FLAGS[@]}" --eval_mode binary
  done
done

# Tower-only thaw runs
for R in 8 16 32; do
  for MODE in norm nonorm; do
    if [[ "$MODE" == "norm" ]]; then
      FLAGS=(--se-where tower --se-reduction-tower "$R" --se-pre-norm-tower --checkpoint-best)
    else
      FLAGS=(--se-where tower --se-reduction-tower "$R" --no-se-pre-norm-tower --checkpoint-best)
    fi
    run "thaw_se_tower_R${R}_${MODE}" "${FLAGS[@]}"
    run "thawbin_se_tower_R${R}_${MODE}" "${FLAGS[@]}" --eval_mode binary
  done
done

# Tower+bridge thaw runs
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
    run "thaw_se_tower_bridge_R${R}_${MODE}" "${FLAGS[@]}"
    run "thawbin_se_tower_bridge_R${R}_${MODE}" "${FLAGS[@]}" --eval_mode binary
  done
done

echo "Gradual thaw grid submitted. Check analysis_data/* and models/* for outputs."
