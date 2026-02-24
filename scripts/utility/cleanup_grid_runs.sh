#!/usr/bin/env bash
# Remove grid-search artifacts (analysis_data/grid_search and models/grid_search).
# Default is a dry run; pass --yes to actually delete.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANALYSIS_DIR="$ROOT_DIR/analysis_data/grid_search"
MODELS_DIR="$ROOT_DIR/models/grid_search"

DRY_RUN=1
for arg in "$@"; do
  if [[ "$arg" == "--yes" ]]; then
    DRY_RUN=0
  fi
done

echo "[cleanup] Target analysis dir: $ANALYSIS_DIR"
echo "[cleanup] Target models dir:   $MODELS_DIR"

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[cleanup] Dry run only. Nothing deleted. Pass --yes to remove."
  exit 0
fi

if [[ -d "$ANALYSIS_DIR" ]]; then
  echo "[cleanup] Removing $ANALYSIS_DIR"
  rm -rf "$ANALYSIS_DIR"
else
  echo "[cleanup] Analysis dir not found; skipping."
fi

if [[ -d "$MODELS_DIR" ]]; then
  echo "[cleanup] Removing $MODELS_DIR"
  rm -rf "$MODELS_DIR"
else
  echo "[cleanup] Models dir not found; skipping."
fi

echo "[cleanup] Done."
