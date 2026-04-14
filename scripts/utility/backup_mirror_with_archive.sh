#!/usr/bin/env bash
set -euo pipefail

# Mirror SOURCE -> DEST while archiving files that would be deleted/overwritten.
# Archived files are moved under DEST/.archive/<timestamp>/ preserving hierarchy.
#
# Usage:
#   bash scripts/utility/backup_mirror_with_archive.sh
#   bash scripts/utility/backup_mirror_with_archive.sh --dry-run
#   bash scripts/utility/backup_mirror_with_archive.sh --source /src --dest /dst

SOURCE="/home/rpotter/hypertower/"
DEST="/Muspelheim/PhD/00.3_hypertower/"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE="$2"
      shift 2
      ;;
    --dest)
      DEST="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

ts="$(date +%Y%m%d_%H%M%S)"
archive_dir="${DEST%/}/.archive/${ts}"

mkdir -p "$archive_dir"

cmd=(
  rsync -avh --progress
  --delete
  --backup
  --backup-dir="$archive_dir"
  # Directories with no backup value
  --exclude=".git/"
  --exclude=".claude/"
  --exclude=".archive/"
  # Large datasets stored elsewhere
  --exclude="refuge/"
  --exclude="Refuge/"
  --exclude="REFUGE/"
  --exclude="papila/"
  --exclude="Papila/"
  --exclude="PAPILA/"
  # Python / general caches
  --exclude="__pycache__/"
  --exclude=".mypy_cache/"
  --exclude=".ruff_cache/"
  --exclude=".pytest_cache/"
  --exclude=".cache/"
  --exclude="*.pyc"
  --exclude="*.pyo"
  # Virtual environments
  --exclude=".venv/"
  --exclude="venv/"
  --exclude="env/"
  # Node
  --exclude="node_modules/"
  # Build / dist artifacts
  --exclude="*.egg-info/"
  --exclude="dist/"
  --exclude="build/"
  --exclude=".eggs/"
  # IDE / editor metadata
  --exclude=".idea/"
  --exclude=".vscode/"
  # OS metadata
  --exclude=".DS_Store"
  --exclude="Thumbs.db"
  "${SOURCE%/}/"
  "${DEST%/}/"
)

if [[ "$DRY_RUN" -eq 1 ]]; then
  cmd+=(--dry-run)
fi

echo "Source:  ${SOURCE%/}/"
echo "Dest:    ${DEST%/}/"
echo "Archive: $archive_dir"
[[ "$DRY_RUN" -eq 1 ]] && echo "Mode:    dry-run"
echo

"${cmd[@]}"

