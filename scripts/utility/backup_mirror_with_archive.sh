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
  --exclude=".archive/"
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

