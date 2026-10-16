#!/bin/bash
# backup_workspace.sh — respaldo del workspace (codigo, configs y checkpoints).
# Excluye dataset (re-descargable) y caches. Uso: bash scripts/backup_workspace.sh
set -e
SRC="$HOME/projects/data-augh"
DEST="$HOME/backups/data-augh-$(date +%F_%H%M)"
mkdir -p "$DEST"
rsync -a \
  --exclude 'data/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude 'experiments/*/test_overlays*/' \
  --exclude 'runs/*_1panel/' \
  "$SRC/" "$DEST/"
echo "backup -> $DEST"
du -sh "$DEST"
