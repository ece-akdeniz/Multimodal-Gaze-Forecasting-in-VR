#!/usr/bin/env bash
# Sync project code to gpuws3 (no data, no results). Run from Mac with VPN on.
set -euo pipefail

LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-fapra_06@gpuws3.hcics.simtech.uni-stuttgart.de}"
REMOTE_DIR="${REMOTE_DIR:-~/fapra-ss26_group6}"

echo "Syncing code to ${REMOTE_HOST}:${REMOTE_DIR}/"

rsync -avz --progress \
  --exclude '.git/' \
  --exclude 'data/' \
  --exclude '.venv/' \
  --exclude 'results/' \
  --exclude '__pycache__/' \
  --exclude '.DS_Store' \
  "$LOCAL/" "${REMOTE_HOST}:${REMOTE_DIR}/"

echo "Done. SSH: ssh ${REMOTE_HOST}"
echo "Then: cd ${REMOTE_DIR} && source .venv/bin/activate"
