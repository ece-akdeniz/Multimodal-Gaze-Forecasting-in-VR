#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-fapra_06@gpuws3.hcics.simtech.uni-stuttgart.de}"
REMOTE_DIR="${REMOTE_DIR:-~/fapra-ss26_group6}"

echo "Pulling results/*.csv from ${REMOTE_HOST}:${REMOTE_DIR}/results/"

rsync -avz \
  --include='*/' \
  --include='*.csv' \
  --exclude='*' \
  "${REMOTE_HOST}:${REMOTE_DIR}/results/" \
  "${ROOT}/results/"

echo "Done. Compare robustness: python3 scripts/compare_robustness_results.py"
