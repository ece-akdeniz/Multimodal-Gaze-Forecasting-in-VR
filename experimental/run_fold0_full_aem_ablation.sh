#!/usr/bin/env bash
# Fold-0 ablation: full AEM grid (128-dim) vs peak UV in main pipeline.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
FOLD="${FOLD:-0}"
EPOCHS="${EPOCHS:-50}"

echo "=== Full AEM grid fold ${FOLD} ==="
"$PYTHON" experimental/prepare_aem_regression.py \
  --root data/raw/D-SAV360 \
  --aem-root data/raw/D-SAV360/audio_energy_maps/dataset/AEM \
  --splits data/splits \
  --fold "$FOLD" \
  --out "data/processed/hagi_aem_regression_fold${FOLD}"

"$PYTHON" experimental/train_aem_regression.py \
  --data "data/processed/hagi_aem_regression_fold${FOLD}" \
  --out "results/hagi_aem_regression_fold${FOLD}" \
  --epochs "$EPOCHS"

"$PYTHON" experimental/evaluate_aem_regression.py \
  --data "data/processed/hagi_aem_regression_fold${FOLD}" \
  --checkpoint "results/hagi_aem_regression_fold${FOLD}/best_model.pt" \
  --out "results/hagi_aem_regression_fold${FOLD}" \
  --method-name "Full AEM Grid Fold ${FOLD}"

echo "Compare with results/hagi_audio_peak_fold${FOLD}_uv/"
