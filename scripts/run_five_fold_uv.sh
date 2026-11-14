#!/usr/bin/env bash
# 5-fold: HAGI baseline + Peak AEM UV (+0.5/+1/+2s). Run on gpuws3.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-512}"
FOLDS="${FOLDS:-0 1 2 3 4}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"

for FOLD in $FOLDS; do
  echo "========== FOLD ${FOLD} =========="

  if [[ "$SKIP_BASELINE" != "1" ]]; then
    echo "--- Preprocess HAGI baseline ---"
    "$PYTHON" scripts/prepare_hagi_regression.py \
      --fold "$FOLD" \
      --out "data/processed/hagi_regression_fold${FOLD}"

    echo "--- Train/eval HAGI baseline ---"
    "$PYTHON" scripts/train_hagi_regression.py \
      --data "data/processed/hagi_regression_fold${FOLD}" \
      --out "results/hagi_regression_fold${FOLD}" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE"

    "$PYTHON" scripts/evaluate_hagi_regression.py \
      --data "data/processed/hagi_regression_fold${FOLD}" \
      --checkpoint "results/hagi_regression_fold${FOLD}/best_model.pt" \
      --out "results/hagi_regression_fold${FOLD}" \
      --method-name "HAGI Regression Fold ${FOLD}"
  fi

  echo "--- Preprocess Peak AEM UV ---"
  "$PYTHON" scripts/prepare_hagi_audio_regression.py \
    --fold "$FOLD" \
    --out "data/processed/hagi_audio_peak_fold${FOLD}_uv" \
    --audio-mode peak_aem \
    --audio-horizons 0.5 1.0 2.0 \
    --aem-use-uv

  echo "--- Train/eval Peak AEM UV ---"
  "$PYTHON" scripts/train_hagi_audio_regression.py \
    --data "data/processed/hagi_audio_peak_fold${FOLD}_uv" \
    --out "results/hagi_audio_peak_fold${FOLD}_uv" \
    --audio-dim 6 \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE"

  "$PYTHON" scripts/evaluate_hagi_audio_regression.py \
    --data "data/processed/hagi_audio_peak_fold${FOLD}_uv" \
    --checkpoint "results/hagi_audio_peak_fold${FOLD}_uv/best_model.pt" \
    --out "results/hagi_audio_peak_fold${FOLD}_uv" \
    --audio-dim 6 \
    --method-name "HAGI + Peak AEM UV Fold ${FOLD}"
done

echo "--- Aggregate ---"
"$PYTHON" scripts/aggregate_fold_results.py \
  --audio-suffix _uv \
  --out results/five_fold_aggregate_summary.csv

echo "Done."
