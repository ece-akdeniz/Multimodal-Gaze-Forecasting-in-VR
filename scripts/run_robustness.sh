#!/usr/bin/env bash
# Peak UV robustness variants: centroid peaks, AEM validity mask, gated/concat fusion.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
FOLD="${FOLD:-0}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-512}"
VARIANTS="${VARIANTS:-centroid_gated argmax_gated centroid_concat}"

run_variant() {
  local name="$1"
  local peak_method="$2"
  local fusion="$3"
  local processed="data/processed/hagi_audio_peak_uv_${name}_fold${FOLD}"
  local results="results/hagi_audio_peak_uv_${name}_fold${FOLD}"

  echo "========== ${name} (fold ${FOLD}) =========="

  "$PYTHON" scripts/prepare_hagi_audio_regression.py \
    --fold "$FOLD" \
    --out "$processed" \
    --audio-mode peak_aem \
    --audio-horizons 0.5 1.0 2.0 \
    --aem-use-uv \
    --peak-method "$peak_method" \
    --aem-mask-mode zero

  "$PYTHON" scripts/train_hagi_audio_regression.py \
    --data "$processed" \
    --out "$results" \
    --audio-coord-dim 6 \
    --audio-validity-dim 3 \
    --fusion "$fusion" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE"

  "$PYTHON" scripts/evaluate_hagi_audio_regression.py \
    --data "$processed" \
    --checkpoint "$results/best_model.pt" \
    --out "$results" \
    --audio-coord-dim 6 \
    --audio-validity-dim 3 \
    --fusion "$fusion" \
    --method-name "HAGI + Peak AEM UV ${name} Fold ${FOLD}"
}

for variant in $VARIANTS; do
  case "$variant" in
    centroid_gated) run_variant "centroid_gated" centroid gated ;;
    argmax_gated) run_variant "argmax_gated" argmax gated ;;
    centroid_concat) run_variant "centroid_concat" centroid concat ;;
    *)
      echo "Unknown variant: $variant" >&2
      exit 1
      ;;
  esac
done

echo "Robustness fold ${FOLD} done."
