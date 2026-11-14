#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
FOLD=0 VARIANTS="${VARIANTS:-centroid_gated argmax_gated centroid_concat}" \
  bash scripts/run_robustness.sh
