#!/usr/bin/env bash
# Transfer AEM data to gpuws3.
#
# Option A (from Mac): rsync extracted AEMs (~1.1 GB)
#   bash scripts/sync_aem_to_gpu.sh
#
# Option B (on GPU, often faster): download zip from Zenodo (~231 MB)
#   bash scripts/sync_aem_to_gpu.sh --download-on-gpu

set -euo pipefail

LOCAL="/Users/Student/Desktop/COURSES/UNI STUTTGART/SoSe 26/CAI/fapra-ss26_group6"
REMOTE_HOST="fapra_06@gpuws3.hcics.simtech.uni-stuttgart.de"
REMOTE_DIR="~/fapra-ss26_group6"
AEM_LOCAL="$LOCAL/data/raw/D-SAV360/audio_energy_maps"
ZENODO_URL="https://zenodo.org/records/10043919/files/AEM.zip/content"

if [[ "${1:-}" == "--download-on-gpu" ]]; then
  echo "=== Download AEM on gpuws3 (Zenodo) ==="
  ssh "$REMOTE_HOST" bash -s <<REMOTE
set -euo pipefail
cd $REMOTE_DIR/data/raw/D-SAV360
mkdir -p audio_energy_maps
if [[ -d audio_energy_maps/dataset/AEM/0001 ]]; then
  echo "AEM already present at audio_energy_maps/dataset/AEM/"
  ls audio_energy_maps/dataset/AEM | head -5
  exit 0
fi
echo "Downloading AEM.zip from Zenodo..."
curl -L -o AEM.zip "$ZENODO_URL"
unzip -q AEM.zip -d audio_energy_maps
rm AEM.zip
echo "Done. Sample videos:"
ls audio_energy_maps/dataset/AEM | head -5
REMOTE
  echo "=== Verify on GPU ==="
  ssh "$REMOTE_HOST" "ls $REMOTE_DIR/data/raw/D-SAV360/audio_energy_maps/dataset/AEM | head -5"
  exit 0
fi

if [[ ! -d "$AEM_LOCAL/dataset/AEM" ]]; then
  echo "Local AEM not found at $AEM_LOCAL/dataset/AEM"
  echo "Download locally first, or use: bash scripts/sync_aem_to_gpu.sh --download-on-gpu"
  exit 1
fi

echo "=== Rsync AEM from Mac to gpuws3 (~1.1 GB, may take several minutes) ==="
ssh "$REMOTE_HOST" "mkdir -p $REMOTE_DIR/data/raw/D-SAV360/audio_energy_maps"
rsync -avz --progress \
  "$AEM_LOCAL/" \
  "${REMOTE_HOST}:${REMOTE_DIR}/data/raw/D-SAV360/audio_energy_maps/"

echo "=== Verify on GPU ==="
ssh "$REMOTE_HOST" "ls $REMOTE_DIR/data/raw/D-SAV360/audio_energy_maps/dataset/AEM | head -5"
