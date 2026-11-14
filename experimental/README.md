# Experimental audio pipelines

These scripts explore **alternative** audio representations. They are **not** the primary project method.

The main approach (supervisor-aligned) uses **peak AEM (u,v) at future +0.5s / +1s / +2s** — see [README.md](../README.md) and `scripts/prepare_hagi_audio_regression.py`.

| # | Variant | Fold 0 (all gaze) | Notes |
|---|---------|------------------:|-------|
| 1 | HAGI++ head-only | 11.50° | Experimental baseline |
| 2 | **Full AEM t+1** (128-dim) | **11.01°** | Best dense-AEM; see `results/hagi_aem_t1_fold0/` |
| 3 | Full AEM + dropout 0.3 | 11.70° | |
| 4 | AEM sequence t+1…t+10 | ~11.50° | Fixation-only; `results/hagi_aem_regression_fold0/` = 11.53° |
| 5 | AEM past + future | ~11.40° | Fixation-only |
| 6 | Ambisonic direction | ~11.60° | Needs `videos_extracted/` |

Primary pipeline peak AEM UV on fold 0: **10.36°** — compact future coordinates beat dense grids.

Summary CSV: `results/experimental_fold0_summary.csv`

All outputs use `data/processed/`.

---

## Full AEM grid (single future frame)

```bash
bash experimental/run_fold0_full_aem_ablation.sh
```

Or step by step:

```bash
python3 experimental/prepare_aem_regression.py \
  --root data/raw/D-SAV360 \
  --aem-root data/raw/D-SAV360/audio_energy_maps/dataset/AEM \
  --splits data/splits --fold 0 \
  --out data/processed/hagi_aem_regression_fold0

python3 experimental/train_aem_regression.py \
  --data data/processed/hagi_aem_regression_fold0 \
  --out results/hagi_aem_regression_fold0 --epochs 50

python3 experimental/evaluate_aem_regression.py \
  --data data/processed/hagi_aem_regression_fold0 \
  --checkpoint results/hagi_aem_regression_fold0/best_model.pt \
  --out results/hagi_aem_regression_fold0 \
  --method-name "Full AEM Grid Fold 0"
```

---

## AEM sequence (10 past frames)

```bash
python3 experimental/prepare_aem_seq_regression.py \
  --root data/raw/D-SAV360 \
  --aem-root data/raw/D-SAV360/audio_energy_maps/dataset/AEM \
  --splits data/splits --fold 0 \
  --out data/processed/hagi_aem_seq_fold0

python3 experimental/train_aem_seq_regression.py \
  --data data/processed/hagi_aem_seq_fold0 \
  --out results/hagi_aem_seq_fold0 --epochs 50 --hidden-dim 256

python3 experimental/evaluate_aem_seq_regression.py \
  --data data/processed/hagi_aem_seq_fold0 \
  --checkpoint results/hagi_aem_seq_fold0/best_model.pt \
  --out results/hagi_aem_seq_fold0 \
  --method-name "AEM sequence Fold 0"
```

---

## AEM past + future

```bash
python3 experimental/prepare_aem_past_regression.py \
  --root data/raw/D-SAV360 \
  --aem-root data/raw/D-SAV360/audio_energy_maps/dataset/AEM \
  --splits data/splits --fold 0 \
  --max-train-samples 500000 --max-test-samples 100000 \
  --out data/processed/hagi_aem_past_fold0

python3 experimental/train_aem_past_regression.py \
  --data data/processed/hagi_aem_past_fold0 \
  --out results/hagi_aem_past_fold0 --hidden-dim 256 --epochs 50

python3 experimental/evaluate_aem_past_regression.py \
  --data data/processed/hagi_aem_past_fold0 \
  --checkpoint results/hagi_aem_past_fold0/best_model.pt \
  --out results/hagi_aem_past_fold0 \
  --method-name "AEM past+future Fold 0"
```

---

## Ambisonic direction (WAV)

```bash
python3 experimental/prepare_ambisonic_direction.py \
  --root data/raw/D-SAV360 \
  --audio-root data/raw/D-SAV360/videos_extracted \
  --splits data/splits --fold 0 \
  --out data/processed/hagi_ambisonic_fold0 --tiny

python3 experimental/train_ambisonic_direction.py \
  --data data/processed/hagi_ambisonic_tiny \
  --out results/hagi_ambisonic_tiny --epochs 5
```
