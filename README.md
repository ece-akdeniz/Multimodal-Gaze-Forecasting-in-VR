# Multimodal Gaze Forecasting in VR

Gaze-direction prediction for 360° virtual reality from **head motion** and **spatial audio**, without eye-tracking hardware. Motivated by **foveated rendering**: inferring where the user will look from signals already available in the headset (head pose, ambisonic/AEM data).

**Dataset:** [D-SAV360](https://graphics.unizar.es/projects/D-SAV360/) · **Splits:** [AViSal360 5-fold](https://github.com/edurnebernal/AViSal360/tree/main/k_folds) · **Metric:** mean angular error (°, lower is better)

---

## Scope

Predict the **next gaze direction** (3D unit vector) from a short input window. Training uses all fixations and saccades; evaluation reports **all gaze**, **fixations only**, and **saccades**. The main contribution is a **compact spatial-audio extension** to a re-implemented head baseline: future peak coordinates from audio energy maps (AEM) at +0.5 s, +1 s, and +2 s.

---

## Methods compared

| Method | Inputs | Model | What it tests |
|--------|--------|-------|----------------|
| **Head Proxy** | Current head orientation | None (gaze := head) | Lower bound; no learning, no temporal context |
| **DGaze Head-only** | Past head velocities / positions | 1D-CNN encoder | Learned head-only forecasting (original DGaze architecture, no saliency) |
| **DGaze Full** | Head sequence + **visual saliency** maps | CNN (sequence + saliency encoders) | Whether **visual** attention maps help gaze prediction |
| **HAGI++ regression** *(re-implemented baseline)* | Past **10** head orientations (3D) | GRU → regression head | Head-only sequence model; inspired by head-assisted gaze work [4,5], simplified to GRU regression (not the full diffusion model) |
| **HAGI++ + audio RMS** | Head window + **4-dim** channel RMS from WAV | Same GRU + audio MLP | Scalar **current-time** audio (no spatial/future structure) |
| **HAGI++ + peak AEM XYZ** | Head window + **future** peak AEM as 3D unit vectors (9-dim) | Same fusion | Spatial audio at +0.5/+1/+2 s, XYZ encoding |
| **HAGI++ + peak AEM UV** *(primary)* | Head window + **future** peak AEM as UV (6-dim) | Same fusion | **Our method** — argmax peak; skip samples with missing AEM |
| **HAGI++ + peak UV, argmax, gated** | Peak UV (6-dim) + AEM validity (3-dim) | GRU + **gated** audio MLP | Argmax peak; keep AEM-missing samples; learned gate on validity |
| **HAGI++ + peak UV, centroid, gated** | Same layout (6+3-dim) | GRU + gated audio MLP | Energy-weighted **centroid** peak instead of argmax |
| **HAGI++ + peak UV, centroid, concat** | Same layout (6+3-dim) | GRU + concat audio MLP | Centroid peak; validity in MLP (no gating) |
| **Full AEM grid** (`experimental/`) | Head window + resized **128-dim** AEM frame(s) | Separate HAGI++ variants | Dense map vs peak coordinates (fold-0 ablation only) |

**Key differences in one line:** Head Proxy and DGaze Head-only use little or no multimodal context; DGaze Full adds **vision (saliency)**; our line adds **audio (AEM)**. Among audio variants, RMS lacks spatial/future structure; full grids are high-dimensional and sparse; **peak UV at future times** is the most compact predictive spatial cue. Robustness variants test centroid peaks, AEM validity masks, and gated fusion vs the primary **drop-missing** policy.

---

## Results

### Fold 0 (all gaze) — primary pipeline (`scripts/`)

| Method | Error (°) | vs HAGI++ baseline |
|--------|----------:|-------------------:|
| Head Proxy | 12.90 | +1.47 |
| DGaze Head-only | 19.41 | +8.16 |
| DGaze Full + saliency (nearest, 300k) | 21.86 | +10.61 |
| **HAGI++ regression** | **11.25** | — |
| HAGI++ + audio RMS | 11.57 | +0.32 |
| HAGI++ + peak AEM XYZ | 10.57 | −0.68 |
| **HAGI++ + peak AEM UV** *(primary)* | **10.36** | **−0.89** |

HAGI++ head-only clearly beats Head Proxy and DGaze on fold 0. **Visual saliency (DGaze Full) does not help** under our setup. **Future peak AEM (UV)** gives the largest gain; scalar RMS does not.

*Note: DGaze runs use the original CNN pipeline and may differ in sample caps; Head Proxy is evaluated on the full synchronized corpus.*

### Fold 0 — peak UV robustness variants (`scripts/`, all gaze)

Same future horizons (+0.5 / +1 / +2 s); **keep** samples with missing AEM (zero coords + validity mask) instead of dropping them.

| Method | Error (°) | vs primary peak UV |
|--------|----------:|-------------------:|
| **HAGI++ + peak AEM UV** *(primary)* | **10.36** | — |
| + peak UV, argmax, gated | 11.34 | +0.98 |
| + peak UV, centroid, gated | 11.43 | +1.07 |
| + peak UV, centroid, concat | 11.43 | +1.08 |

Argmax peak beats centroid (+0.09°); gating vs concat makes no practical difference. All three variants are **worse than primary** — dropping unreliable AEM frames works better than masking/gating. Fold 0 only; not run at 5-fold scale.

**Files:** `results/hagi_audio_peak_uv_argmax_gated_fold0/` · `results/hagi_audio_peak_uv_centroid_gated_fold0/` · `results/hagi_audio_peak_uv_centroid_concat_fold0/` · `results/robustness_fold0_comparison.csv`

### Fold 0 — experimental dense-AEM ablations (`experimental/`)

| # | Method | Error (°) | Protocol notes |
|---|--------|----------:|----------------|
| 1 | HAGI++ head-only | 11.50 | All gaze |
| 2 | **Full AEM t+1** (128-dim) | **11.01** | All gaze; best dense-AEM result |
| 3 | Full AEM t+1 + dropout 0.3 | 11.70 | All gaze |
| 4 | AEM sequence (t+1…t+10) | ~11.50 | Fixation-only train; 30-frame head |
| 5 | AEM past + future | ~11.40 | Fixation-only train; 30-frame head |
| 6 | Ambisonic direction | ~11.60 | Fixation-only train; raw WAV |

Dense AEM improves over the experimental head-only baseline (11.50° → 11.01°) but **does not beat peak UV (10.36°)**. Methods 4–6 use a different training/eval protocol; treat them as fold-0 explorations, not as the main 5-fold claim.

**Files:** `results/experimental_fold0_summary.csv` · `results/hagi_aem_t1_fold0/` · `results/robustness_fold0_comparison.csv`

### Five-fold cross-validation — primary models

| | All gaze | Fixations | Saccades |
|---|--:|--:|--:|
| HAGI++ (head only) | 10.69 ± 1.31 | 10.48 ± 1.35 | 11.72 ± 1.11 |
| **HAGI++ + peak AEM UV** | **10.41 ± 1.03** | **10.18 ± 1.06** | **11.47 ± 0.86** |

Mean gain on all gaze: **~0.28°**. Audio helps both fixations and saccades on average; folds 3–4 are mixed (weak AEM on some test videos).

**Files:** `results/five_fold_aggregate_summary.csv`

---

## Reproduce

```bash
pip install -r requirements.txt

# Data (not in git): data/raw/D-SAV360/{gaze,head,audio_energy_maps/}, data/splits/
bash scripts/run_five_fold_uv.sh          # primary 5-fold experiment
bash scripts/run_robustness_fold0.sh      # peak UV robustness variants (fold 0)
bash experimental/run_fold0_full_aem_ablation.sh   # optional dense-AEM ablation
```

**Layout:** `scripts/` (primary) · `experimental/` (dense AEM) · `results/`

### GPU cluster (gpuws3)

```bash
ssh fapra_06@gpuws3.hcics.simtech.uni-stuttgart.de
cd ~/fapra-ss26_group6 && source .venv/bin/activate
EPOCHS=50 BATCH_SIZE=512 bash scripts/run_five_fold_uv.sh
bash scripts/pull_results_from_gpu.sh     # pull summary CSVs to laptop
```

Sync code or data from laptop: `bash scripts/sync_code_to_gpu.sh` · `bash scripts/sync_aem_to_gpu.sh`

---

## References

1. E. Bernal-Berdun et al., "D-SAV360: A Dataset of Gaze Scanpaths on 360° Ambisonic Videos," *IEEE TVCG*, 2023. [doi:10.1109/TVCG.2023.3320237](https://doi.org/10.1109/TVCG.2023.3320237)

2. E. Bernal-Berdun et al., "AViSal360: Audiovisual Saliency Prediction for 360° Video," *IEEE ISMAR*, 2024. [doi:10.1109/ISMAR62088.2024.00141](https://doi.org/10.1109/ISMAR62088.2024.00141)

3. Z. Hu et al., "DGaze: CNN-Based Gaze Prediction in Dynamic Scenes," *IEEE TVCG*, vol. 26, no. 5, pp. 1902–1911, 2020. [doi:10.1109/TVCG.2020.2973473](https://doi.org/10.1109/TVCG.2020.2973473)

4. C. Jiao, Z. Hu, and A. Bulling, "HAGI: Head-Assisted Gaze Imputation for Mobile Eye Trackers," *Proc. ACM UIST*, 2025. [doi:10.1145/3746059.3747749](https://doi.org/10.1145/3746059.3747749)

5. C. Jiao, Z. Hu, and A. Bulling, "HAGI++: Head-Assisted Gaze Imputation and Generation," *arXiv:2511.02468*, 2025. [https://arxiv.org/abs/2511.02468](https://arxiv.org/abs/2511.02468)

6. D-SAV360 Audio Energy Maps (AEM), Zenodo, 2023. [https://zenodo.org/records/10043919](https://zenodo.org/records/10043919)
