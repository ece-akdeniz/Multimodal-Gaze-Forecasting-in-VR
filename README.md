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

The reported quantity is mean angular error, in degrees, between the predicted gaze direction and the measured direction. Lower values indicate a more accurate forecast. The principal comparison is whether a compact encoding of the future audio-energy peak improves a head-only sequence model. Unless noted, models are trained and evaluated on all gaze samples.

### Fold 0

Fold 0 was used to compare input representations. Head orientation already accounts for most of the predictable signal. A head-only GRU regressor attains 11.25°, against 12.90° for a non-learned head proxy and 19.41° for the head-only DGaze network. Providing DGaze with visual saliency maps increases the error to 21.86° in the convolutional pipeline used here. Those two reference systems are not strictly matched to the HAGI++ runs: DGaze follows its original sample cap, and the head proxy is computed on the full synchronized corpus.

Audio reduces error only when it carries future spatial structure. Channel-wise RMS at the current frame yields 11.57°. Representing the audio-energy peak at +0.5 s, +1 s, and +2 s as a 3D unit vector lowers the error to 10.57°. The same peak in UV coordinates, the primary model, reaches 10.36°, which is 0.89° below the head-only regressor.

| Method | Angular error (°) | Difference from head-only HAGI++ (°) |
|--------|------------------:|-------------------------------------:|
| Head proxy | 12.90 | +1.65 |
| DGaze, head only | 19.41 | +8.16 |
| DGaze, head and visual saliency | 21.86 | +10.61 |
| HAGI++, head only | 11.25 | — |
| HAGI++ and channel RMS | 11.57 | +0.32 |
| HAGI++ and peak AEM, XYZ | 10.57 | −0.68 |
| HAGI++ and peak AEM, UV | 10.36 | −0.89 |

### Five-fold cross-validation

The head-only regressor and the primary peak-UV model were then trained on each of the five AViSal360 folds. The table gives the mean and standard deviation across folds.

| Model | All gaze | Fixations | Saccades |
|--------|----------:|----------:|---------:|
| HAGI++, head only | 10.69 ± 1.31 | 10.48 ± 1.35 | 11.72 ± 1.11 |
| HAGI++ and peak AEM, UV | 10.41 ± 1.03 | 10.18 ± 1.06 | 11.47 ± 0.86 |

The mean reduction on all gaze is 0.28°, and the same direction of change appears for fixations and for saccades. The standard deviation across folds is about one degree, and the audio gain is not present on every fold. The five-fold estimate is therefore a small average improvement, concentrated on videos for which the audio-energy maps are informative.

### Further checks on fold 0

Two additional checks, restricted to fold 0, bear on the choice of encoding.

The primary model omits samples for which an audio-energy peak is unavailable. Retaining those samples and marking them with a validity mask raises the error to between 11.34° and 11.43°, whether the peak is the spatial argmax or an energy-weighted centroid, and whether the audio features are gated or concatenated. Under this protocol, discarding unreliable frames is preferable to imputing them.

A dense 128-dimensional map of the next audio-energy frame reduces error from 11.50° to 11.01° relative to a matched head-only run, which remains above the peak-UV result of 10.36°. Longer map sequences and raw ambisonic direction, explored with fixation-only training, fall in the range 11.4–11.6° and are not part of the five-fold comparison.

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
