# Results

Git tracks summary CSVs only. Checkpoints and `.npy` files are local/GPU only.

## Primary (5-fold)

- `five_fold_aggregate_summary.csv`
- `hagi_regression_fold{0-4}/`
- `hagi_audio_peak_fold{0-4}_uv/`

## Fold 0 baselines

- `head_proxy/head_proxy_summary.csv`
- `dgaze_head_only_fold0/`
- `dgaze_full_fold0_nearest_300k/`
- `hagi_audio_peak_fold0/` (peak AEM XYZ ablation)

## Experimental (`experimental/`)

- `experimental_fold0_summary.csv`
- `hagi_aem_t1_fold0/` — dense AEM t+1 (11.01 deg)
- `hagi_aem_regression_fold0/` — AEM sequence (11.53 deg)

## Robustness variants (fold 0)

- `robustness_fold0_comparison.csv`
- `hagi_audio_peak_uv_argmax_gated_fold0/`
- `hagi_audio_peak_uv_centroid_gated_fold0/`
- `hagi_audio_peak_uv_centroid_concat_fold0/`
