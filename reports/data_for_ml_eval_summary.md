# Detector evaluation on data_for_ml

- generated: 2026-05-09 09:26:46
- tolerance: ±0.5 km around event range
- positives evaluated: **92**, negatives: **38**

## Threshold-based detection
- recall:    **0.380** (35/92)
- precision: **0.636**  (FP=20)
- F1:        **0.476**
- median localization error (hits only): **0.061 km**

## Top-usable-candidate (single, threshold-agnostic)
- recall: **0.467** (43/92)
- median localization error: **0.684 km**

## Recall by pulse duration

| pulse_ns | n | recall_thr | recall_top | median_loc_err_top_km |
|---|---:|---:|---:|---:|
| 100 | 18 | 0.111 | 0.278 | 0.939 |
| 200 | 16 | 0.500 | 0.688 | 0.061 |
| 300 | 17 | 0.588 | 0.706 | 0.061 |
| 500 | 20 | 0.550 | 0.500 | 0.398 |
| 1000 | 21 | 0.190 | 0.238 | 1.246 |

## Recall by id_line

| id_line | n | recall_thr | recall_top |
|---|---:|---:|---:|
| 1 | 24 | 0.292 | 0.375 |
| 2 | 29 | 0.448 | 0.517 |
| 3 | 9 | 0.333 | 0.333 |
| 4 | 30 | 0.400 | 0.533 |
