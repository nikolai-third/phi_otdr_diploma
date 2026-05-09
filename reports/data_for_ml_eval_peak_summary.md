# Detector evaluation on data_for_ml

- generated: 2026-05-09 11:03:21
- tolerance: ±0.5 km around event range
- positives evaluated: **92**, negatives: **38**

## Threshold-based detection
- recall:    **0.478** (44/92)
- precision: **0.383**  (FP=71)
- F1:        **0.425**
- median localization error (hits only): **0.061 km**

## Top-usable-candidate (single, threshold-agnostic)
- recall: **0.478** (44/92)
- median localization error: **0.684 km**

## Recall by pulse duration

| pulse_ns | n | recall_thr | recall_top | median_loc_err_top_km |
|---|---:|---:|---:|---:|
| 100 | 18 | 0.278 | 0.222 | 0.990 |
| 200 | 16 | 0.812 | 0.812 | 0.061 |
| 300 | 17 | 0.706 | 0.706 | 0.102 |
| 500 | 20 | 0.500 | 0.500 | 0.398 |
| 1000 | 21 | 0.190 | 0.238 | 1.368 |

## Recall by id_line

| id_line | n | recall_thr | recall_top |
|---|---:|---:|---:|
| 1 | 24 | 0.417 | 0.417 |
| 2 | 29 | 0.586 | 0.552 |
| 3 | 9 | 0.222 | 0.222 |
| 4 | 30 | 0.500 | 0.533 |
