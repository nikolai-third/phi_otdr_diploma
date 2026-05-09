# Detector evaluation on data_for_ml

- generated: 2026-05-09 11:01:49
- tolerance: ±0.5 km around event range
- positives evaluated: **92**, negatives: **38**

## Threshold-based detection
- recall:    **0.391** (36/92)
- precision: **0.288**  (FP=89)
- F1:        **0.332**
- median localization error (hits only): **0.061 km**

## Top-usable-candidate (single, threshold-agnostic)
- recall: **0.424** (39/92)
- median localization error: **0.858 km**

## Recall by pulse duration

| pulse_ns | n | recall_thr | recall_top | median_loc_err_top_km |
|---|---:|---:|---:|---:|
| 100 | 18 | 0.111 | 0.222 | 1.021 |
| 200 | 16 | 0.688 | 0.750 | 0.071 |
| 300 | 17 | 0.706 | 0.706 | 0.061 |
| 500 | 20 | 0.350 | 0.350 | 1.072 |
| 1000 | 21 | 0.190 | 0.190 | 1.082 |

## Recall by id_line

| id_line | n | recall_thr | recall_top |
|---|---:|---:|---:|
| 1 | 24 | 0.292 | 0.333 |
| 2 | 29 | 0.448 | 0.483 |
| 3 | 9 | 0.222 | 0.222 |
| 4 | 30 | 0.467 | 0.500 |
