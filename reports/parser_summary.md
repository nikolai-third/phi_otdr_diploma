# Parser quality summary

- total files indexed: **882**, ok: **865**, err: **17**, ok rate: **98.1%**

| group | total | ok | err | ok% | align% | cov_med | res_before_med | res_after_med | res_after_p90 | after>5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 05_10_2024 | 56 | 56 | 0 | 100.0 | 60.7 | 1.0 | 0.081 | 0.0 | 0.37 | 1 |
| 06_10_2024 | 50 | 50 | 0 | 100.0 | 68.0 | 1.0 | 0.109 | 0.0 | 0.139 | 2 |
| 07_10_2024 | 48 | 48 | 0 | 100.0 | 52.1 | 1.0 | 0.003 | 0.0 | 0.017 | 0 |
| 08_10_2024 | 40 | 40 | 0 | 100.0 | 87.5 | 1.0 | 1.03 | 0.0 | 3.03 | 4 |
| 10_10_2024 | 40 | 40 | 0 | 100.0 | 87.5 | 1.0 | 0.402 | 0.0 | 0.916 | 3 |
| 30_09_2024 | 52 | 52 | 0 | 100.0 | 90.4 | 1.0 | 0.085 | 0.0 | 5.974 | 6 |
| some_test | 95 | 89 | 6 | 93.7 | 51.7 | 1.0 | 33.283 | 20.044 | 51.473 | 51 |
| измерение_возмущение | 161 | 157 | 4 | 97.5 | 84.7 | 1.0 | 1.361 | 0.014 | 12.183 | 20 |
| растяжение | 10 | 3 | 7 | 30.0 | 66.7 | 1.0 | 0.032 | 0.0 | 0.0 | 0 |
| data_for_ml (no parser, already-extracted) | 330 | 330 | 0 | 100.0 | 0.0 | nan | 1.008 | 1.008 | nan | 51 |

## Error type breakdown

- 16: `ValueError: No complete traces extracted; tune parser thresholds`
- 1: `ArrowInvalid: Parquet magic bytes not found in footer. Either the file is corrupted or this is not a`
