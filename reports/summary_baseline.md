# Baseline Summary

- Files processed: **50**
- Total volume (bytes): **2561690724**
- Average file size (MB): **48.860**
- Float dtype share: **71.43%**
- Int dtype share: **0.00%**
- Error count: **15**

## Coverage

- Successful records: **35**
- Failed records: **15**
- Cache volume (approx): **see `du -sh cache`**

## Noise Observations

- Медианный std=0.0507407, 90-перцентиль std=0.0560227. Есть выраженный разброс шумовых уровней между файлами.

## Top Noisy Files

| record_id | path | format | sample_std |
|---|---|---|---:|
| 00f85adf-d9ab-5e45-a29e-3d38b15640a8 | 30_09_2024/2024-09-30_08_45.parquet | parquet | 0.125913 |
| 0b743132-90c2-5c64-8bd4-d1a1f7c9b4e1 | data_for_ml/parsed_2025-03-06_17_12.parquet | parquet | 0.0876624 |
| 03e8c623-dc38-53de-aacf-ee47d2d96378 | data_for_ml/parsed_2024-10-21_16_50.parquet | parquet | 0.0563275 |
| 075df4e0-e5d4-5912-8c60-f17aa98377a4 | data_for_ml/parsed_2024-10-21_16_56.parquet | parquet | 0.0555656 |
| 003a3cc5-99ae-55c2-a0d7-5262db41e8fa | data_for_ml/parsed_2024-10-15_14_28.parquet | parquet | 0.0554509 |
| 0414f461-b892-588f-8086-adcb981c1be4 | data_for_ml/parsed_2024-10-11_19_10.parquet | parquet | 0.0541308 |
| 063a5210-9e6e-5d96-aa11-e1170c54a407 | data_for_ml/parsed_2024-10-15_14_57.parquet | parquet | 0.0540387 |
| 00c435fe-79ae-5b41-af12-96882e35d4d0 | data_for_ml/parsed_2024-10-21_17_22.parquet | parquet | 0.0535012 |
| 0aa5520c-acc2-5d06-a934-3af9931d21e4 | измерение_возмущение/2024-10-11_19_02.parquet | parquet | 0.0530261 |
| 066ee96a-aae9-5839-a50b-56691d8dd401 | измерение_возмущение/2024-10-21_17_10.parquet | parquet | 0.0522885 |

## Generated Artifacts

- reports/figures/hist_file_sizes.png
- reports/figures/hist_file_sizes_log.png
- reports/figures/hist_std_distribution.png
- reports/figures/format_distribution.png
- reports/figures/box_std_by_format.png
- reports/figures/scatter_size_vs_std.png
- reports/tables/top20_largest_files.csv
- reports/tables/top20_noisiest_files.csv
- reports/tables/format_distribution.csv
- reports/tables/summary_by_format.csv
- reports/tables/summary_by_group.csv
- reports/tables/noise_quantiles.csv
- reports/tables/baseline_status.csv
