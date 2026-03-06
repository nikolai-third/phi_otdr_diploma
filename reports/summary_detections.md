# Summary of Detection Runs

- generated: 2026-03-06 10:53:56
- source table: `reports/tables/detection_runs_summary.csv`
- total detection runs indexed: **25**

## Key Outcomes (quick recall)

- Reference case `2024-10-11_18_59`: stable detections near **31.736 km** and **42.600 km** in tuned/final variants.
- Detection CLI now supports strict full-file mode by default (`--max-traces` not set, `--use-stable-segment` off).
- For long 10-second files, thresholded detections can be empty; top usable-zone candidate is still saved for manual validation.

## Reference Case (`reports/figures`)

- `post_align_detect_2024-10-11_18_59`: traces=999, detected=20, km=[0.000 ; 2.410 ; 5.963 ; 10.946 ; 13.683 ; 16.705 ; 19.115 ; 21.565 ; 23.812 ; 26.589 ; 29.244 ; 31.736 ; 34.268 ; 37.127 ; 42.600 ; 45.377 ; 49.135 ; 51.300 ; 54.322 ; 60.367]
- `post_align_detect_2024-10-11_18_59_final`: traces=999, detected=2, km=[31.736 ; 42.600]
- `post_align_detect_2024-10-11_18_59_readable`: traces=999, detected=2, km=[31.736 ; 42.600]
- `post_align_detect_2024-10-11_18_59_tuned`: traces=999, detected=12, km=[0.000 ; 5.963 ; 10.946 ; 21.565 ; 26.589 ; 31.736 ; 37.127 ; 42.600 ; 46.684 ; 51.300 ; 54.322 ; 60.367]
- `post_align_detect_2024-10-11_18_59_v2`: traces=999, detected=8, km=[5.963 ; 31.736 ; 34.268 ; 37.127 ; 42.600 ; 45.377 ; 49.135 ; 60.367]
- `post_align_detect_2024-11-06_14_02`: traces=999, detected=7, km=[64.574 ; 81.973 ; 85.159 ; 88.181 ; 90.877 ; 98.842 ; 101.905]
- `post_align_detect_2024-11-11_11_39`: traces=998, detected=5, km=[52.321 ; 55.507 ; 75.071 ; 95.574 ; 100.557]

## USB Batch Records (`data/processed_usb/parser_cache`)

### some_test/2024-11-06_17_19
- config: 10 s test | 2024-11-06_17_19 | Длительность импульса - 100 нс | Температура лазера - 33.18 | Температура нагреваемого волокна в начале - 24.88 | Температура нагреваемого волокна в конце - 23.57 | Температура волокна - 22.35
- latest full run `detection_v9_full`: traces=9997, duration=9.9964 s, detected=0, top_usable=42.804 km (score 3.966), usable_end_km=58.464
- max thresholded detections: `detection_v2` -> 8 (km=[2.777 ; 4.207 ; 20.667 ; 23.567 ; 26.017 ; 32.471 ; 42.600 ; 56.487])
- runs: detection_v1, detection_v2, detection_v3, detection_v4, detection_v4_relaxed, detection_v5, detection_v5_relaxed, detection_v6_sens, detection_v6_thr099, detection_v7_topusable, detection_v8_full, detection_v9_full

### some_test/2024-11-11_11_39
- config: 10 s test 2 | 2024-11-11_11_39 | Длительность импульса - 100 нс | Температура лазера - 25.80 | Температура нагреваемого волокна в начале - 18.35 | Температура нагреваемого волокна в конце - 20.02 | Температура волокна - 16.35
- max thresholded detections: `detection_v3` -> 8 (km=[87.650 ; 90.428 ; 91.612 ; 92.797 ; 93.981 ; 96.064 ; 98.760 ; 101.170])
- runs: detection_v2, detection_v3, detection_v4, detection_v5

### some_test/2024-11-11_13_06
- config: 10 s test | 2024-11-11_13_06 | Длительность импульса - 100 нс | Температура лазера - 27.67 | Температура нагреваемого волокна в начале - 21.91 | Температура нагреваемого волокна в конце - 22.63 | Температура волокна - 18.45
- latest full run `detection_v2_full`: traces=9999, duration=9.9986 s, detected=0, top_usable=17.726 km (score 1.800), usable_end_km=57.944
- max thresholded detections: `detection_v1_topusable` -> 7 (km=[3.513 ; 10.456 ; 19.278 ; 24.833 ; 30.633 ; 41.783 ; 52.402])
- runs: detection_v1_topusable, detection_v2_full

## What To Re-run First

1. `some_test/2024-11-11_11_39` with current best alignment settings (visual quality was strongest).
2. Full detection on `2024-11-06_17_19` and inspect `top_candidate_within_usable.json` (currently top around 42.8 km).
3. Full detection on `2024-11-11_13_06` and inspect top zone around 17.7 km; thresholded events are absent in full run.
