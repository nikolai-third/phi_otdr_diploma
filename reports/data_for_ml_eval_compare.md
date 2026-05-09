# Detector ablation: combined vs single-component baselines

Evaluated on 92 labeled positive + 38 labeled negative records from
`data/raw_usb/data_for_ml/data_description.json`. Tolerance ±0.5 km around
the marked event range. All four runs use the same input traces (one
prepared `aligned.npz` per record); only the score formula and the
threshold logic differ.

## Headline numbers (threshold-based detection)

| Score | Recall | Precision | F1 | FP | Median loc err |
|---|---:|---:|---:|---:|---:|
| **combined** (proposed: 0.55 peak + 0.30 broad + 0.15 energy, peak gate at 5σ) | 0.380 | **0.636** | **0.476** | 20 | 61 m |
| combined relaxed (peak gate 3σ, combined gate 2σ) | 0.424 | 0.371 | 0.396 | 66 | 61 m |
| peak_only (only the spectral peak z-score, single gate at 3σ) | **0.478** | 0.383 | 0.425 | 71 | 61 m |
| energy_only (RMS z-score baseline, single gate at 3σ) | 0.391 | 0.288 | 0.332 | 89 | 61 m |

Top-usable-candidate (single ranked candidate per file, threshold-agnostic)
recall is 0.467 / 0.467 / 0.478 / 0.424 for combined / relaxed / peak / energy.

## Reading the ablation

- **combined wins on F1 and precision.** The dual gate (combined ≥ 3σ AND peak ≥ 5σ)
  filters out 51 of the 71 peak-only false positives without giving up much recall:
  it only loses 9 hits relative to peak_only.
- **energy_only is the weakest** detector across the board (F1=0.33). RMS
  alone cannot distinguish a localised disturbance from a noisy stretch of
  fibre — a third more false positives than peak_only with no recall gain.
- **peak_only catches the highest recall** but pays in precision: half of
  its detections on the labeled set are wrong locations. This is the kind
  of result one would expect from a literature method that flags every
  per-frequency outlier without enforcing a stable spatial peak.
- **Localisation, when right, is essentially the same** across all four
  methods (61 m median): the score formula picks *whether* the right zone
  reaches the threshold, not *where* the peak ends up.

## Recall by pulse duration (top-usable)

| pulse | n | combined | peak_only | energy_only |
|---|---:|---:|---:|---:|
| 100 ns | 18 | 0.278 | 0.222 | 0.222 |
| 200 ns | 16 | 0.688 | **0.812** | 0.750 |
| 300 ns | 17 | 0.706 | 0.706 | 0.706 |
| 500 ns | 20 | 0.500 | 0.500 | 0.350 |
| 1000 ns | 21 | 0.238 | 0.238 | 0.190 |

Pattern is robust to score formula:
- **mid-range pulses (200–300 ns)** are easy for every method.
- **100 ns** is recall-limited by SNR (small impulse energy).
- **1000 ns** has the worst recall — the long pulse smears the disturbance
  across ~100 m of fibre, so its peak amplitude per distance bin drops below
  the threshold.

## Why combined is the recommended choice

Two empirical reasons:

1. **Best F1 by 5 absolute points** over the next-strongest baseline (peak_only),
   which translates to ~2× fewer false positives per detected disturbance.
2. **The peak gate suppresses background-noise lookalikes.** Negatives
   (fibre at rest, no activity) regularly produce energy-z scores in the
   2.5–5 range; they fail the spectral peak test (which requires a tight
   frequency-localised excess, not just elevated RMS).

## Limits visible in the data

- Recall plateaus at ~0.47 (top-usable) regardless of score formula. For
  ~half of the labeled files the most spectrally anomalous distance bin
  is *not* in the marked event zone — these are the misses, and they
  cannot be recovered by tuning thresholds. Likely cause is signal-to-noise:
  short 1-second windows give Δf ≈ 1 Hz, so the 0.57 Hz step rate is
  sub-resolution and the detector relies entirely on broadband impacts.
- All 38 labeled negatives come from a single line (id_line=3) and one
  pulse (500 ns), so the precision number above is an upper bound — a
  more representative negative pool would likely lower it.
