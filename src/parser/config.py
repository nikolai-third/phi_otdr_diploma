"""Parser runtime configuration and internal tuning constants."""

from __future__ import annotations

from dataclasses import dataclass

# Internal tuning constants (not exposed via CLI).
DET_MIN_GAP_FACTOR = 0.82
DET_LOW_LEVEL_FACTOR = 0.10
DET_RISE_FACTOR = 0.075
DET_DECIMATION = 10
DET_MA_WINDOW = 31
DET_ENVELOPE_ALPHA = 0.995
DET_REFINE_RADIUS = 300
DET_FILL_MISSING = True
DET_RECOVERY_MIN_SPACING_FACTOR = 0.82
DET_RECOVERY_CORR_DECIMATION = 8
DET_PERIOD_MIN = 20_000
DET_PERIOD_MAX = 260_000
DET_TEMPLATE_REFINE_RADIUS = 450
DET_TEMPLATE_REFINE_LEN = 3000
DET_TEMPLATE_REFINE_TRACES = 48
DET_TARGET_DECIMATED_POINTS = 5_000_000
DET_MAX_ADAPTIVE_DECIMATION = 4096
DET_LOW_RUN_RATIO = 0.82
DET_SCORE_Z_THRESHOLD = 3.2
DET_EDGE_MAX_CANDIDATES = 120_000

DETECTOR_VARIANTS: tuple[tuple[float, float, float, int], ...] = (
    (DET_LOW_LEVEL_FACTOR, DET_RISE_FACTOR, DET_MIN_GAP_FACTOR, DET_DECIMATION),
    (0.09, 0.07, 0.78, 8),
    (0.10, 0.09, 0.82, 10),
    (0.12, 0.11, 0.90, 10),
    (0.10, 0.07, 1.00, 8),
)


@dataclass(frozen=True)
class ParseConfig:
    # None means: extract all reflectograms detected by parser.
    max_traces: int | None = None
    max_shift: int = 300
    cc_window_start: int = 500
    cc_window_len: int = 12_000
    raw_plot_points: int = 1_000_000
    adc_fs_hz: float = 50_000_000.0
    align_iters: int = 1
    align_decimation: int = 4
    waterfall_cmap: str = "jet"
    waterfall_exp_alpha: float = 4.0
    plot_max_traces: int = 1200
    plot_max_bins: int = 4000
    auto_select_cc_window: bool = True
    cc_scan_step: int = 2000
