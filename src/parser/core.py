"""Core reflectogram parsing logic (detection, period inference, extraction)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.parser.config import (
    DETECTOR_VARIANTS,
    DET_DECIMATION,
    DET_ENVELOPE_ALPHA,
    DET_FILL_MISSING,
    DET_LOW_LEVEL_FACTOR,
    DET_MA_WINDOW,
    DET_MIN_GAP_FACTOR,
    DET_PERIOD_MAX,
    DET_PERIOD_MIN,
    DET_RECOVERY_CORR_DECIMATION,
    DET_RECOVERY_MIN_SPACING_FACTOR,
    DET_REFINE_RADIUS,
    DET_RISE_FACTOR,
    ParseConfig,
)
from src.parser.templates import refine_starts_with_template


@dataclass(frozen=True)
class ParseResult:
    trace_len: int
    starts: np.ndarray
    traces: np.ndarray


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(x, kernel, mode="same")


def envelopes(x: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute upper/lower envelopes and mid-envelope (EMA style)."""

    upper = np.empty_like(x)
    lower = np.empty_like(x)
    upper[0] = x[0]
    lower[0] = x[0]

    for i in range(1, len(x)):
        upper[i] = x[i] if x[i] > upper[i - 1] else alpha * upper[i - 1] + (1.0 - alpha) * x[i]
        lower[i] = x[i] if x[i] < lower[i - 1] else alpha * lower[i - 1] + (1.0 - alpha) * x[i]

    mid = 0.5 * (upper + lower)
    return upper, lower, mid


def collect_start_candidates(
    data: np.ndarray,
    *,
    low_level_factor: float = DET_LOW_LEVEL_FACTOR,
    rise_factor: float = DET_RISE_FACTOR,
    decimation: int = DET_DECIMATION,
    ma_window: int = DET_MA_WINDOW,
    envelope_alpha: float = DET_ENVELOPE_ALPHA,
    refine_radius: int = DET_REFINE_RADIUS,
) -> np.ndarray:
    """Return dense start candidates before period-based spacing filter."""

    dec = max(1, int(decimation))
    dec_data = data[::dec]
    sm = moving_average(dec_data, ma_window)
    _, _, mid = envelopes(sm, alpha=envelope_alpha)

    lo_q = float(np.quantile(mid, 0.02))
    hi_q = float(np.quantile(mid, 0.98))
    dyn = max(1e-12, hi_q - lo_q)

    low_thr = lo_q + low_level_factor * dyn
    rise_thr = rise_factor * dyn

    grad = np.zeros_like(mid)
    grad[1:-1] = (mid[2:] - mid[:-2]) * 0.5
    grad[0] = grad[1]
    grad[-1] = grad[-2]

    low_regions = mid < low_thr
    sharp_rises = grad > rise_thr
    potential = np.where(low_regions[:-1] & sharp_rises[1:])[0] + 1
    if len(potential) == 0:
        return np.array([], dtype=np.int64)

    raw_starts: list[int] = []
    raw_grad = np.zeros_like(data)
    raw_grad[1:-1] = (data[2:] - data[:-2]) * 0.5
    raw_grad[0] = raw_grad[1]
    raw_grad[-1] = raw_grad[-2]

    radius = max(8, int(refine_radius))
    for p in potential:
        center = int(p * dec)
        left = max(1, center - radius)
        right = min(len(data) - 2, center + radius)
        local = raw_grad[left:right]
        if local.size == 0:
            continue
        idx = int(np.argmax(local)) + left
        raw_starts.append(idx)

    if not raw_starts:
        return np.array([], dtype=np.int64)

    return np.array(sorted(raw_starts), dtype=np.int64)


def estimate_trace_len_from_candidates(candidates: np.ndarray, fallback: int = 55_000) -> int:
    """Estimate reflectogram length from candidate start gaps without external prior."""

    if len(candidates) < 3:
        return fallback

    diffs = np.diff(candidates).astype(np.float64)
    diffs = diffs[(diffs > 1_000) & (diffs < 2_000_000)]
    if len(diffs) == 0:
        return fallback

    period_votes: list[float] = []
    vote_weights: list[float] = []
    for gap in diffs:
        max_k = min(12, max(1, int(gap // 10_000)))
        for k in range(1, max_k + 1):
            period = gap / float(k)
            if 5_000 <= period <= 300_000:
                period_votes.append(period)
                vote_weights.append(1.0 / np.sqrt(float(k)))

    if not period_votes:
        return fallback

    values = np.asarray(period_votes, dtype=np.float64)
    weights = np.asarray(vote_weights, dtype=np.float64)
    bin_size = 100.0
    bins = np.floor(values / bin_size).astype(np.int64)
    uniq, inv = np.unique(bins, return_inverse=True)
    agg = np.zeros(len(uniq), dtype=np.float64)
    np.add.at(agg, inv, weights)

    best_bin = uniq[int(np.argmax(agg))]
    center = float(best_bin) * bin_size
    near = values[np.abs(values - center) <= 2_000.0]
    est = center if len(near) == 0 else float(np.median(near))
    return int(round(est))


def estimate_trace_len_from_autocorr(data: np.ndarray) -> int:
    """Estimate reflectogram period from signal autocorrelation on decimated stream."""

    dec = max(1, int(DET_DECIMATION))
    y = data[::dec]
    if len(y) < 2048:
        return 55_000

    smoothed = moving_average(y, DET_MA_WINDOW)
    centered = smoothed - float(np.median(smoothed))
    n = len(centered)

    size = 1 << int(np.ceil(np.log2(2 * n - 1)))
    spec = np.fft.rfft(centered, size)
    ac = np.fft.irfft(spec * np.conj(spec), size)[:n]
    ac[0] = 0.0

    min_lag = max(8, int(DET_PERIOD_MIN // dec))
    max_lag = min(n - 2, int(DET_PERIOD_MAX // dec))
    if max_lag <= min_lag:
        return 55_000

    arr = ac[min_lag : max_lag + 1]
    if len(arr) < 5:
        return 55_000
    arr = moving_average(arr, 9)

    peaks = np.where((arr[1:-1] > arr[:-2]) & (arr[1:-1] >= arr[2:]))[0] + 1
    if len(peaks) == 0:
        best = int(np.argmax(arr))
    else:
        vals = arr[peaks]
        vmax = float(np.max(vals))
        good = peaks[vals >= 0.70 * vmax]
        best = int(np.min(good)) if len(good) else int(peaks[int(np.argmax(vals))])
    return int((best + min_lag) * dec)


def infer_trace_len(data: np.ndarray, candidates: np.ndarray) -> int:
    """Infer trace length by blending autocorr and candidate-gap estimates."""

    p_ac = estimate_trace_len_from_autocorr(data)
    p_cand = estimate_trace_len_from_candidates(candidates, fallback=p_ac)
    ratio = float(p_cand) / float(max(1, p_ac))
    if 0.60 <= ratio <= 1.60:
        period = int(round(0.75 * p_ac + 0.25 * p_cand))
    else:
        period = p_ac
    return int(np.clip(period, DET_PERIOD_MIN, DET_PERIOD_MAX))


def recover_starts_periodic(data: np.ndarray, anchors: np.ndarray, trace_len: int) -> np.ndarray:
    """Recover missing starts by phase-locked periodic tracking around anchor starts."""

    n_samples = len(data)
    if len(anchors) < 2:
        return anchors

    diffs = np.diff(anchors)
    period_parts: list[float] = []
    for gap in diffs:
        k = max(1, int(round(float(gap) / float(trace_len))))
        period_parts.append(float(gap) / float(k))
    if period_parts:
        p_anchor = float(np.median(period_parts))
        period = int(round(0.7 * float(trace_len) + 0.3 * p_anchor))
    else:
        period = int(trace_len)
    if period <= 0:
        return anchors

    predicted: list[int] = [int(anchors[0])]
    for i in range(1, len(anchors)):
        prev = int(anchors[i - 1])
        cur = int(anchors[i])
        gap = cur - prev
        k = max(1, int(round(float(gap) / float(period))))
        step = float(gap) / float(k)
        for j in range(1, k):
            predicted.append(int(round(prev + j * step)))
        predicted.append(cur)

    while predicted and predicted[0] - period >= 0:
        predicted.insert(0, predicted[0] - period)
    while predicted and predicted[-1] + period + trace_len <= n_samples:
        predicted.append(predicted[-1] + period)

    if not predicted:
        return anchors

    smoothed = moving_average(data, max(7, DET_MA_WINDOW // 2))
    grad = np.zeros_like(smoothed)
    grad[1:-1] = (smoothed[2:] - smoothed[:-2]) * 0.5
    grad[0] = grad[1]
    grad[-1] = grad[-2]

    refine_radius = max(40, min(200, int(DET_REFINE_RADIUS)))
    refined: list[int] = []
    for p in predicted:
        left = max(1, int(p) - refine_radius)
        right = min(n_samples - 2, int(p) + refine_radius)
        if right <= left:
            continue
        idx = int(left + int(np.argmax(grad[left : right + 1])))
        refined.append(idx)

    if not refined:
        return anchors

    recovered = np.array(sorted(set(refined)), dtype=np.int64)
    recovered = recovered[recovered + trace_len <= n_samples]
    if len(recovered) == 0:
        return anchors

    min_spacing = max(1, int(DET_RECOVERY_MIN_SPACING_FACTOR * period))
    packed: list[int] = [int(recovered[0])]
    for s in recovered[1:]:
        if int(s) - packed[-1] >= min_spacing:
            packed.append(int(s))
    recovered = np.array(packed, dtype=np.int64)

    if len(recovered) > 2:
        rd = np.diff(recovered)
        if float(np.std(rd)) > 0.25 * period:
            return anchors
    return recovered


def detect_starts(
    data: np.ndarray,
    trace_len: int,
    *,
    low_level_factor: float = DET_LOW_LEVEL_FACTOR,
    rise_factor: float = DET_RISE_FACTOR,
    min_gap_factor: float = DET_MIN_GAP_FACTOR,
    decimation: int = DET_DECIMATION,
) -> np.ndarray:
    starts = collect_start_candidates(
        data,
        low_level_factor=low_level_factor,
        rise_factor=rise_factor,
        decimation=decimation,
    )
    if len(starts) == 0:
        return starts

    min_gap = max(1, int(trace_len * min_gap_factor))
    filtered = [int(starts[0])]
    for s in starts[1:]:
        if int(s) - filtered[-1] >= min_gap:
            filtered.append(int(s))

    starts = np.array(filtered, dtype=np.int64)
    if len(starts) <= 1:
        return starts
    if DET_FILL_MISSING:
        starts = recover_starts_periodic(data=data, anchors=starts, trace_len=trace_len)
    return refine_starts_with_template(data=data, starts=starts, trace_len=trace_len)


def extract_reflectograms(data: np.ndarray, starts: np.ndarray, trace_len: int, max_traces: int) -> np.ndarray:
    starts = starts[:max_traces]
    traces: list[np.ndarray] = []
    for s in starts:
        end = int(s) + trace_len
        if end <= len(data):
            traces.append(data[int(s) : end])
    if not traces:
        raise ValueError("No complete traces extracted; tune parser thresholds")
    return np.vstack(traces)


def extract_with_fallbacks(data: np.ndarray, trace_len: int, max_traces: int) -> tuple[np.ndarray, np.ndarray]:
    """Try primary detector and nearby variants to avoid empty extraction."""

    best: tuple[int, np.ndarray, np.ndarray] | None = None
    for low, rise, gap, dec in DETECTOR_VARIANTS:
        starts = detect_starts(
            data,
            trace_len=trace_len,
            low_level_factor=low,
            rise_factor=rise,
            min_gap_factor=gap,
            decimation=dec,
        )
        if len(starts) == 0:
            continue
        try:
            traces = extract_reflectograms(data, starts, trace_len=trace_len, max_traces=max_traces)
        except Exception:
            continue
        n_traces = traces.shape[0]
        if best is None or n_traces > best[0]:
            best = (n_traces, starts, traces)
            if n_traces >= max_traces:
                break

    if best is None:
        raise ValueError("No complete traces extracted; tune parser thresholds")
    return best[1], best[2]


def parse_reflectograms(data: np.ndarray, cfg: ParseConfig) -> ParseResult:
    raw_candidates = collect_start_candidates(data)
    trace_len = infer_trace_len(data, raw_candidates)
    starts, traces = extract_with_fallbacks(data, trace_len=trace_len, max_traces=cfg.max_traces)
    return ParseResult(trace_len=trace_len, starts=starts, traces=traces)
