"""Core reflectogram parsing logic (detection, period inference, extraction)."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from src.parser.config import (
    DETECTOR_VARIANTS,
    DET_DECIMATION,
    DET_EDGE_MAX_CANDIDATES,
    DET_ENVELOPE_ALPHA,
    DET_FILL_MISSING,
    DET_LOW_LEVEL_FACTOR,
    DET_LOW_RUN_RATIO,
    DET_MA_WINDOW,
    DET_MAX_ADAPTIVE_DECIMATION,
    DET_MIN_GAP_FACTOR,
    DET_PERIOD_MAX,
    DET_PERIOD_MIN,
    DET_RECOVERY_CORR_DECIMATION,
    DET_RECOVERY_MIN_SPACING_FACTOR,
    DET_REFINE_RADIUS,
    DET_RISE_FACTOR,
    DET_SCORE_Z_THRESHOLD,
    DET_TARGET_DECIMATED_POINTS,
    ParseConfig,
)
from src.parser.templates import refine_starts_with_template


@dataclass(frozen=True)
class ParseResult:
    trace_len: int
    starts: np.ndarray
    traces: np.ndarray


def _adaptive_decimation(n_samples: int, base_decimation: int) -> int:
    """Increase decimation on very large files to cap detector working set."""

    base = max(1, int(base_decimation))
    if n_samples <= 0:
        return base
    adaptive = int(math.ceil(n_samples / float(DET_TARGET_DECIMATED_POINTS)))
    dec = max(base, adaptive)
    return min(dec, DET_MAX_ADAPTIVE_DECIMATION)


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.asarray(x, dtype=np.float32).copy()
    xf = np.asarray(x, dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / np.float32(window)
    return np.convolve(xf, kernel, mode="same").astype(np.float32, copy=False)


def envelopes(x: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute upper/lower envelopes and mid-envelope (EMA style)."""

    xs = np.asarray(x, dtype=np.float32)
    upper = np.empty_like(xs)
    lower = np.empty_like(xs)
    upper[0] = xs[0]
    lower[0] = xs[0]

    for i in range(1, len(xs)):
        upper[i] = xs[i] if xs[i] > upper[i - 1] else alpha * upper[i - 1] + (1.0 - alpha) * xs[i]
        lower[i] = xs[i] if xs[i] < lower[i - 1] else alpha * lower[i - 1] + (1.0 - alpha) * xs[i]

    mid = 0.5 * (upper + lower)
    return upper, lower, mid


def _local_refine_by_gradient(data: np.ndarray, center: int, radius: int) -> int:
    """Refine candidate start using local central-difference peak only."""

    n = len(data)
    if n < 5:
        return int(np.clip(center, 0, max(0, n - 1)))
    left = max(1, int(center) - int(radius))
    right = min(n - 2, int(center) + int(radius))
    if right <= left:
        return int(np.clip(center, 0, n - 1))

    seg = np.asarray(data[left - 1 : right + 2], dtype=np.float32)
    if seg.size < 3:
        return int(np.clip(center, 0, n - 1))
    grad = 0.5 * (seg[2:] - seg[:-2])
    return int(left + int(np.argmax(grad)))


def _robust_positive_z(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    scale = 1.4826 * mad + 1e-9
    return (arr - np.float32(med)) / np.float32(scale)


def _window_mean(prefix: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    num = prefix[right] - prefix[left]
    den = np.maximum(1, right - left).astype(np.float32)
    return num / den


def _nms_peaks(score: np.ndarray, min_gap: int, max_keep: int) -> np.ndarray:
    idx = np.where(score > 0)[0]
    if len(idx) == 0:
        return np.array([], dtype=np.int64)

    order = idx[np.argsort(score[idx])[::-1]]
    taken = np.zeros(len(score), dtype=bool)
    kept: list[int] = []
    gap = max(1, int(min_gap))
    for p in order:
        if taken[p]:
            continue
        kept.append(int(p))
        l = max(0, int(p) - gap)
        r = min(len(score), int(p) + gap + 1)
        taken[l:r] = True
        if len(kept) >= max_keep:
            break
    kept.sort()
    return np.asarray(kept, dtype=np.int64)


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

    dec = _adaptive_decimation(len(data), decimation)
    dec_data = data[::dec]
    sm = moving_average(dec_data, ma_window)
    _, _, mid = envelopes(sm, alpha=envelope_alpha)
    n = len(mid)
    if n < 64:
        return np.array([], dtype=np.int64)

    lo_q = float(np.quantile(mid, 0.02))
    hi_q = float(np.quantile(mid, 0.98))
    dyn = max(1e-12, hi_q - lo_q)

    low_thr = lo_q + low_level_factor * dyn
    rise_thr = max(1e-12, rise_factor * dyn)

    # Detector A: lagged derivative (accumulated derivative over lag k).
    period_min_dec = max(8, int(DET_PERIOD_MIN // max(1, dec)))
    lag = max(4, min(48, max(6, period_min_dec // 8)))
    lag = min(lag, max(4, n // 10))
    lag_diff = np.zeros(n, dtype=np.float32)
    lag_diff[lag:] = mid[lag:] - mid[:-lag]
    lag_diff = moving_average(lag_diff, max(3, lag // 3))

    # Detector B: step-response (difference of post/pre window means).
    idx = np.arange(n, dtype=np.int64)
    pre_w = max(16, min(320, 5 * lag))
    post_w = max(8, min(128, max(10, lag)))

    prefix = np.concatenate((np.array([0.0], dtype=np.float64), np.cumsum(mid, dtype=np.float64)))
    pre_l = np.maximum(0, idx - pre_w)
    pre_r = idx
    post_l = idx
    post_r = np.minimum(n, idx + post_w)
    pre_mean = _window_mean(prefix, pre_l, pre_r)
    post_mean = _window_mean(prefix, post_l, post_r)
    step_resp = (post_mean - pre_mean).astype(np.float32, copy=False)

    # Legacy-like local gradient (kept as third weak detector).
    grad = np.zeros(n, dtype=np.float32)
    grad[1:-1] = 0.5 * (mid[2:] - mid[:-2])
    grad[0] = grad[1]
    grad[-1] = grad[-2]

    # Long low-level condition before edge to suppress intra-trace spikes.
    low_mask = mid < low_thr
    low_prefix = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(low_mask.astype(np.int64))))
    low_ratio = (low_prefix[pre_r] - low_prefix[pre_l]) / np.maximum(1, pre_r - pre_l)
    low_ok = low_ratio >= float(DET_LOW_RUN_RATIO)

    z_lag = _robust_positive_z(lag_diff)
    z_step = _robust_positive_z(step_resp)
    z_grad = _robust_positive_z(grad)

    score = np.zeros(n, dtype=np.float32)
    z_thr = float(DET_SCORE_Z_THRESHOLD)
    m1 = low_ok & (lag_diff > rise_thr) & (z_lag > z_thr)
    m2 = low_ok & (step_resp > 1.1 * rise_thr) & (z_step > z_thr)
    m3 = low_ok & (grad > 0.8 * rise_thr) & (z_grad > (z_thr + 0.4))
    legacy = np.zeros(n, dtype=bool)
    legacy[1:] = (mid[:-1] < low_thr) & (grad[1:] > rise_thr)
    m4 = legacy & (z_grad > max(2.0, z_thr - 1.0))
    score[m1] = np.maximum(score[m1], z_lag[m1])
    score[m2] = np.maximum(score[m2], z_step[m2] + np.float32(0.2))
    score[m3] = np.maximum(score[m3], z_grad[m3])
    score[m4] = np.maximum(score[m4], z_grad[m4] + np.float32(0.1))

    potential = _nms_peaks(
        score=score,
        min_gap=max(2, lag // 2),
        max_keep=int(DET_EDGE_MAX_CANDIDATES),
    )
    if len(potential) < 128:
        # Relaxed fallback if strict thresholds under-detect on noisy records.
        relaxed = np.zeros(n, dtype=np.float32)
        low_ok_relaxed = low_ratio >= max(0.55, float(DET_LOW_RUN_RATIO) - 0.22)
        r1 = low_ok_relaxed & (lag_diff > 0.65 * rise_thr) & (z_lag > max(1.9, z_thr - 1.2))
        r2 = low_ok_relaxed & (step_resp > 0.8 * rise_thr) & (z_step > max(1.9, z_thr - 1.2))
        r3 = legacy & (z_grad > 1.8)
        relaxed[r1] = z_lag[r1]
        relaxed[r2] = np.maximum(relaxed[r2], z_step[r2])
        relaxed[r3] = np.maximum(relaxed[r3], z_grad[r3])
        potential = _nms_peaks(
            score=relaxed,
            min_gap=max(2, lag // 2),
            max_keep=int(DET_EDGE_MAX_CANDIDATES),
        )

    if len(potential) == 0:
        return np.array([], dtype=np.int64)

    raw_starts: list[int] = []
    radius = max(8, int(refine_radius))
    for p in potential:
        center = int(p * dec)
        idx = _local_refine_by_gradient(data, center=center, radius=radius)
        raw_starts.append(idx)

    if not raw_starts:
        return np.array([], dtype=np.int64)

    arr = np.array(sorted(raw_starts), dtype=np.int64)
    if len(arr) == 0:
        return arr
    return np.unique(arr)


def estimate_trace_len_from_candidates(
    candidates: np.ndarray,
    fallback: int = 55_000,
    period_hint: int | None = None,
) -> int:
    """Estimate reflectogram period from sparse candidate gaps.

    If ``period_hint`` is provided, gap decomposition is locked around it,
    which is robust when candidates are sparse (many missed starts).
    """

    if len(candidates) < 3:
        return fallback

    diffs = np.diff(candidates).astype(np.float64)
    diffs = diffs[(diffs > 1_000) & (diffs < 2_000_000)]
    if len(diffs) == 0:
        return fallback

    period_votes: list[float] = []
    vote_weights: list[float] = []
    hint = int(period_hint) if period_hint is not None else 0
    use_hint = hint > 0
    for gap in diffs:
        if use_hint:
            k0 = max(1, int(round(gap / float(hint))))
            ks = (k0 - 1, k0, k0 + 1)
        else:
            max_k = min(256, max(1, int(gap // 10_000)))
            ks = tuple(range(1, max_k + 1))

        for k in ks:
            if k <= 0:
                continue
            period = gap / float(k)
            if DET_PERIOD_MIN <= period <= DET_PERIOD_MAX:
                period_votes.append(period)
                # Long-gap decomposition is usually more stable.
                if use_hint:
                    vote_weights.append(np.sqrt(float(max(1, k))))
                else:
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

    dec = _adaptive_decimation(len(data), DET_DECIMATION)
    y = data[::dec]
    if len(y) < 2048:
        return 55_000

    smoothed = moving_average(y, DET_MA_WINDOW)
    centered = smoothed - np.float32(np.median(smoothed))
    n = len(centered)

    size = 1 << int(np.ceil(np.log2(2 * n - 1)))
    spec = np.fft.rfft(centered.astype(np.float32, copy=False), size)
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
    p_cand = estimate_trace_len_from_candidates(candidates, fallback=p_ac, period_hint=p_ac)
    ratio = float(p_cand) / float(max(1, p_ac))
    if 0.90 <= ratio <= 1.10:
        # With period hint, candidate decomposition is robust to missing starts.
        period = int(round(p_cand))
    else:
        period = p_ac
    return int(np.clip(period, DET_PERIOD_MIN, DET_PERIOD_MAX))


def _estimate_period_from_anchors(anchors: np.ndarray, period_hint: int) -> int:
    if len(anchors) < 2:
        return int(period_hint)

    diffs = np.diff(anchors).astype(np.float64)
    parts: list[float] = []
    for gap in diffs:
        k = max(1, int(round(float(gap) / float(max(1, period_hint)))))
        p = float(gap) / float(k)
        if DET_PERIOD_MIN <= p <= DET_PERIOD_MAX:
            parts.append(p)
    if not parts:
        return int(period_hint)

    arr = np.asarray(parts, dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    tol = 4.0 * 1.4826 * mad + 10.0
    good = np.abs(arr - med) <= tol
    if int(np.sum(good)) >= max(8, int(0.25 * len(arr))):
        arr = arr[good]
    return int(np.clip(round(float(np.median(arr))), DET_PERIOD_MIN, DET_PERIOD_MAX))


def recover_starts_periodic(data: np.ndarray, anchors: np.ndarray, trace_len: int) -> np.ndarray:
    """Recover missing starts by phase-locked periodic tracking around anchor starts."""

    n_samples = len(data)
    if len(anchors) < 2:
        return anchors

    period = _estimate_period_from_anchors(anchors=anchors, period_hint=int(trace_len))
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

    refine_radius = max(40, min(200, int(DET_REFINE_RADIUS)))
    refined: list[int] = []
    for p in predicted:
        idx = _local_refine_by_gradient(data, center=int(p), radius=refine_radius)
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


def extract_reflectograms(data: np.ndarray, starts: np.ndarray, trace_len: int, max_traces: int | None) -> np.ndarray:
    valid = starts[(starts >= 0) & (starts + trace_len <= len(data))]
    if max_traces is not None:
        valid = valid[:max_traces]
    if len(valid) == 0:
        raise ValueError("No complete traces extracted; tune parser thresholds")

    traces = np.empty((len(valid), trace_len), dtype=np.float32)
    for i, s in enumerate(valid):
        seg = np.asarray(data[int(s) : int(s) + trace_len], dtype=np.float32)
        if seg.size != trace_len:
            raise ValueError("Incomplete trace extracted; input slicing mismatch")
        traces[i, :] = seg
    return traces


def _sample_indices(n_total: int, n_keep: int) -> np.ndarray:
    if n_total <= 0:
        return np.array([], dtype=np.int64)
    if n_keep >= n_total:
        return np.arange(n_total, dtype=np.int64)
    return np.linspace(0, n_total - 1, num=n_keep, dtype=np.int64)


def _adjacent_corr_proxy(traces: np.ndarray) -> tuple[float, float]:
    """Cheap proxy of segmentation quality using row-to-row similarity.

    Returns (median_corr, p10_corr) on a few fixed windows across range bins.
    Larger values mean adjacent traces are more coherent.
    """

    n_rows, n_cols = traces.shape
    if n_rows < 4 or n_cols < 512:
        return 0.0, -1.0

    row_idx = _sample_indices(n_rows, min(1200, n_rows))
    win = int(min(12_000, max(1024, n_cols // 4)))
    if win >= n_cols:
        win = max(256, n_cols - 1)
    starts = [
        max(0, min(n_cols - win, n_cols // 6 - win // 2)),
        max(0, min(n_cols - win, n_cols // 3 - win // 2)),
        max(0, min(n_cols - win, n_cols // 2 - win // 2)),
    ]

    best_med = -1.0
    best_p10 = -1.0
    for s in starts:
        seg = np.asarray(traces[row_idx, s : s + win], dtype=np.float32)
        a = seg[:-1] - np.mean(seg[:-1], axis=1, keepdims=True)
        b = seg[1:] - np.mean(seg[1:], axis=1, keepdims=True)
        den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9
        corr = np.sum(a * b, axis=1) / den
        med = float(np.median(corr))
        p10 = float(np.quantile(corr, 0.10))
        if p10 > best_p10 or (p10 == best_p10 and med > best_med):
            best_med = med
            best_p10 = p10
    return best_med, best_p10


def extract_with_fallbacks(data: np.ndarray, trace_len: int, max_traces: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Try primary detector and nearby variants to avoid empty extraction."""

    candidates: list[tuple[int, float, float, float, np.ndarray, np.ndarray]] = []
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
        if len(starts) > 2:
            d = np.diff(starts).astype(np.float64)
            med = float(np.median(d))
            mad = float(np.median(np.abs(d - med)))
            reg_cv = (1.4826 * mad) / max(1.0, abs(med))
        else:
            reg_cv = 1e9

        corr_med, corr_p10 = _adjacent_corr_proxy(traces)
        candidates.append((n_traces, reg_cv, corr_med, corr_p10, starts, traces))
        if max_traces is not None and n_traces >= max_traces:
            break

    if not candidates:
        raise ValueError("No complete traces extracted; tune parser thresholds")

    best_n = max(c[0] for c in candidates)
    # Allow tiny coverage drop if segmentation quality is much cleaner.
    min_n = int(max(1, math.floor(0.995 * best_n)))
    pool = [c for c in candidates if c[0] >= min_n]
    if not pool:
        pool = candidates

    # Quality-first among near-best counts: robust low-tail correlation,
    # then median correlation, then start regularity, then count.
    pool.sort(key=lambda c: (-c[3], -c[2], c[1], -c[0]))
    chosen = pool[0]
    return chosen[4], chosen[5]


def parse_reflectograms(data: np.ndarray, cfg: ParseConfig) -> ParseResult:
    raw_candidates = collect_start_candidates(data)
    trace_len = infer_trace_len(data, raw_candidates)
    starts, traces = extract_with_fallbacks(data, trace_len=trace_len, max_traces=cfg.max_traces)
    return ParseResult(trace_len=trace_len, starts=starts, traces=traces)
