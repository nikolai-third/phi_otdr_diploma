"""Template-based refinement and cross-correlation alignment."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.parser.config import (
    DET_MA_WINDOW,
    DET_RECOVERY_CORR_DECIMATION,
    DET_RECOVERY_MIN_SPACING_FACTOR,
    DET_TEMPLATE_REFINE_LEN,
    DET_TEMPLATE_REFINE_RADIUS,
    DET_TEMPLATE_REFINE_TRACES,
    ParseConfig,
)


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.asarray(x, dtype=np.float32).copy()
    xf = np.asarray(x, dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / np.float32(window)
    return np.convolve(xf, kernel, mode="same").astype(np.float32, copy=False)


def _local_gradient_segment(data: np.ndarray, start: int, window: int, ds: int) -> np.ndarray:
    """Return decimated gradient segment around start without full-signal gradients."""

    n = len(data)
    if n < 5:
        return np.array([], dtype=np.float32)

    s = int(np.clip(start, 1, max(1, n - 3)))
    w = max(8, int(window))
    left = max(0, s - 1)
    right = min(n, s + w + 1)
    seg = np.asarray(data[left:right], dtype=np.float32)
    if seg.size < w + 2:
        pad = np.empty(w + 2, dtype=np.float32)
        pad[: seg.size] = seg
        pad[seg.size :] = seg[-1] if seg.size else np.float32(0.0)
        seg = pad

    # Local smoothing keeps gradient robust while staying memory-bounded.
    seg_s = _moving_average(seg, max(3, DET_MA_WINDOW // 6))
    grad = 0.5 * (seg_s[2:] - seg_s[:-2])
    grad_ds = grad[:: max(1, int(ds))]
    return grad_ds.astype(np.float32, copy=False)


def refine_starts_with_template(data: np.ndarray, starts: np.ndarray, trace_len: int) -> np.ndarray:
    """Refine starts by local template matching on gradient signal."""

    if len(starts) < 3:
        return starts

    n = len(data)
    ds = max(1, int(DET_RECOVERY_CORR_DECIMATION))
    window = max(256, min(int(DET_TEMPLATE_REFINE_LEN), max(512, trace_len // 3)))
    radius = max(80, int(DET_TEMPLATE_REFINE_RADIUS))

    valid = starts[(starts >= 1) & (starts + window + 2 < n)]
    if len(valid) < 3:
        return starts
    max_traces = min(len(valid), int(DET_TEMPLATE_REFINE_TRACES))
    pick = np.linspace(0, len(valid) - 1, num=max_traces, dtype=int)
    segments = np.vstack([_local_gradient_segment(data, int(valid[i]), window=window, ds=ds) for i in pick])

    template = np.median(segments, axis=0)
    template = template - float(np.mean(template))
    template = template / (float(np.linalg.norm(template)) + 1e-12)

    refined: list[int] = []
    for start in starts:
        s = int(start)
        left = max(1, s - radius)
        right = min(n - window - 2, s + radius)
        if right <= left:
            refined.append(s)
            continue
        search_window = (right - left) + window + 1
        region = _local_gradient_segment(data, left, window=search_window, ds=ds)
        if region.size < template.size:
            refined.append(s)
            continue
        corr = np.correlate(region, template, mode="valid")
        if len(corr) == 0:
            refined.append(s)
            continue
        best = int(np.argmax(corr))
        refined.append(int(left + best * ds))

    arr = np.array(sorted(refined), dtype=np.int64)
    min_spacing = max(1, int(DET_RECOVERY_MIN_SPACING_FACTOR * trace_len))
    packed: list[int] = [int(arr[0])]
    for s in arr[1:]:
        if int(s) - packed[-1] >= min_spacing:
            packed.append(int(s))
    out = np.array(packed, dtype=np.int64)
    return out[out + trace_len <= n]


def best_shift_fft(ref: np.ndarray, cur: np.ndarray, max_shift: int) -> int:
    """Estimate lag between `ref` and `cur` in bounded lag interval."""

    ref0 = ref - np.mean(ref)
    cur0 = cur - np.mean(cur)
    n = len(ref0)
    if n == 0:
        return 0
    size = 1 << int(np.ceil(np.log2(2 * n - 1)))
    ref_fft = np.fft.rfft(ref0, size)
    cur_fft = np.fft.rfft(cur0, size)
    corr = np.fft.irfft(ref_fft * np.conj(cur_fft), size)
    corr = np.concatenate((corr[-(n - 1) :], corr[:n]))
    lags = np.arange(-(n - 1), n)
    mask = (lags >= -max_shift) & (lags <= max_shift)
    if not np.any(mask):
        return 0
    idx = int(np.argmax(corr[mask]))
    return int(lags[mask][idx])


def _sample_indices(n_total: int, n_keep: int) -> np.ndarray:
    if n_total <= 0:
        return np.array([], dtype=np.int64)
    if n_keep >= n_total:
        return np.arange(n_total, dtype=np.int64)
    return np.linspace(0, n_total - 1, num=n_keep, dtype=np.int64)


def _rolling_median_int(x: np.ndarray, window: int) -> np.ndarray:
    n = len(x)
    if n == 0:
        return np.asarray(x, dtype=np.int64)
    w = max(3, int(window) | 1)  # force odd window
    h = w // 2
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        l = max(0, i - h)
        r = min(n, i + h + 1)
        out[i] = int(np.median(x[l:r]))
    return out


def _regularize_pairwise_lags(pairwise_lags: np.ndarray, max_step_shift: int) -> np.ndarray:
    """Suppress outlier pairwise lags without replacing data rows."""

    x = np.asarray(pairwise_lags, dtype=np.int64).copy()
    n = len(x)
    if n <= 2:
        return x

    # Ignore first entry (always 0) and remove global bias to avoid cumulative drift.
    core = x[1:].copy()
    drift = int(round(float(np.median(core))))
    core = core - drift

    if len(core) >= 5:
        w = 9 if len(core) >= 9 else (len(core) | 1)
        local_med = _rolling_median_int(core, window=w)
        resid = core - local_med
        mad = float(np.median(np.abs(resid)))
        sigma = 1.4826 * mad
        thr = max(0.35 * float(max_step_shift), 3.5 * sigma, 2.0)
        bad = np.abs(resid) > thr
        core[bad] = local_med[bad]

    x[0] = 0
    x[1:] = np.clip(core, -max_step_shift, max_step_shift)
    return x


def _pairwise_lags(
    traces: np.ndarray,
    start: int,
    end: int,
    decimation: int,
    max_step_shift: int,
) -> np.ndarray:
    """Lag of each trace against previous trace on selected CC window."""

    n = int(traces.shape[0])
    if n <= 1 or end - start < 64:
        return np.zeros(n, dtype=np.int64)

    step = max(1, int(decimation))
    max_shift_dec = max(1, int(round(max_step_shift / step)))
    lags = np.zeros(n, dtype=np.int64)

    prev = traces[0, start:end]
    prev_dec = prev[::step] if step > 1 else prev
    for i in range(1, n):
        cur = traces[i, start:end]
        cur_dec = cur[::step] if step > 1 else cur
        lag = best_shift_fft(prev_dec, cur_dec, max_shift=max_shift_dec) * step
        lags[i] = int(np.clip(lag, -max_step_shift, max_step_shift))
        prev_dec = cur_dec

    return _regularize_pairwise_lags(lags, max_step_shift=max_step_shift)


def _pairwise_to_cumulative(pairwise_lags: np.ndarray, max_total_shift: int) -> np.ndarray:
    """Convert adjacent lags to absolute per-trace shift corrections."""

    if len(pairwise_lags) == 0:
        return np.asarray(pairwise_lags, dtype=np.int64)
    arr = np.asarray(pairwise_lags, dtype=np.float64)
    cumulative = np.cumsum(arr, dtype=np.float64)
    cumulative[0] = 0.0

    # Remove small systematic drift in pairwise lags to avoid long-run walk-off.
    if len(arr) > 8:
        core = arr[1:]
        q10, q90 = np.quantile(core, [0.10, 0.90])
        core_trim = core[(core >= q10) & (core <= q90)]
        drift = float(np.mean(core_trim if len(core_trim) else core))
        if np.isfinite(drift) and abs(drift) > 1e-4:
            idx = np.arange(len(cumulative), dtype=np.float64)
            cumulative -= drift * idx

    cumulative = np.rint(cumulative).astype(np.int64, copy=False)
    cumulative[0] = 0
    return np.clip(cumulative, -max_total_shift, max_total_shift).astype(np.int64, copy=False)


def _roughness_q99(traces: np.ndarray, start: int, end: int, max_rows: int = 2048) -> float:
    if traces.shape[0] < 3 or end - start < 16:
        return float("inf")
    idx = _sample_indices(traces.shape[0], min(max_rows, traces.shape[0]))
    x = np.asarray(traces[idx, start:end], dtype=np.float32)
    if x.shape[0] < 3:
        return float("inf")
    mid = 0.5 * (x[:-2] + x[2:])
    d = np.mean(np.abs(x[1:-1] - mid), axis=1)
    return float(np.quantile(d, 0.99))


def should_apply_alignment(
    before: tuple[float, float, float],
    after: tuple[float, float, float],
    traces_before: np.ndarray,
    traces_after: np.ndarray,
    start: int,
    end: int,
) -> bool:
    """Robust accept/reject for alignment result.

    Mean residual can be unstable with a few outliers. We also compare p95 and
    row-to-row roughness (more aligned waterfall => lower roughness).
    """

    mean_b, p95_b, std_b = before
    mean_a, p95_a, std_a = after

    rough_b = _roughness_q99(traces_before, start=start, end=end)
    rough_a = _roughness_q99(traces_after, start=start, end=end)
    if np.isfinite(rough_a) and np.isfinite(rough_b) and rough_a > 1.02 * rough_b:
        return False

    if mean_a <= 0.92 * mean_b:
        return True
    if p95_a <= 0.70 * p95_b and std_a <= 1.05 * std_b:
        return True
    if np.isfinite(rough_a) and np.isfinite(rough_b):
        if rough_a <= 0.90 * rough_b and p95_a <= 1.10 * p95_b:
            return True

    return False


def _apply_fine_pass(
    aligned: np.ndarray,
    shifts_total: np.ndarray,
    cfg: ParseConfig,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Small-lag refinement after coarse CC alignment.

    Uses full-resolution correlation with tiny lag bounds and keeps the update
    only if robust quality criteria improve.
    """

    if end - start < 512 or aligned.shape[0] < 4:
        return aligned, shifts_total

    fine_max_shift = 6
    pairwise = _pairwise_lags(
        traces=aligned,
        start=start,
        end=end,
        decimation=1,
        max_step_shift=fine_max_shift,
    )
    lags = _pairwise_to_cumulative(pairwise, max_total_shift=fine_max_shift)

    if float(np.mean(np.abs(lags[1:]))) < 0.05:
        return aligned, shifts_total

    remaining = np.maximum(0, cfg.max_shift - np.abs(shifts_total))
    lags = np.sign(lags) * np.minimum(np.abs(lags), remaining)
    if np.max(np.abs(lags)) == 0:
        return aligned, shifts_total

    before = estimate_residual_jitter(aligned, cfg)
    candidate = aligned.copy()
    for i, lag in enumerate(lags):
        if lag != 0:
            candidate[i] = shift_trace(candidate[i], -int(lag))
    after = estimate_residual_jitter(candidate, cfg)

    if should_apply_alignment(
        before=before,
        after=after,
        traces_before=aligned,
        traces_after=candidate,
        start=start,
        end=end,
    ):
        return candidate, shifts_total + lags

    return aligned, shifts_total


def shift_trace(trace: np.ndarray, shift: int) -> np.ndarray:
    out = np.empty_like(trace)
    if shift >= 0:
        out[:-shift or None] = trace[shift:]
        if shift > 0:
            out[-shift:] = trace[-1]
    else:
        out[-shift:] = trace[:shift]
        out[:-shift] = trace[0]
    return out


def align_traces_cc(traces: np.ndarray, cfg: ParseConfig) -> tuple[np.ndarray, np.ndarray]:
    start = cfg.cc_window_start
    end = min(traces.shape[1], start + cfg.cc_window_len)
    if end - start < 512:
        return traces.copy(), np.zeros(traces.shape[0], dtype=np.int64)

    decimation = max(1, int(cfg.align_decimation))
    aligned = traces.copy()
    shifts_total = np.zeros(traces.shape[0], dtype=np.int64)
    max_step_shift = max(6, min(64, int(cfg.max_shift)))
    n_iters = max(1, int(cfg.align_iters))

    for _ in range(n_iters):
        pairwise = _pairwise_lags(
            traces=aligned,
            start=start,
            end=end,
            decimation=decimation,
            max_step_shift=max_step_shift,
        )
        iter_shifts = _pairwise_to_cumulative(pairwise, max_total_shift=cfg.max_shift)

        # Keep cumulative correction bounded for each trace.
        remaining = np.maximum(0, cfg.max_shift - np.abs(shifts_total))
        iter_sign = np.sign(iter_shifts)
        iter_shifts = iter_sign * np.minimum(np.abs(iter_shifts), remaining)
        for i, lag in enumerate(iter_shifts):
            if lag != 0:
                aligned[i] = shift_trace(aligned[i], -int(lag))

        shifts_total += iter_shifts
        if float(np.mean(np.abs(iter_shifts[1:]))) < 0.25:
            break

    aligned, shifts_total = _apply_fine_pass(
        aligned=aligned,
        shifts_total=shifts_total,
        cfg=cfg,
        start=start,
        end=end,
    )

    return aligned, shifts_total


def estimate_residual_jitter(traces: np.ndarray, cfg: ParseConfig) -> tuple[float, float, float]:
    start = cfg.cc_window_start
    end = min(traces.shape[1], start + cfg.cc_window_len)
    if end - start < 256:
        return 0.0, 0.0, 0.0

    decimation = max(1, int(cfg.align_decimation))
    pairwise = _pairwise_lags(
        traces=traces,
        start=start,
        end=end,
        decimation=decimation,
        max_step_shift=max(6, min(64, int(cfg.max_shift))),
    )
    if len(pairwise) <= 1:
        return 0.0, 0.0, 0.0
    arr = np.asarray(pairwise[1:], dtype=np.int64)
    return float(np.mean(np.abs(arr))), float(np.quantile(np.abs(arr), 0.95)), float(np.std(arr))


def select_alignment_window(traces: np.ndarray, cfg: ParseConfig) -> int:
    """Pick cross-correlation window start with lowest residual jitter."""

    window = int(min(cfg.cc_window_len, traces.shape[1] - 1))
    if window < 1024:
        return int(cfg.cc_window_start)

    step = max(256, int(cfg.cc_scan_step))
    last_start = max(0, traces.shape[1] - window - 1)
    starts = list(range(0, last_start + 1, step))
    if last_start not in starts:
        starts.append(last_start)

    n_subset = min(384, traces.shape[0])
    subset = traces[_sample_indices(traces.shape[0], n_subset)]
    best_start = int(cfg.cc_window_start)
    best_score = float("inf")
    for start in starts:
        candidate_cfg = replace(cfg, cc_window_start=int(start), cc_window_len=window)
        score, _, _ = estimate_residual_jitter(subset, candidate_cfg)
        if score < best_score:
            best_score = score
            best_start = int(start)
    return best_start
