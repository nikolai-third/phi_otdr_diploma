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
        return x.copy()
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(x, kernel, mode="same")


def refine_starts_with_template(data: np.ndarray, starts: np.ndarray, trace_len: int) -> np.ndarray:
    """Refine starts by local template matching on gradient signal."""

    if len(starts) < 3:
        return starts

    n = len(data)
    ds = max(1, int(DET_RECOVERY_CORR_DECIMATION))
    window = max(256, min(int(DET_TEMPLATE_REFINE_LEN), max(512, trace_len // 3)))
    radius = max(80, int(DET_TEMPLATE_REFINE_RADIUS))

    smoothed = _moving_average(data, 7)
    gradient = np.zeros_like(smoothed)
    gradient[1:-1] = (smoothed[2:] - smoothed[:-2]) * 0.5
    gradient[0] = gradient[1]
    gradient[-1] = gradient[-2]

    valid = starts[(starts >= 1) & (starts + window + 2 < n)]
    if len(valid) < 3:
        return starts
    max_traces = min(len(valid), int(DET_TEMPLATE_REFINE_TRACES))
    pick = np.linspace(0, len(valid) - 1, num=max_traces, dtype=int)
    segments = np.vstack([gradient[int(valid[i]) : int(valid[i]) + window : ds] for i in pick])

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
        region = gradient[left : right + window + 1 : ds]
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
    max_shift_dec = max(1, int(round(cfg.max_shift / decimation)))

    aligned = traces.copy()
    shifts_total = np.zeros(traces.shape[0], dtype=np.int64)
    n_ref = min(256, traces.shape[0])
    n_iters = max(1, int(cfg.align_iters))

    for _ in range(n_iters):
        template = np.median(aligned[:n_ref, start:end], axis=0)
        template_dec = template[::decimation] if decimation > 1 else template

        iter_shifts = np.zeros(traces.shape[0], dtype=np.int64)
        for i in range(traces.shape[0]):
            cur = aligned[i, start:end]
            cur_dec = cur[::decimation] if decimation > 1 else cur
            lag = best_shift_fft(template_dec, cur_dec, max_shift=max_shift_dec) * decimation
            lag = int(np.clip(lag, -cfg.max_shift, cfg.max_shift))
            iter_shifts[i] = lag
            aligned[i] = shift_trace(aligned[i], -lag)

        shifts_total += iter_shifts
        if float(np.mean(np.abs(iter_shifts))) < 0.25:
            break

    return aligned, shifts_total


def estimate_residual_jitter(traces: np.ndarray, cfg: ParseConfig) -> tuple[float, float, float]:
    start = cfg.cc_window_start
    end = min(traces.shape[1], start + cfg.cc_window_len)
    if end - start < 256:
        return 0.0, 0.0, 0.0

    decimation = max(1, int(cfg.align_decimation))
    max_shift_dec = max(1, int(round(cfg.max_shift / decimation)))
    n_ref = min(256, traces.shape[0])
    ref = np.median(traces[:n_ref, start:end], axis=0)
    ref_dec = ref[::decimation] if decimation > 1 else ref

    shifts: list[int] = []
    for i in range(traces.shape[0]):
        cur = traces[i, start:end]
        cur_dec = cur[::decimation] if decimation > 1 else cur
        shift = best_shift_fft(ref_dec, cur_dec, max_shift=max_shift_dec) * decimation
        shifts.append(int(shift))

    arr = np.asarray(shifts, dtype=np.int64)
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

    subset = traces[: min(96, traces.shape[0])]
    best_start = int(cfg.cc_window_start)
    best_score = float("inf")
    for start in starts:
        candidate_cfg = replace(cfg, cc_window_start=int(start), cc_window_len=window)
        score, _, _ = estimate_residual_jitter(subset, candidate_cfg)
        if score < best_score:
            best_score = score
            best_start = int(start)
    return best_start
