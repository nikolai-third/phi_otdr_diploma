"""One-file reflectogram parsing pipeline with diagnostic plots."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path

import matplotlib

# Writable matplotlib cache in restricted environments.
_cache_root = (Path("data") / "interim" / ".cache").resolve()
_mpl_root = (Path("data") / "interim" / ".mplconfig").resolve()
_cache_root.mkdir(parents=True, exist_ok=True)
_mpl_root.mkdir(parents=True, exist_ok=True)
os.environ["XDG_CACHE_HOME"] = str(_cache_root)
os.environ["MPLCONFIGDIR"] = str(_mpl_root)

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


@dataclass(frozen=True)
class ParseConfig:
    trace_len: int | None = None
    max_traces: int = 500
    min_gap_factor: float = 0.82
    low_level_factor: float = 0.10
    rise_factor: float = 0.075
    max_shift: int = 300
    cc_window_start: int = 500
    cc_window_len: int = 12_000
    raw_plot_points: int = 1_000_000
    auto_tune: bool = False
    adc_fs_hz: float = 50_000_000.0
    # Preprocessing for robust start detection
    decimation: int = 10
    ma_window: int = 31
    envelope_alpha: float = 0.995
    refine_radius: int = 300
    align_iters: int = 1
    align_decimation: int = 4
    max_samples: int | None = 12_000_000
    fill_missing: bool = False
    fill_gap_factor: float = 1.6
    recovery_search_radius: int = 1200
    recovery_anchor_tol: int = 450
    recovery_low_weight: float = 0.8
    recovery_pre_window: int = 220
    recovery_min_spacing_factor: float = 0.82
    recovery_corr_len: int = 3000
    recovery_corr_decimation: int = 8
    period_min: int = 20_000
    period_max: int = 260_000
    template_refine_radius: int = 450
    template_refine_len: int = 3000
    template_refine_traces: int = 48
    waterfall_cmap: str = "jet"
    waterfall_exp_alpha: float = 4.0
    auto_select_cc_window: bool = True
    cc_scan_step: int = 2000


def _read_data_stream(path: Path, max_points: int | None = None) -> np.ndarray:
    """Read parquet `data` column in streaming mode."""

    pf = pq.ParquetFile(path)
    chunks: list[np.ndarray] = []
    total = 0

    for batch in pf.iter_batches(columns=["data"], batch_size=1_000_000):
        arr = batch.column(0).to_numpy(zero_copy_only=False)
        if max_points is not None and total + len(arr) > max_points:
            arr = arr[: max_points - total]
        chunks.append(np.asarray(arr, dtype=np.float64))
        total += len(arr)
        if max_points is not None and total >= max_points:
            break

    if not chunks:
        raise ValueError(f"No data in parquet column 'data': {path}")

    return np.concatenate(chunks)


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(x, kernel, mode="same")


def _envelopes(x: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute upper/lower envelopes and mid-envelope (fast EMA-style)."""

    upper = np.empty_like(x)
    lower = np.empty_like(x)
    upper[0] = x[0]
    lower[0] = x[0]

    for i in range(1, len(x)):
        upper[i] = x[i] if x[i] > upper[i - 1] else alpha * upper[i - 1] + (1.0 - alpha) * x[i]
        lower[i] = x[i] if x[i] < lower[i - 1] else alpha * lower[i - 1] + (1.0 - alpha) * x[i]

    mid = 0.5 * (upper + lower)
    return upper, lower, mid


def _collect_start_candidates(data: np.ndarray, cfg: ParseConfig) -> np.ndarray:
    """Return dense start candidates before period-based spacing filter."""

    dec = max(1, int(cfg.decimation))
    dec_data = data[::dec]
    sm = _moving_average(dec_data, cfg.ma_window)
    _, _, mid = _envelopes(sm, alpha=cfg.envelope_alpha)

    lo_q = float(np.quantile(mid, 0.02))
    hi_q = float(np.quantile(mid, 0.98))
    dyn = max(1e-12, hi_q - lo_q)

    low_thr = lo_q + cfg.low_level_factor * dyn
    rise_thr = cfg.rise_factor * dyn

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

    radius = max(8, int(cfg.refine_radius))
    for p in potential:
        center = int(p * dec)
        l = max(1, center - radius)
        r = min(len(data) - 2, center + radius)
        local = raw_grad[l:r]
        if local.size == 0:
            continue
        idx = int(np.argmax(local)) + l
        raw_starts.append(idx)

    if not raw_starts:
        return np.array([], dtype=np.int64)

    return np.array(sorted(raw_starts), dtype=np.int64)


def _estimate_trace_len_from_candidates(candidates: np.ndarray, fallback: int = 55_000) -> int:
    """Estimate reflectogram length from candidate start gaps without external prior."""

    if len(candidates) < 3:
        return fallback

    diffs = np.diff(candidates).astype(np.float64)
    diffs = diffs[(diffs > 1_000) & (diffs < 2_000_000)]
    if len(diffs) == 0:
        return fallback

    period_votes: list[float] = []
    vote_weights: list[float] = []
    for d in diffs:
        max_k = min(12, max(1, int(d // 10_000)))
        for k in range(1, max_k + 1):
            p = d / float(k)
            if 5_000 <= p <= 300_000:
                period_votes.append(p)
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
    if len(near) == 0:
        est = center
    else:
        est = float(np.median(near))
    return int(round(est))


def _estimate_trace_len_from_autocorr(data: np.ndarray, cfg: ParseConfig) -> int:
    """Estimate reflectogram period from signal autocorrelation on decimated stream."""

    dec = max(1, int(cfg.decimation))
    y = data[::dec]
    if len(y) < 2048:
        return 55_000

    s = _moving_average(y, cfg.ma_window)
    s = s - float(np.median(s))
    n = len(s)

    size = 1 << int(np.ceil(np.log2(2 * n - 1)))
    sf = np.fft.rfft(s, size)
    ac = np.fft.irfft(sf * np.conj(sf), size)[:n]
    ac[0] = 0.0

    min_lag = max(8, int(cfg.period_min // dec))
    max_lag = min(n - 2, int(cfg.period_max // dec))
    if max_lag <= min_lag:
        return 55_000

    arr = ac[min_lag : max_lag + 1]
    if len(arr) < 5:
        return 55_000
    arr = _moving_average(arr, 9)

    peaks = np.where((arr[1:-1] > arr[:-2]) & (arr[1:-1] >= arr[2:]))[0] + 1
    if len(peaks) == 0:
        best = int(np.argmax(arr))
    else:
        vals = arr[peaks]
        vmax = float(np.max(vals))
        good = peaks[vals >= 0.70 * vmax]
        best = int(np.min(good)) if len(good) else int(peaks[int(np.argmax(vals))])
    return int((best + min_lag) * dec)


def _infer_trace_len(data: np.ndarray, cfg: ParseConfig, candidates: np.ndarray) -> int:
    """Infer trace length without external value, blending autocorr and candidate-gap estimates."""

    if cfg.trace_len is not None:
        return int(cfg.trace_len)

    p_ac = _estimate_trace_len_from_autocorr(data, cfg)
    p_cand = _estimate_trace_len_from_candidates(candidates, fallback=p_ac)
    ratio = float(p_cand) / float(max(1, p_ac))
    if 0.60 <= ratio <= 1.60:
        p = int(round(0.75 * p_ac + 0.25 * p_cand))
    else:
        p = p_ac
    return int(np.clip(p, cfg.period_min, cfg.period_max))


def _refine_starts_with_template(data: np.ndarray, starts: np.ndarray, trace_len: int, cfg: ParseConfig) -> np.ndarray:
    """Refine starts by local template matching on gradient signal."""

    if len(starts) < 3:
        return starts

    n = len(data)
    ds = max(1, int(cfg.recovery_corr_decimation))
    w = max(256, min(int(cfg.template_refine_len), max(512, trace_len // 3)))
    r = max(80, int(cfg.template_refine_radius))

    sm = _moving_average(data, 7)
    grad = np.zeros_like(sm)
    grad[1:-1] = (sm[2:] - sm[:-2]) * 0.5
    grad[0] = grad[1]
    grad[-1] = grad[-2]

    valid = starts[(starts >= 1) & (starts + w + 2 < n)]
    if len(valid) < 3:
        return starts
    m = min(len(valid), int(cfg.template_refine_traces))
    pick = np.linspace(0, len(valid) - 1, num=m, dtype=int)
    segs = np.vstack([grad[int(valid[i]) : int(valid[i]) + w : ds] for i in pick])
    template = np.median(segs, axis=0)
    template = template - float(np.mean(template))
    tnorm = float(np.linalg.norm(template)) + 1e-12
    template = template / tnorm

    refined: list[int] = []
    for s in starts:
        s = int(s)
        l = max(1, s - r)
        rr = min(n - w - 2, s + r)
        if rr <= l:
            refined.append(s)
            continue
        region = grad[l : rr + w + 1 : ds]
        corr = np.correlate(region, template, mode="valid")
        if len(corr) == 0:
            refined.append(s)
            continue
        best = int(np.argmax(corr))
        refined.append(int(l + best * ds))

    arr = np.array(sorted(refined), dtype=np.int64)
    min_spacing = max(1, int(cfg.recovery_min_spacing_factor * trace_len))
    packed: list[int] = [int(arr[0])]
    for s in arr[1:]:
        if int(s) - packed[-1] >= min_spacing:
            packed.append(int(s))
    out = np.array(packed, dtype=np.int64)
    return out[out + trace_len <= n]


def _detect_starts(data: np.ndarray, cfg: ParseConfig, trace_len: int) -> np.ndarray:
    starts = _collect_start_candidates(data, cfg)
    if len(starts) == 0:
        return starts

    min_gap = max(1, int(trace_len * cfg.min_gap_factor))
    filtered = [int(starts[0])]
    for s in starts[1:]:
        if int(s) - filtered[-1] >= min_gap:
            filtered.append(int(s))

    starts = np.array(filtered, dtype=np.int64)
    if len(starts) <= 1:
        return starts
    if cfg.fill_missing:
        starts = _recover_starts_periodic(data=data, anchors=starts, cfg=cfg, trace_len=trace_len)
    return _refine_starts_with_template(data=data, starts=starts, trace_len=trace_len, cfg=cfg)


def _recover_starts_periodic(data: np.ndarray, anchors: np.ndarray, cfg: ParseConfig, trace_len: int) -> np.ndarray:
    """Recover missing starts by phase-locked periodic tracking around v1 anchors."""

    n_samples = len(data)
    if len(anchors) < 2:
        return anchors

    diffs = np.diff(anchors)
    # Estimate true period from anchor gaps and keep it close to inferred trace_len.
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

    # Interpolate starts between anchor pairs using local period implied by each gap.
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

    # Extrapolate to file edges using estimated period.
    while predicted and predicted[0] - period >= 0:
        predicted.insert(0, predicted[0] - period)
    while predicted and predicted[-1] + period + trace_len <= n_samples:
        predicted.append(predicted[-1] + period)

    if not predicted:
        return anchors

    # Local onset refinement around predicted positions.
    sm = _moving_average(data, max(7, cfg.ma_window // 2))
    grad = np.zeros_like(sm)
    grad[1:-1] = (sm[2:] - sm[:-2]) * 0.5
    grad[0] = grad[1]
    grad[-1] = grad[-2]

    refine_r = max(40, min(200, int(cfg.refine_radius)))
    refined: list[int] = []
    for p in predicted:
        l = max(1, int(p) - refine_r)
        r = min(n_samples - 2, int(p) + refine_r)
        if r <= l:
            continue
        idx = int(l + int(np.argmax(grad[l : r + 1])))
        refined.append(idx)

    if not refined:
        return anchors

    rec_arr = np.array(sorted(set(refined)), dtype=np.int64)
    rec_arr = rec_arr[rec_arr + trace_len <= n_samples]
    if len(rec_arr) == 0:
        return anchors

    min_spacing = max(1, int(cfg.recovery_min_spacing_factor * period))
    packed: list[int] = [int(rec_arr[0])]
    for s in rec_arr[1:]:
        if int(s) - packed[-1] >= min_spacing:
            packed.append(int(s))
    rec_arr = np.array(packed, dtype=np.int64)

    # Safety gate: keep anchors if recovery became irregular.
    if len(rec_arr) > 2:
        rd = np.diff(rec_arr)
        if float(np.std(rd)) > 0.25 * period:
            return anchors
    return rec_arr


def _extract_reflectograms(data: np.ndarray, starts: np.ndarray, trace_len: int, max_traces: int) -> np.ndarray:
    starts = starts[:max_traces]
    traces: list[np.ndarray] = []
    for s in starts:
        e = int(s) + trace_len
        if e <= len(data):
            traces.append(data[int(s):e])
    if not traces:
        raise ValueError("No complete traces extracted; tune parser thresholds")
    return np.vstack(traces)


def _best_shift_fft(ref: np.ndarray, cur: np.ndarray, max_shift: int) -> int:
    ref0 = ref - np.mean(ref)
    cur0 = cur - np.mean(cur)
    n = len(ref0)
    if n == 0:
        return 0
    size = 1 << int(np.ceil(np.log2(2 * n - 1)))
    ref_fft = np.fft.rfft(ref0, size)
    cur_fft = np.fft.rfft(cur0, size)
    corr = np.fft.irfft(ref_fft * np.conj(cur_fft), size)
    # Rearrange for lags [-n+1, n-1]
    corr = np.concatenate((corr[-(n - 1) :], corr[:n]))
    lags = np.arange(-(n - 1), n)
    mask = (lags >= -max_shift) & (lags <= max_shift)
    if not np.any(mask):
        return 0
    idx = np.argmax(corr[mask])
    return int(lags[mask][idx])


def _shift_trace(trace: np.ndarray, s: int) -> np.ndarray:
    out = np.empty_like(trace)
    if s >= 0:
        out[:-s or None] = trace[s:]
        if s > 0:
            out[-s:] = trace[-1]
    else:
        out[-s:] = trace[:s]
        out[: -s] = trace[0]
    return out


def _align_traces_cc(traces: np.ndarray, cfg: ParseConfig) -> tuple[np.ndarray, np.ndarray]:
    start = cfg.cc_window_start
    end = min(traces.shape[1], start + cfg.cc_window_len)
    if end - start < 512:
        return traces.copy(), np.zeros(traces.shape[0], dtype=np.int64)

    ref = traces[0, start:end]
    aligned = np.empty_like(traces)
    shifts = np.zeros(traces.shape[0], dtype=np.int64)

    for i in range(traces.shape[0]):
        cur = traces[i, start:end]
        s = _best_shift_fft(ref, cur, cfg.max_shift)
        shifts[i] = s
        aligned[i] = _shift_trace(traces[i], s)

    return aligned, shifts


def _estimate_residual_jitter(traces: np.ndarray, cfg: ParseConfig) -> tuple[float, float, float]:
    start = cfg.cc_window_start
    end = min(traces.shape[1], start + cfg.cc_window_len)
    if end - start < 256:
        return 0.0, 0.0, 0.0

    ref = traces[0, start:end]
    shifts: list[int] = []
    for i in range(traces.shape[0]):
        cur = traces[i, start:end]
        s = _best_shift_fft(ref, cur, max_shift=cfg.max_shift)
        shifts.append(int(s))

    arr = np.asarray(shifts, dtype=np.int64)
    return float(np.mean(np.abs(arr))), float(np.quantile(np.abs(arr), 0.95)), float(np.std(arr))


def _select_alignment_window(traces: np.ndarray, cfg: ParseConfig) -> int:
    """Pick cc window start with lowest residual jitter on a coarse grid."""

    w = int(min(cfg.cc_window_len, traces.shape[1] - 1))
    if w < 1024:
        return int(cfg.cc_window_start)

    step = max(256, int(cfg.cc_scan_step))
    last_start = max(0, traces.shape[1] - w - 1)
    starts = list(range(0, last_start + 1, step))
    if last_start not in starts:
        starts.append(last_start)

    subset = traces[: min(96, traces.shape[0])]
    best_start = int(cfg.cc_window_start)
    best_score = float("inf")
    for st in starts:
        c = replace(cfg, cc_window_start=int(st), cc_window_len=w)
        score, _, _ = _estimate_residual_jitter(subset, c)
        if score < best_score:
            best_score = score
            best_start = int(st)
    return best_start


def _save_plot(path: Path, fig: plt.Figure) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_raw_segment(data: np.ndarray, starts: np.ndarray, out: Path, points: int, fs_hz: float) -> None:
    n = min(points, len(data))
    dt_us = 1e6 / fs_hz
    x_us = np.arange(n, dtype=np.float64) * dt_us

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x_us, data[:n], linewidth=0.8)
    starts_vis = starts[starts < n]
    if len(starts_vis) > 0:
        ax.scatter(starts_vis * dt_us, data[starts_vis], s=8)
    ax.set_title("Raw signal segment with detected starts")
    ax.set_xlabel("Time, us")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.2)
    _save_plot(out, fig)


def _plot_waterfall(traces: np.ndarray, out: Path, title: str, fs_hz: float, cmap: str, exp_alpha: float) -> None:
    dt_us = 1e6 / fs_hz
    extent = [0.0, traces.shape[1] * dt_us, 0.0, float(traces.shape[0])]

    data = traces.astype(np.float64)
    q_lo = float(np.quantile(data, 0.005))
    q_hi = float(np.quantile(data, 0.995))
    span = max(1e-12, q_hi - q_lo)
    normed = np.clip((data - q_lo) / span, 0.0, 1.0)
    if exp_alpha > 0.0:
        normed = (1.0 - np.exp(-exp_alpha * normed)) / (1.0 - np.exp(-exp_alpha))

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(normed, aspect="auto", origin="lower", extent=extent, cmap=cmap, vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Time inside reflectogram, us")
    ax.set_ylabel("Trace index")
    _save_plot(out, fig)


def _auto_tune_config(data: np.ndarray, cfg: ParseConfig, trace_len: int) -> ParseConfig:
    tune_data = data[: min(len(data), 12_000_000)]

    low_grid = [0.07, 0.09, 0.10, 0.12]
    rise_grid = [0.05, 0.07, 0.09, 0.11]
    gap_grid = [0.78, 0.82, 0.90, 1.00]

    best_score = np.inf
    best_cfg = cfg

    for low in low_grid:
        for rise in rise_grid:
            for gap in gap_grid:
                candidate = ParseConfig(
                    trace_len=trace_len,
                    max_traces=cfg.max_traces,
                    min_gap_factor=gap,
                    low_level_factor=low,
                    rise_factor=rise,
                    max_shift=cfg.max_shift,
                    cc_window_start=cfg.cc_window_start,
                    cc_window_len=cfg.cc_window_len,
                    raw_plot_points=cfg.raw_plot_points,
                    auto_tune=cfg.auto_tune,
                    adc_fs_hz=cfg.adc_fs_hz,
                    decimation=cfg.decimation,
                    ma_window=cfg.ma_window,
                    envelope_alpha=cfg.envelope_alpha,
                    refine_radius=cfg.refine_radius,
                    align_iters=cfg.align_iters,
                    align_decimation=cfg.align_decimation,
                    max_samples=cfg.max_samples,
                    fill_missing=cfg.fill_missing,
                    fill_gap_factor=cfg.fill_gap_factor,
                    recovery_search_radius=cfg.recovery_search_radius,
                    recovery_anchor_tol=cfg.recovery_anchor_tol,
                    recovery_low_weight=cfg.recovery_low_weight,
                    recovery_pre_window=cfg.recovery_pre_window,
                    recovery_min_spacing_factor=cfg.recovery_min_spacing_factor,
                    recovery_corr_len=cfg.recovery_corr_len,
                    recovery_corr_decimation=cfg.recovery_corr_decimation,
                    period_min=cfg.period_min,
                    period_max=cfg.period_max,
                    template_refine_radius=cfg.template_refine_radius,
                    template_refine_len=cfg.template_refine_len,
                    template_refine_traces=cfg.template_refine_traces,
                    waterfall_cmap=cfg.waterfall_cmap,
                    waterfall_exp_alpha=cfg.waterfall_exp_alpha,
                    auto_select_cc_window=cfg.auto_select_cc_window,
                    cc_scan_step=cfg.cc_scan_step,
                )

                starts = _detect_starts(tune_data, candidate, trace_len=trace_len)
                if len(starts) < 20:
                    continue
                try:
                    traces = _extract_reflectograms(
                        tune_data, starts, trace_len=trace_len, max_traces=candidate.max_traces
                    )
                    _, shifts = _align_traces_cc(traces, candidate)
                except Exception:
                    continue

                mean_abs = float(np.mean(np.abs(shifts)))
                p95 = float(np.quantile(np.abs(shifts), 0.95))
                std = float(np.std(shifts))
                n = traces.shape[0]
                score = mean_abs + 0.35 * p95 + 0.03 * std - 0.02 * n
                if score < best_score:
                    best_score = score
                    best_cfg = candidate

    return best_cfg


def _extract_with_fallbacks(data: np.ndarray, cfg: ParseConfig, trace_len: int) -> tuple[ParseConfig, np.ndarray, np.ndarray]:
    """Try primary config and a few nearby configs to avoid empty extraction."""

    candidates: list[ParseConfig] = [cfg]
    for low in [cfg.low_level_factor, 0.09, 0.10, 0.12]:
        for rise in [cfg.rise_factor, 0.07, 0.09, 0.11]:
            for gap in [cfg.min_gap_factor, 0.78, 0.82, 0.90, 1.00]:
                for dec in [cfg.decimation, 8, 10]:
                    candidates.append(
                        ParseConfig(
                            trace_len=trace_len,
                            max_traces=cfg.max_traces,
                            min_gap_factor=gap,
                            low_level_factor=low,
                            rise_factor=rise,
                            max_shift=cfg.max_shift,
                            cc_window_start=cfg.cc_window_start,
                            cc_window_len=cfg.cc_window_len,
                            raw_plot_points=cfg.raw_plot_points,
                            auto_tune=cfg.auto_tune,
                            adc_fs_hz=cfg.adc_fs_hz,
                            decimation=dec,
                            ma_window=cfg.ma_window,
                            envelope_alpha=cfg.envelope_alpha,
                            refine_radius=cfg.refine_radius,
                            align_iters=cfg.align_iters,
                            align_decimation=cfg.align_decimation,
                            max_samples=cfg.max_samples,
                            fill_missing=cfg.fill_missing,
                            fill_gap_factor=cfg.fill_gap_factor,
                            recovery_search_radius=cfg.recovery_search_radius,
                            recovery_anchor_tol=cfg.recovery_anchor_tol,
                            recovery_low_weight=cfg.recovery_low_weight,
                            recovery_pre_window=cfg.recovery_pre_window,
                            recovery_min_spacing_factor=cfg.recovery_min_spacing_factor,
                            recovery_corr_len=cfg.recovery_corr_len,
                            recovery_corr_decimation=cfg.recovery_corr_decimation,
                            period_min=cfg.period_min,
                            period_max=cfg.period_max,
                            template_refine_radius=cfg.template_refine_radius,
                            template_refine_len=cfg.template_refine_len,
                            template_refine_traces=cfg.template_refine_traces,
                            waterfall_cmap=cfg.waterfall_cmap,
                            waterfall_exp_alpha=cfg.waterfall_exp_alpha,
                            auto_select_cc_window=cfg.auto_select_cc_window,
                            cc_scan_step=cfg.cc_scan_step,
                        )
                    )

    seen = set()
    best: tuple[int, ParseConfig, np.ndarray, np.ndarray] | None = None
    for c in candidates:
        key = (c.low_level_factor, c.rise_factor, c.min_gap_factor, c.decimation)
        if key in seen:
            continue
        seen.add(key)
        starts = _detect_starts(data, c, trace_len=trace_len)
        if len(starts) == 0:
            continue
        try:
            traces = _extract_reflectograms(data, starts, trace_len=trace_len, max_traces=c.max_traces)
        except Exception:
            continue
        n = traces.shape[0]
        if best is None or n > best[0]:
            best = (n, c, starts, traces)
            if n >= c.max_traces:
                break

    if best is None:
        raise ValueError("No complete traces extracted; tune parser thresholds")
    return best[1], best[2], best[3]


def run_one_file(path: Path, outdir: Path, cfg: ParseConfig) -> dict[str, float | int | str]:
    data = _read_data_stream(path, max_points=cfg.max_samples)
    raw_candidates = _collect_start_candidates(data, cfg)
    if cfg.trace_len is None:
        trace_len = _infer_trace_len(data, cfg, raw_candidates)
        trace_len_source = "inferred"
    else:
        trace_len = int(cfg.trace_len)
        trace_len_source = "manual"
    cfg_work = replace(cfg, trace_len=trace_len)

    if cfg_work.auto_tune:
        cfg_work = _auto_tune_config(data, cfg_work, trace_len=trace_len)

    cfg_work, starts, traces = _extract_with_fallbacks(data, cfg_work, trace_len=trace_len)
    if cfg_work.auto_select_cc_window:
        cc_start = _select_alignment_window(traces, cfg_work)
        cfg_align = replace(cfg_work, cc_window_start=cc_start)
    else:
        cfg_align = cfg_work

    residual_before = _estimate_residual_jitter(traces, cfg_align)
    aligned_candidate, shifts_candidate = _align_traces_cc(traces, cfg_align)
    residual_after_candidate = _estimate_residual_jitter(aligned_candidate, cfg_align)

    alignment_applied = residual_after_candidate[0] < residual_before[0]
    if alignment_applied:
        aligned = aligned_candidate
        shifts = shifts_candidate
        residual_after = residual_after_candidate
    else:
        aligned = traces.copy()
        shifts = np.zeros(traces.shape[0], dtype=np.int64)
        residual_after = residual_before

    _plot_raw_segment(
        data,
        starts,
        outdir / "parser_raw_segment.png",
        points=cfg_work.raw_plot_points,
        fs_hz=cfg_work.adc_fs_hz,
    )
    _plot_waterfall(
        traces,
        outdir / "parser_waterfall_before.png",
        "Waterfall before alignment",
        fs_hz=cfg_work.adc_fs_hz,
        cmap=cfg_work.waterfall_cmap,
        exp_alpha=cfg_work.waterfall_exp_alpha,
    )
    _plot_waterfall(
        aligned,
        outdir / "parser_waterfall_after.png",
        "Waterfall after cross-correlation alignment",
        fs_hz=cfg_work.adc_fs_hz,
        cmap=cfg_work.waterfall_cmap,
        exp_alpha=cfg_work.waterfall_exp_alpha,
    )

    detected_period = int(round(float(np.median(np.diff(starts))))) if len(starts) > 1 else int(trace_len)
    detected_period = max(1, detected_period)
    if len(starts) > 0:
        first_start = int(starts[0])
        max_start = len(data) - trace_len
        expected_traces = max(0, (max_start - first_start) // trace_len + 1) if max_start >= first_start else 0
    else:
        expected_traces = 0
    coverage = float(len(starts) / expected_traces) if expected_traces > 0 else 0.0

    metrics = {
        "n_samples": int(len(data)),
        "n_detected_starts": int(len(starts)),
        "n_extracted_traces": int(traces.shape[0]),
        "first_start": int(starts[0]) if len(starts) else -1,
        "last_start": int(starts[-1]) if len(starts) else -1,
        "trace_len": int(traces.shape[1]),
        "trace_len_source": trace_len_source,
        "trace_duration_us": float(traces.shape[1] * 1e6 / cfg_work.adc_fs_hz),
        "shift_abs_mean": float(np.mean(np.abs(shifts))),
        "shift_abs_p95": float(np.quantile(np.abs(shifts), 0.95)),
        "shift_std": float(np.std(shifts)),
        "residual_before_abs_mean": residual_before[0],
        "residual_before_abs_p95": residual_before[1],
        "residual_after_abs_mean": residual_after[0],
        "residual_after_abs_p95": residual_after[1],
        "alignment_applied": int(alignment_applied),
        "cc_window_start_used": int(cfg_align.cc_window_start),
        "cc_window_len_used": int(cfg_align.cc_window_len),
        "expected_traces": int(expected_traces),
        "coverage_ratio": coverage,
        "detected_period_points": int(detected_period),
        "detected_period_us": float(detected_period * 1e6 / cfg_work.adc_fs_hz),
    }

    lines = [
        "# Parser Diagnostics",
        "",
        f"- file: `{path}`",
        f"- ADC sampling rate: **{cfg_work.adc_fs_hz:.0f} Hz**",
        f"- samples: **{metrics['n_samples']}**",
        f"- detected starts: **{metrics['n_detected_starts']}**",
        f"- extracted traces: **{metrics['n_extracted_traces']}**",
        f"- first start: **{metrics['first_start']}**",
        f"- last start: **{metrics['last_start']}**",
        f"- trace_len: **{metrics['trace_len']} points**",
        f"- trace_len source: **{metrics['trace_len_source']}**",
        f"- detected period: **{metrics['detected_period_points']} points ({metrics['detected_period_us']:.3f} us)**",
        f"- expected traces by period: **{metrics['expected_traces']}**",
        f"- coverage: **{100.0 * metrics['coverage_ratio']:.2f}%**",
        f"- trace duration: **{metrics['trace_duration_us']:.3f} us**",
        f"- mean |shift|: **{metrics['shift_abs_mean']:.3f} points**",
        f"- p95 |shift|: **{metrics['shift_abs_p95']:.3f} points**",
        f"- shift std: **{metrics['shift_std']:.3f} points**",
        f"- residual jitter before (mean |shift|): **{metrics['residual_before_abs_mean']:.3f} points**",
        f"- residual jitter before (p95 |shift|): **{metrics['residual_before_abs_p95']:.3f} points**",
        f"- residual jitter after (mean |shift|): **{metrics['residual_after_abs_mean']:.3f} points**",
        f"- residual jitter after (p95 |shift|): **{metrics['residual_after_abs_p95']:.3f} points**",
        f"- cc window used: **start={metrics['cc_window_start_used']}, len={metrics['cc_window_len_used']}**",
        f"- alignment applied: **{bool(metrics['alignment_applied'])}**",
        "",
        "## Files",
        "",
        "- parser_raw_segment.png",
        "- parser_waterfall_before.png",
        "- parser_waterfall_after.png",
    ]
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "parser_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (outdir / "parser_params.json").write_text(
        json.dumps(
            {
                "trace_len": cfg.trace_len,
                "trace_len_used": trace_len,
                "trace_len_source": trace_len_source,
                "max_traces": cfg_work.max_traces,
                "min_gap_factor": cfg_work.min_gap_factor,
                "low_level_factor": cfg_work.low_level_factor,
                "rise_factor": cfg_work.rise_factor,
                "max_shift": cfg_work.max_shift,
                "cc_window_start": cfg_work.cc_window_start,
                "cc_window_len": cfg_work.cc_window_len,
                "raw_plot_points": cfg_work.raw_plot_points,
                "auto_tune": cfg_work.auto_tune,
                "adc_fs_hz": cfg_work.adc_fs_hz,
                "decimation": cfg_work.decimation,
                "ma_window": cfg_work.ma_window,
                "envelope_alpha": cfg_work.envelope_alpha,
                "refine_radius": cfg_work.refine_radius,
                "align_iters": cfg_work.align_iters,
                "align_decimation": cfg_work.align_decimation,
                "max_samples": cfg_work.max_samples,
                "fill_missing": cfg_work.fill_missing,
                "fill_gap_factor": cfg_work.fill_gap_factor,
                "recovery_search_radius": cfg_work.recovery_search_radius,
                "recovery_anchor_tol": cfg_work.recovery_anchor_tol,
                "recovery_low_weight": cfg_work.recovery_low_weight,
                "recovery_pre_window": cfg_work.recovery_pre_window,
                "recovery_min_spacing_factor": cfg_work.recovery_min_spacing_factor,
                "recovery_corr_len": cfg_work.recovery_corr_len,
                "recovery_corr_decimation": cfg_work.recovery_corr_decimation,
                "period_min": cfg_work.period_min,
                "period_max": cfg_work.period_max,
                "template_refine_radius": cfg_work.template_refine_radius,
                "template_refine_len": cfg_work.template_refine_len,
                "template_refine_traces": cfg_work.template_refine_traces,
                "waterfall_cmap": cfg_work.waterfall_cmap,
                "waterfall_exp_alpha": cfg_work.waterfall_exp_alpha,
                "auto_select_cc_window": cfg_work.auto_select_cc_window,
                "cc_scan_step": cfg_work.cc_scan_step,
                "cc_window_start_used": int(cfg_align.cc_window_start),
                "cc_window_len_used": int(cfg_align.cc_window_len),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.parser.one_file", description="Run reflectogram parser on one file")
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("reports/figures/parser"))
    parser.add_argument(
        "--trace-len",
        type=int,
        default=None,
        help="Reflectogram length in points. If omitted, parser estimates it from signal.",
    )
    parser.add_argument("--max-traces", type=int, default=500)
    parser.add_argument("--min-gap-factor", type=float, default=0.82)
    parser.add_argument("--low-level-factor", type=float, default=0.10)
    parser.add_argument("--rise-factor", type=float, default=0.075)
    parser.add_argument("--max-shift", type=int, default=300)
    parser.add_argument("--cc-window-start", type=int, default=500)
    parser.add_argument("--cc-window-len", type=int, default=12000)
    parser.add_argument("--raw-plot-points", type=int, default=1_000_000)
    parser.add_argument("--adc-fs-hz", type=float, default=50_000_000.0)
    parser.add_argument("--decimation", type=int, default=10)
    parser.add_argument("--ma-window", type=int, default=31)
    parser.add_argument("--envelope-alpha", type=float, default=0.995)
    parser.add_argument("--refine-radius", type=int, default=300)
    parser.add_argument("--align-iters", type=int, default=1)
    parser.add_argument("--align-decimation", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=12_000_000)
    parser.add_argument("--fill-missing", action="store_true", help="Interpolate missing starts using trace period")
    parser.add_argument("--fill-gap-factor", type=float, default=1.6)
    parser.add_argument("--recovery-search-radius", type=int, default=1200)
    parser.add_argument("--recovery-anchor-tol", type=int, default=450)
    parser.add_argument("--recovery-low-weight", type=float, default=0.8)
    parser.add_argument("--recovery-pre-window", type=int, default=220)
    parser.add_argument("--recovery-min-spacing-factor", type=float, default=0.82)
    parser.add_argument("--recovery-corr-len", type=int, default=3000)
    parser.add_argument("--recovery-corr-decimation", type=int, default=8)
    parser.add_argument("--period-min", type=int, default=20_000)
    parser.add_argument("--period-max", type=int, default=260_000)
    parser.add_argument("--template-refine-radius", type=int, default=450)
    parser.add_argument("--template-refine-len", type=int, default=3000)
    parser.add_argument("--template-refine-traces", type=int, default=48)
    parser.add_argument("--waterfall-cmap", type=str, default="jet")
    parser.add_argument("--waterfall-exp-alpha", type=float, default=4.0)
    parser.add_argument("--auto-select-cc-window", action="store_true", default=True)
    parser.add_argument("--no-auto-select-cc-window", dest="auto_select_cc_window", action="store_false")
    parser.add_argument("--cc-scan-step", type=int, default=2000)
    parser.add_argument("--auto-tune", action="store_true", help="Auto-select thresholds by jitter score")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = ParseConfig(
        trace_len=args.trace_len,
        max_traces=args.max_traces,
        min_gap_factor=args.min_gap_factor,
        low_level_factor=args.low_level_factor,
        rise_factor=args.rise_factor,
        max_shift=args.max_shift,
        cc_window_start=args.cc_window_start,
        cc_window_len=args.cc_window_len,
        raw_plot_points=args.raw_plot_points,
        auto_tune=args.auto_tune,
        adc_fs_hz=args.adc_fs_hz,
        decimation=args.decimation,
        ma_window=args.ma_window,
        envelope_alpha=args.envelope_alpha,
        refine_radius=args.refine_radius,
        align_iters=args.align_iters,
        align_decimation=args.align_decimation,
        max_samples=args.max_samples,
        fill_missing=args.fill_missing,
        fill_gap_factor=args.fill_gap_factor,
        recovery_search_radius=args.recovery_search_radius,
        recovery_anchor_tol=args.recovery_anchor_tol,
        recovery_low_weight=args.recovery_low_weight,
        recovery_pre_window=args.recovery_pre_window,
        recovery_min_spacing_factor=args.recovery_min_spacing_factor,
        recovery_corr_len=args.recovery_corr_len,
        recovery_corr_decimation=args.recovery_corr_decimation,
        period_min=args.period_min,
        period_max=args.period_max,
        template_refine_radius=args.template_refine_radius,
        template_refine_len=args.template_refine_len,
        template_refine_traces=args.template_refine_traces,
        waterfall_cmap=args.waterfall_cmap,
        waterfall_exp_alpha=args.waterfall_exp_alpha,
        auto_select_cc_window=args.auto_select_cc_window,
        cc_scan_step=args.cc_scan_step,
    )

    metrics = run_one_file(path=args.file, outdir=args.outdir, cfg=cfg)
    for k, v in metrics.items():
        print(f"{k}={v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
