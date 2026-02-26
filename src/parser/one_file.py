"""One-file reflectogram parsing pipeline with diagnostic plots."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re

# Writable matplotlib cache in restricted environments.
_cache_root = (Path("data") / "interim" / ".cache").resolve()
_mpl_root = (Path("data") / "interim" / ".mplconfig").resolve()
_cache_root.mkdir(parents=True, exist_ok=True)
_mpl_root.mkdir(parents=True, exist_ok=True)
os.environ["XDG_CACHE_HOME"] = str(_cache_root)
os.environ["MPLCONFIGDIR"] = str(_mpl_root)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

# Internal tuning constants (not exposed via CLI).
_DET_MIN_GAP_FACTOR = 0.82
_DET_LOW_LEVEL_FACTOR = 0.10
_DET_RISE_FACTOR = 0.075
_DET_DECIMATION = 10
_DET_MA_WINDOW = 31
_DET_ENVELOPE_ALPHA = 0.995
_DET_REFINE_RADIUS = 300
_DET_FILL_MISSING = True
_DET_RECOVERY_MIN_SPACING_FACTOR = 0.82
_DET_RECOVERY_CORR_DECIMATION = 8
_DET_PERIOD_MIN = 20_000
_DET_PERIOD_MAX = 260_000
_DET_TEMPLATE_REFINE_RADIUS = 450
_DET_TEMPLATE_REFINE_LEN = 3000
_DET_TEMPLATE_REFINE_TRACES = 48


@dataclass(frozen=True)
class ParseConfig:
    max_traces: int = 500
    max_shift: int = 300
    cc_window_start: int = 500
    cc_window_len: int = 12_000
    raw_plot_points: int = 1_000_000
    adc_fs_hz: float = 50_000_000.0
    align_iters: int = 1
    align_decimation: int = 4
    max_samples: int | None = 12_000_000
    waterfall_cmap: str = "jet"
    waterfall_exp_alpha: float = 4.0
    auto_select_cc_window: bool = True
    cc_scan_step: int = 2000


def _sanitize_tag(text: str, max_len: int = 80) -> str:
    """Build filesystem-friendly tag chunk from free text."""

    cleaned = re.sub(r"[^\w.-]+", "_", text.strip(), flags=re.UNICODE).strip("_.")
    if not cleaned:
        cleaned = "unknown"
    return cleaned[:max_len]


def _build_source_tag(path: Path) -> str:
    """Stable short tag that encodes source file identity for output names."""

    parent = _sanitize_tag(path.parent.name)
    stem = _sanitize_tag(path.stem)
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    return f"{parent}__{stem}__{digest}"


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


def _collect_start_candidates(
    data: np.ndarray,
    *,
    low_level_factor: float = _DET_LOW_LEVEL_FACTOR,
    rise_factor: float = _DET_RISE_FACTOR,
    decimation: int = _DET_DECIMATION,
    ma_window: int = _DET_MA_WINDOW,
    envelope_alpha: float = _DET_ENVELOPE_ALPHA,
    refine_radius: int = _DET_REFINE_RADIUS,
) -> np.ndarray:
    """Return dense start candidates before period-based spacing filter."""

    dec = max(1, int(decimation))
    dec_data = data[::dec]
    sm = _moving_average(dec_data, ma_window)
    _, _, mid = _envelopes(sm, alpha=envelope_alpha)

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


def _estimate_trace_len_from_autocorr(data: np.ndarray) -> int:
    """Estimate reflectogram period from signal autocorrelation on decimated stream."""

    dec = max(1, int(_DET_DECIMATION))
    y = data[::dec]
    if len(y) < 2048:
        return 55_000

    s = _moving_average(y, _DET_MA_WINDOW)
    s = s - float(np.median(s))
    n = len(s)

    size = 1 << int(np.ceil(np.log2(2 * n - 1)))
    sf = np.fft.rfft(s, size)
    ac = np.fft.irfft(sf * np.conj(sf), size)[:n]
    ac[0] = 0.0

    min_lag = max(8, int(_DET_PERIOD_MIN // dec))
    max_lag = min(n - 2, int(_DET_PERIOD_MAX // dec))
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


def _infer_trace_len(data: np.ndarray, candidates: np.ndarray) -> int:
    """Infer trace length without external value, blending autocorr and candidate-gap estimates."""

    p_ac = _estimate_trace_len_from_autocorr(data)
    p_cand = _estimate_trace_len_from_candidates(candidates, fallback=p_ac)
    ratio = float(p_cand) / float(max(1, p_ac))
    if 0.60 <= ratio <= 1.60:
        p = int(round(0.75 * p_ac + 0.25 * p_cand))
    else:
        p = p_ac
    return int(np.clip(p, _DET_PERIOD_MIN, _DET_PERIOD_MAX))


def _refine_starts_with_template(data: np.ndarray, starts: np.ndarray, trace_len: int) -> np.ndarray:
    """Refine starts by local template matching on gradient signal."""

    if len(starts) < 3:
        return starts

    n = len(data)
    ds = max(1, int(_DET_RECOVERY_CORR_DECIMATION))
    w = max(256, min(int(_DET_TEMPLATE_REFINE_LEN), max(512, trace_len // 3)))
    r = max(80, int(_DET_TEMPLATE_REFINE_RADIUS))

    sm = _moving_average(data, 7)
    grad = np.zeros_like(sm)
    grad[1:-1] = (sm[2:] - sm[:-2]) * 0.5
    grad[0] = grad[1]
    grad[-1] = grad[-2]

    valid = starts[(starts >= 1) & (starts + w + 2 < n)]
    if len(valid) < 3:
        return starts
    m = min(len(valid), int(_DET_TEMPLATE_REFINE_TRACES))
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
    min_spacing = max(1, int(_DET_RECOVERY_MIN_SPACING_FACTOR * trace_len))
    packed: list[int] = [int(arr[0])]
    for s in arr[1:]:
        if int(s) - packed[-1] >= min_spacing:
            packed.append(int(s))
    out = np.array(packed, dtype=np.int64)
    return out[out + trace_len <= n]


def _detect_starts(
    data: np.ndarray,
    trace_len: int,
    *,
    low_level_factor: float = _DET_LOW_LEVEL_FACTOR,
    rise_factor: float = _DET_RISE_FACTOR,
    min_gap_factor: float = _DET_MIN_GAP_FACTOR,
    decimation: int = _DET_DECIMATION,
) -> np.ndarray:
    starts = _collect_start_candidates(
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
    if _DET_FILL_MISSING:
        starts = _recover_starts_periodic(data=data, anchors=starts, trace_len=trace_len)
    return _refine_starts_with_template(data=data, starts=starts, trace_len=trace_len)


def _recover_starts_periodic(data: np.ndarray, anchors: np.ndarray, trace_len: int) -> np.ndarray:
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
    sm = _moving_average(data, max(7, _DET_MA_WINDOW // 2))
    grad = np.zeros_like(sm)
    grad[1:-1] = (sm[2:] - sm[:-2]) * 0.5
    grad[0] = grad[1]
    grad[-1] = grad[-2]

    refine_r = max(40, min(200, int(_DET_REFINE_RADIUS)))
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

    min_spacing = max(1, int(_DET_RECOVERY_MIN_SPACING_FACTOR * period))
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
    """Estimate lag between `ref` and `cur`.

    Positive return value means `cur` should be shifted to the right by that lag
    to match `ref` (i.e. alignment shift is `-lag` for `_shift_trace`).
    """

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

    dec = max(1, int(cfg.align_decimation))
    max_shift_dec = max(1, int(round(cfg.max_shift / dec)))

    aligned = traces.copy()
    shifts_total = np.zeros(traces.shape[0], dtype=np.int64)
    n_ref = min(256, traces.shape[0])
    n_iters = max(1, int(cfg.align_iters))

    for _ in range(n_iters):
        # Robust template: median over early traces in selected window.
        template = np.median(aligned[:n_ref, start:end], axis=0)
        if dec > 1:
            template_d = template[::dec]
        else:
            template_d = template

        iter_shifts = np.zeros(traces.shape[0], dtype=np.int64)
        for i in range(traces.shape[0]):
            cur = aligned[i, start:end]
            cur_d = cur[::dec] if dec > 1 else cur
            lag = _best_shift_fft(template_d, cur_d, max_shift=max_shift_dec) * dec
            lag = int(np.clip(lag, -cfg.max_shift, cfg.max_shift))
            iter_shifts[i] = lag
            # Important: apply opposite sign of lag to actually align.
            aligned[i] = _shift_trace(aligned[i], -lag)

        shifts_total += iter_shifts
        if float(np.mean(np.abs(iter_shifts))) < 0.25:
            break

    return aligned, shifts_total


def _estimate_residual_jitter(traces: np.ndarray, cfg: ParseConfig) -> tuple[float, float, float]:
    start = cfg.cc_window_start
    end = min(traces.shape[1], start + cfg.cc_window_len)
    if end - start < 256:
        return 0.0, 0.0, 0.0

    dec = max(1, int(cfg.align_decimation))
    max_shift_dec = max(1, int(round(cfg.max_shift / dec)))
    n_ref = min(256, traces.shape[0])
    ref = np.median(traces[:n_ref, start:end], axis=0)
    ref_d = ref[::dec] if dec > 1 else ref

    shifts: list[int] = []
    for i in range(traces.shape[0]):
        cur = traces[i, start:end]
        cur_d = cur[::dec] if dec > 1 else cur
        s = _best_shift_fft(ref_d, cur_d, max_shift=max_shift_dec) * dec
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


def _extract_with_fallbacks(data: np.ndarray, cfg: ParseConfig, trace_len: int) -> tuple[ParseConfig, np.ndarray, np.ndarray]:
    """Try primary config and a few nearby configs to avoid empty extraction."""

    detector_variants = [
        (_DET_LOW_LEVEL_FACTOR, _DET_RISE_FACTOR, _DET_MIN_GAP_FACTOR, _DET_DECIMATION),
        (0.09, 0.07, 0.78, 8),
        (0.10, 0.09, 0.82, 10),
        (0.12, 0.11, 0.90, 10),
        (0.10, 0.07, 1.00, 8),
    ]

    seen = set()
    best: tuple[int, np.ndarray, np.ndarray] | None = None
    for low, rise, gap, dec in detector_variants:
        key = (low, rise, gap, dec)
        if key in seen:
            continue
        seen.add(key)
        starts = _detect_starts(
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
            traces = _extract_reflectograms(data, starts, trace_len=trace_len, max_traces=cfg.max_traces)
        except Exception:
            continue
        n = traces.shape[0]
        if best is None or n > best[0]:
            best = (n, starts, traces)
            if n >= cfg.max_traces:
                break

    if best is None:
        raise ValueError("No complete traces extracted; tune parser thresholds")
    return cfg, best[1], best[2]


def run_one_file(path: Path, outdir: Path, cfg: ParseConfig) -> dict[str, float | int | str]:
    data = _read_data_stream(path, max_points=cfg.max_samples)
    source_tag = _build_source_tag(path)
    raw_candidates = _collect_start_candidates(data)
    trace_len = _infer_trace_len(data, raw_candidates)
    cfg_work = cfg

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

    raw_segment_name = f"{source_tag}__raw_segment.png"
    waterfall_before_name = f"{source_tag}__waterfall_before.png"
    waterfall_after_name = f"{source_tag}__waterfall_after.png"

    _plot_raw_segment(
        data,
        starts,
        outdir / raw_segment_name,
        points=cfg_work.raw_plot_points,
        fs_hz=cfg_work.adc_fs_hz,
    )
    _plot_waterfall(
        traces,
        outdir / waterfall_before_name,
        "Waterfall before alignment",
        fs_hz=cfg_work.adc_fs_hz,
        cmap=cfg_work.waterfall_cmap,
        exp_alpha=cfg_work.waterfall_exp_alpha,
    )
    _plot_waterfall(
        aligned,
        outdir / waterfall_after_name,
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
        "source_tag": source_tag,
        "n_samples": int(len(data)),
        "n_detected_starts": int(len(starts)),
        "n_extracted_traces": int(traces.shape[0]),
        "first_start": int(starts[0]) if len(starts) else -1,
        "last_start": int(starts[-1]) if len(starts) else -1,
        "trace_len": int(traces.shape[1]),
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
        f"- {raw_segment_name}",
        f"- {waterfall_before_name}",
        f"- {waterfall_after_name}",
    ]
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "parser_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (outdir / "parser_params.json").write_text(
        json.dumps(
            {
                "trace_len_used": trace_len,
                "max_traces": cfg_work.max_traces,
                "max_samples": cfg_work.max_samples,
                "adc_fs_hz": cfg_work.adc_fs_hz,
                "max_shift": cfg_work.max_shift,
                "align_iters": cfg_work.align_iters,
                "align_decimation": cfg_work.align_decimation,
                "waterfall_cmap": cfg_work.waterfall_cmap,
                "waterfall_exp_alpha": cfg_work.waterfall_exp_alpha,
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
    parser = argparse.ArgumentParser(
        prog="python -m src.parser.one_file",
        description="Run reflectogram parser on one file (auto mode).",
    )
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("reports/figures/parser"))
    parser.add_argument("--max-samples", type=int, default=50_000_000)
    parser.add_argument("--max-traces", type=int, default=2_000)
    parser.add_argument("--adc-fs-hz", type=float, default=50_000_000.0)
    parser.add_argument("--raw-plot-points", type=int, default=1_000_000)
    parser.add_argument("--waterfall-cmap", type=str, default="jet")
    parser.add_argument("--waterfall-exp-alpha", type=float, default=4.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = ParseConfig(
        max_traces=args.max_traces,
        raw_plot_points=args.raw_plot_points,
        adc_fs_hz=args.adc_fs_hz,
        max_samples=args.max_samples,
        waterfall_cmap=args.waterfall_cmap,
        waterfall_exp_alpha=args.waterfall_exp_alpha,
    )

    metrics = run_one_file(path=args.file, outdir=args.outdir, cfg=cfg)
    for k, v in metrics.items():
        print(f"{k}={v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
