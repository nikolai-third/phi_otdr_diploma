"""Post-alignment filtering and frequency-agnostic disturbance detection.

This module treats detection as a continuation of the reflectogram alignment stage:
- input: aligned matrix from parser cache (aligned.npz)
- output: step-by-step diagnostics and detected disturbance locations.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

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


def _mad(x: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray | float:
    med = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - med), axis=axis, keepdims=True)
    if axis is None and not keepdims:
        return float(np.asarray(mad).reshape(()))
    if not keepdims:
        mad = np.squeeze(mad, axis=axis)
    return mad


def _group_by_distance(x: np.ndarray, group_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Group [time, distance] matrix into [time, groups] using local averaging."""

    group_bins = max(1, int(group_bins))
    n_dist = x.shape[1]
    n_groups = n_dist // group_bins
    if n_groups < 4:
        raise ValueError(f"Too few groups ({n_groups}) for group_bins={group_bins}")
    trim = x[:, : n_groups * group_bins]
    grouped = trim.reshape(trim.shape[0], n_groups, group_bins).mean(axis=2)
    return grouped, np.arange(n_groups, dtype=np.int64)


def _sample_rows(x: np.ndarray, starts: np.ndarray, max_traces: int | None) -> tuple[np.ndarray, np.ndarray]:
    if max_traces is None or max_traces <= 0 or x.shape[0] <= int(max_traces):
        return x, starts
    idx = np.linspace(0, x.shape[0] - 1, num=int(max_traces), dtype=np.int64)
    return np.asarray(x[idx], dtype=np.float32, copy=False), np.asarray(starts[idx], dtype=np.int64, copy=False)


def _plot_waterfall(
    arr: np.ndarray,
    out: Path,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    xlabel: str,
    ylabel: str,
    title: str,
    cbar_label: str,
    cmap: str = "jet",
    q_lo: float = 0.01,
    q_hi: float = 0.99,
) -> None:
    vmin, vmax = np.quantile(arr, [q_lo, q_hi])
    fig, ax = plt.subplots(figsize=(12, 5.8))
    im = ax.imshow(
        arr,
        origin="lower",
        aspect="auto",
        extent=[x_min, x_max, y_min, y_max],
        cmap=cmap,
        vmin=float(vmin),
        vmax=float(vmax),
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(cbar_label)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_waterfall_signed(
    arr: np.ndarray,
    out: Path,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    xlabel: str,
    ylabel: str,
    title: str,
    cbar_label: str,
    cmap: str = "RdBu_r",
    q_abs: float = 0.995,
) -> None:
    vmax = float(np.quantile(np.abs(arr), q_abs))
    vmax = max(vmax, 1e-12)
    fig, ax = plt.subplots(figsize=(12, 5.8))
    im = ax.imshow(
        arr,
        origin="lower",
        aspect="auto",
        extent=[x_min, x_max, y_min, y_max],
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(cbar_label)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_top_usable_zoom(
    aligned: np.ndarray,
    dist_km: np.ndarray,
    time_s: np.ndarray,
    best_km: float,
    out: Path,
    window_km: float = 3.0,
) -> None:
    left_km = float(max(dist_km[0], best_km - window_km))
    right_km = float(min(dist_km[-1], best_km + window_km))
    i0 = int(np.searchsorted(dist_km, left_km, side="left"))
    i1 = int(np.searchsorted(dist_km, right_km, side="right"))
    i0 = max(0, min(i0, len(dist_km) - 1))
    i1 = max(i0 + 1, min(i1, len(dist_km)))

    zone = np.asarray(aligned[:, i0:i1], dtype=np.float32)
    vmin, vmax = np.quantile(zone, [0.01, 0.99])
    fig, ax = plt.subplots(figsize=(11, 5.2))
    im = ax.imshow(
        zone,
        origin="lower",
        aspect="auto",
        extent=[float(dist_km[i0]), float(dist_km[i1 - 1]), float(time_s[0]), float(time_s[-1])],
        cmap="jet",
        vmin=float(vmin),
        vmax=float(vmax),
    )
    ax.axvline(float(best_km), color="white", linestyle="--", linewidth=1.4)
    ax.set_xlabel("Distance, km")
    ax.set_ylabel("Time, s")
    ax.set_title(f"Most suspicious usable-zone point: {best_km:.3f} km (raw, no normalization)")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Raw amplitude, a.u.")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _distance_axis_km(trace_len: int, adc_fs_hz: float, n_fiber: float = 1.468) -> np.ndarray:
    c0 = 299_792_458.0
    return (np.arange(trace_len, dtype=np.float64) * (c0 / (2.0 * n_fiber * adc_fs_hz))) / 1000.0


def _time_axis_s(starts: np.ndarray, n_traces: int, trace_len: int, adc_fs_hz: float) -> np.ndarray:
    if len(starts) >= n_traces:
        return (starts[:n_traces] - starts[0]).astype(np.float64) / adc_fs_hz
    if len(starts) > 1:
        dt = float(np.median(np.diff(starts))) / adc_fs_hz
    else:
        dt = float(trace_len / adc_fs_hz)
    return np.arange(n_traces, dtype=np.float64) * dt


def _pick_peaks(score: np.ndarray, dist_km: np.ndarray, threshold: float, min_sep_km: float) -> list[int]:
    """Greedy peak picking with minimum distance separation."""

    cand = np.where(score >= threshold)[0]
    if len(cand) == 0:
        return []

    order = cand[np.argsort(score[cand])[::-1]]
    picked: list[int] = []
    for idx in order:
        d = dist_km[idx]
        if all(abs(d - dist_km[p]) >= min_sep_km for p in picked):
            picked.append(int(idx))
    picked.sort()
    return picked


def _limit_by_score(picked: list[int], score: np.ndarray, max_detections: int) -> list[int]:
    if len(picked) <= max_detections:
        return picked
    order = sorted(picked, key=lambda i: float(score[i]), reverse=True)
    keep = sorted(order[:max_detections])
    return keep


def _moving_average_1d(x: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window <= 1:
        return x.copy()
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(x, kernel, mode="same")


def _select_stable_trace_segment(
    aligned: np.ndarray,
    dist_km: np.ndarray,
    corr_threshold: float = 0.95,
    min_len: int = 400,
) -> tuple[int, int, dict[str, float]]:
    """Select longest high-coherence time segment by adjacent-trace correlation."""

    n_traces = int(aligned.shape[0])
    if n_traces < max(8, min_len):
        return 0, n_traces, {
            "corr_median": float("nan"),
            "corr_q10": float("nan"),
            "corr_q90": float("nan"),
            "segment_corr_median": float("nan"),
            "segment_corr_q10": float("nan"),
            "segment_corr_q90": float("nan"),
        }

    d = np.asarray(dist_km, dtype=np.float64)
    lo = float(np.quantile(d, 0.08))
    hi = float(np.quantile(d, 0.62))
    mask = (d >= lo) & (d <= hi)
    if int(np.sum(mask)) < 512:
        mask = np.ones_like(d, dtype=bool)

    x = np.asarray(aligned[:, mask], dtype=np.float32)
    x0 = x - np.mean(x, axis=1, keepdims=True)
    nrm = np.linalg.norm(x0, axis=1, keepdims=True) + 1e-9
    x0 = x0 / nrm
    corr = np.sum(x0[1:] * x0[:-1], axis=1)

    stats: dict[str, float] = {
        "corr_median": float(np.median(corr)),
        "corr_q10": float(np.quantile(corr, 0.10)),
        "corr_q90": float(np.quantile(corr, 0.90)),
        "segment_corr_median": float(np.median(corr)),
        "segment_corr_q10": float(np.quantile(corr, 0.10)),
        "segment_corr_q90": float(np.quantile(corr, 0.90)),
    }

    # Use raw adjacent correlation to preserve true continuity boundaries.
    # Small gap-closing prevents one-off drops from splitting one stable block.
    good = corr >= float(corr_threshold)
    if len(good) >= 5:
        max_gap = 2
        i = 0
        while i < len(good):
            if good[i]:
                i += 1
                continue
            j = i + 1
            while j < len(good) and not good[j]:
                j += 1
            gap_len = j - i
            left_ok = i > 0 and good[i - 1]
            right_ok = j < len(good) and good[j]
            if left_ok and right_ok and gap_len <= max_gap:
                good[i:j] = True
            i = j

    best_i, best_j, best_len = 0, n_traces, 0
    i = 0
    while i < len(good):
        if not good[i]:
            i += 1
            continue
        j = i + 1
        while j < len(good) and good[j]:
            j += 1
        run_len = j - i + 1  # corr[i:j] maps to traces [i, j]
        if run_len > best_len:
            best_len = run_len
            best_i, best_j = i, j + 1
        i = j

    if best_len < int(min_len):
        return 0, n_traces, stats
    seg_corr = corr[best_i : max(best_i + 1, best_j - 1)]
    if len(seg_corr):
        stats["segment_corr_median"] = float(np.median(seg_corr))
        stats["segment_corr_q10"] = float(np.quantile(seg_corr, 0.10))
        stats["segment_corr_q90"] = float(np.quantile(seg_corr, 0.90))
    return int(best_i), int(best_j), stats


def _estimate_signal_end_km(background: np.ndarray, dist_km: np.ndarray) -> float:
    """Estimate end of useful reflectogram by transition to stable low-level tail."""

    b = np.asarray(background, dtype=np.float64)
    d = np.asarray(dist_km, dtype=np.float64)
    if len(b) < 64:
        return float(d[-1])

    n = len(b)
    smooth_w = max(31, (n // 200) | 1)
    bs = _moving_average_1d(b, smooth_w)
    tail = bs[int(0.85 * n) :]
    tail_level = float(np.median(tail))
    high_level = float(np.quantile(bs, 0.95))
    thr = tail_level + 0.18 * max(1e-9, high_level - tail_level)

    # First location where profile enters low-level regime and stays there.
    run = max(128, n // 35)
    low = bs <= thr
    cnt = np.convolve(low.astype(np.int32), np.ones(run, dtype=np.int32), mode="same")
    stable_low = cnt >= int(0.95 * run)
    grad = np.diff(bs, prepend=bs[0])
    cand = np.where(stable_low & low & (grad < 0.0))[0]
    cand = cand[cand >= int(0.20 * n)]
    if len(cand):
        idx = int(cand[0])
        return float(d[max(0, min(idx, len(d) - 1))])

    # Fallback: last strong-above-tail sample.
    idx2 = np.where(bs > thr)[0]
    if len(idx2):
        return float(d[int(idx2[-1])])
    return float(d[-1])


def run_detection(
    aligned_npz: Path,
    outdir: Path,
    group_bins: int,
    freq_min_hz: float,
    freq_max_hz: float,
    min_sep_km: float,
    threshold_k: float,
    peak_threshold_k: float,
    ignore_start_km: float,
    max_detections: int,
    expected_km: list[float] | None,
    max_traces: int | None,
    end_guard_km: float,
    stable_corr_threshold: float,
    stable_min_traces: int,
    use_stable_segment: bool,
) -> dict[str, Any]:
    data = np.load(aligned_npz)
    aligned = np.asarray(data["aligned"], dtype=np.float32)
    starts = np.asarray(data["starts"], dtype=np.int64)
    trace_len = int(data["trace_len"])
    adc_fs_hz = float(data["adc_fs_hz"])
    aligned, starts = _sample_rows(aligned, starts, max_traces=max_traces)

    n_traces, n_bins = aligned.shape
    dist_km = _distance_axis_km(trace_len=n_bins, adc_fs_hz=adc_fs_hz)
    seg_i0_raw, seg_i1_raw, corr_stats = _select_stable_trace_segment(
        aligned=aligned,
        dist_km=dist_km,
        corr_threshold=stable_corr_threshold,
        min_len=stable_min_traces,
    )
    if use_stable_segment:
        seg_i0 = int(seg_i0_raw)
        seg_i1 = int(seg_i1_raw)
    else:
        seg_i0 = 0
        seg_i1 = int(aligned.shape[0])
    aligned = aligned[seg_i0:seg_i1]
    starts = starts[seg_i0:seg_i1]
    n_traces, n_bins = aligned.shape
    time_s = _time_axis_s(starts=starts, n_traces=n_traces, trace_len=trace_len, adc_fs_hz=adc_fs_hz)

    # Step 0. aligned signal overview (raw)
    _plot_waterfall(
        arr=aligned,
        out=outdir / "step0_aligned_waterfall_raw.png",
        x_min=float(dist_km[0]),
        x_max=float(dist_km[-1]),
        y_min=float(time_s[0]),
        y_max=float(time_s[-1]),
        xlabel="Distance, km",
        ylabel="Time, s",
        title="Step 0: Aligned reflectograms (raw)",
        cbar_label="Amplitude, a.u.",
        q_lo=0.01,
        q_hi=0.99,
    )

    # Step 1. robust background and residual
    background = np.median(aligned, axis=0)
    residual = aligned - background[None, :]
    signal_end_km = _estimate_signal_end_km(background=background, dist_km=dist_km)
    usable_end_km = max(float(dist_km[0]), signal_end_km - float(max(0.0, end_guard_km)))

    fig1, ax1 = plt.subplots(figsize=(12, 3.6))
    ax1.plot(dist_km, background, linewidth=0.9)
    ax1.set_xlabel("Distance, km")
    ax1.set_ylabel("Amplitude, a.u.")
    ax1.set_title("Step 1: Robust background profile (median over time)")
    ax1.grid(alpha=0.25)
    fig1.savefig(outdir / "step1_background_profile.png", dpi=200, bbox_inches="tight")
    plt.close(fig1)

    _plot_waterfall(
        arr=residual,
        out=outdir / "step1b_residual_waterfall.png",
        x_min=float(dist_km[0]),
        x_max=float(dist_km[-1]),
        y_min=float(time_s[0]),
        y_max=float(time_s[-1]),
        xlabel="Distance, km",
        ylabel="Time, s",
        title="Step 1b: Residual after background subtraction",
        cbar_label="Residual, a.u.",
        q_lo=0.01,
        q_hi=0.99,
    )
    # Display-only contrast boost for human readability.
    res_ref = float(np.quantile(np.abs(residual), 0.80)) + 1e-12
    residual_disp = np.arcsinh(residual / res_ref)
    _plot_waterfall_signed(
        arr=residual_disp,
        out=outdir / "step1c_residual_waterfall_display.png",
        x_min=float(dist_km[0]),
        x_max=float(dist_km[-1]),
        y_min=float(time_s[0]),
        y_max=float(time_s[-1]),
        xlabel="Distance, km",
        ylabel="Time, s",
        title="Step 1c: Residual (display-contrast, full range)",
        cbar_label="asinh(residual / q80_abs)",
        cmap="RdBu_r",
        q_abs=0.995,
    )

    # Step 2. robust normalization per distance bin
    scale = np.asarray(_mad(residual, axis=0), dtype=np.float32) + np.float32(1e-9)
    z = residual / scale[None, :]

    fig2, ax2 = plt.subplots(figsize=(12, 3.6))
    ax2.plot(dist_km, scale, linewidth=0.9)
    ax2.set_xlabel("Distance, km")
    ax2.set_ylabel("MAD scale")
    ax2.set_title("Step 2: Robust scale (MAD) per distance")
    ax2.grid(alpha=0.25)
    fig2.savefig(outdir / "step2_mad_profile.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)

    _plot_waterfall(
        arr=z,
        out=outdir / "step2b_normalized_waterfall_z.png",
        x_min=float(dist_km[0]),
        x_max=float(dist_km[-1]),
        y_min=float(time_s[0]),
        y_max=float(time_s[-1]),
        xlabel="Distance, km",
        ylabel="Time, s",
        title="Step 2b: Normalized residual (z-score)",
        cbar_label="z-score",
        q_lo=0.02,
        q_hi=0.98,
    )
    z_disp = np.arcsinh(z / 0.75)
    _plot_waterfall_signed(
        arr=z_disp,
        out=outdir / "step2c_normalized_waterfall_display.png",
        x_min=float(dist_km[0]),
        x_max=float(dist_km[-1]),
        y_min=float(time_s[0]),
        y_max=float(time_s[-1]),
        xlabel="Distance, km",
        ylabel="Time, s",
        title="Step 2c: Normalized residual (display-contrast, full range)",
        cbar_label="asinh(z / 0.75)",
        cmap="RdBu_r",
        q_abs=0.995,
    )

    # Step 3. full-range frequency analysis on grouped signal
    grouped, g_idx = _group_by_distance(z, group_bins=group_bins)
    dist_group = dist_km[g_idx * group_bins]

    dt_trace = float(np.median(np.diff(time_s))) if len(time_s) > 1 else float(trace_len / adc_fs_hz)
    dt_trace = max(dt_trace, 1e-9)

    spec = np.fft.rfft(grouped, axis=0)
    freq = np.fft.rfftfreq(grouped.shape[0], d=dt_trace)
    power_db_raw = 10.0 * np.log10((np.abs(spec) ** 2) / max(1, grouped.shape[0]) + 1e-24)

    fmask = (freq >= float(freq_min_hz)) & (freq <= float(min(freq_max_hz, freq[-1])))
    if not np.any(fmask):
        raise ValueError("No frequencies left after masking; check freq_min_hz/freq_max_hz")

    f_view = freq[fmask]
    p_view = power_db_raw[fmask, :]

    _plot_waterfall(
        arr=p_view,
        out=outdir / "step3_fft_full_range_raw_db.png",
        x_min=float(dist_group[0]),
        x_max=float(dist_group[-1]),
        y_min=float(f_view[0]),
        y_max=float(f_view[-1]),
        xlabel="Distance, km",
        ylabel="Frequency, Hz",
        title="Step 3: Full-range FFT map (raw dB)",
        cbar_label="Power, dB (raw)",
        q_lo=0.01,
        q_hi=0.995,
    )
    # Display-only contrast map: highlight deviations from frequency-wise median.
    p_view_contrast = p_view - np.median(p_view, axis=1, keepdims=True)
    _plot_waterfall_signed(
        arr=p_view_contrast,
        out=outdir / "step3b_fft_full_range_contrast.png",
        x_min=float(dist_group[0]),
        x_max=float(dist_group[-1]),
        y_min=float(f_view[0]),
        y_max=float(f_view[-1]),
        xlabel="Distance, km",
        ylabel="Frequency, Hz",
        title="Step 3b: FFT map (display-contrast, full range)",
        cbar_label="dB relative to per-frequency median",
        cmap="RdBu_r",
        q_abs=0.995,
    )

    # Step 4. frequency-agnostic anomaly score across whole band
    # Baseline across distance for each frequency row.
    row_med = np.median(p_view, axis=1, keepdims=True)
    row_mad = np.asarray(_mad(p_view, axis=1, keepdims=True), dtype=np.float32) + np.float32(1e-9)
    zf = (p_view - row_med) / row_mad

    peak_score = np.max(zf, axis=0)
    broad_score = np.mean(np.clip(zf - 1.0, 0.0, None), axis=0)

    energy_rms = np.sqrt(np.mean(grouped**2, axis=0))
    energy_z = (energy_rms - np.median(energy_rms)) / (float(_mad(energy_rms)) + 1e-9)

    combined = 0.55 * peak_score + 0.30 * broad_score + 0.15 * energy_z

    combined_med = float(np.median(combined))
    combined_mad = float(_mad(combined))
    peak_med = float(np.median(peak_score))
    peak_mad = float(_mad(peak_score))

    thr = combined_med + threshold_k * combined_mad
    peak_thr = peak_med + peak_threshold_k * peak_mad

    candidate_score = combined.copy()
    candidate_mask = (
        (peak_score >= peak_thr)
        & (dist_group >= float(ignore_start_km))
        & (dist_group <= float(usable_end_km))
    )
    candidate_score[~candidate_mask] = -np.inf

    picked = _pick_peaks(score=candidate_score, dist_km=dist_group, threshold=thr, min_sep_km=min_sep_km)
    picked = _limit_by_score(picked, score=combined, max_detections=max_detections)

    usable_mask = (
        np.isfinite(combined)
        & (dist_group >= float(ignore_start_km))
        & (dist_group <= float(usable_end_km))
    )
    top_usable: dict[str, float] | None = None
    if np.any(usable_mask):
        usable_idx = np.where(usable_mask)[0]
        best_local = int(np.argmax(combined[usable_idx]))
        best_idx = int(usable_idx[best_local])
        best_km = float(dist_group[best_idx])
        top_usable = {
            "best_distance_km_within_usable": best_km,
            "best_combined_score_within_usable": float(combined[best_idx]),
            "best_peak_spectral_z_within_usable": float(peak_score[best_idx]),
            "best_broad_score_within_usable": float(broad_score[best_idx]),
            "best_energy_z_within_usable": float(energy_z[best_idx]),
            "usable_end_km": float(usable_end_km),
            "ignore_start_km": float(ignore_start_km),
        }
        (outdir / "top_candidate_within_usable.json").write_text(
            json.dumps(top_usable, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _save_top_usable_zoom(
            aligned=aligned,
            dist_km=dist_km,
            time_s=time_s,
            best_km=best_km,
            out=outdir / "suspicious_zone_raw_pm3km_top_usable.png",
            window_km=3.0,
        )

    # Step 5. score diagnostics
    fig3, ax3 = plt.subplots(figsize=(12, 4.0))
    ax3.plot(dist_group, peak_score, label="peak spectral z", linewidth=1.0)
    ax3.plot(dist_group, broad_score, label="broadband spectral score", linewidth=1.0)
    ax3.plot(dist_group, energy_z, label="time-domain energy z", linewidth=1.0, alpha=0.8)
    ax3.plot(dist_group, combined, label="combined score", linewidth=1.5, color="black")
    ax3.axhline(thr, color="red", linestyle="--", linewidth=1.2, label=f"combined threshold={thr:.2f}")
    ax3.axhline(peak_thr, color="orange", linestyle="--", linewidth=1.2, label=f"peak threshold={peak_thr:.2f}")
    for i in picked:
        ax3.axvline(float(dist_group[i]), color="magenta", alpha=0.6, linewidth=1.0)
    ax3.axvline(float(usable_end_km), color="gray", linestyle="--", linewidth=1.1, alpha=0.8)
    if expected_km:
        for d in expected_km:
            ax3.axvline(float(d), color="green", linestyle=":", linewidth=1.2)
    ax3.set_xlabel("Distance, km")
    ax3.set_ylabel("Score")
    ax3.set_title("Step 4: Frequency-agnostic detection scores")
    ax3.grid(alpha=0.25)
    ax3.legend(loc="upper right", ncol=2, fontsize=9)
    fig3.savefig(outdir / "step4_detection_scores.png", dpi=200, bbox_inches="tight")
    plt.close(fig3)

    # Step 6. overlay detections on normalized waterfall
    fig4, ax4 = plt.subplots(figsize=(12, 5.8))
    vz_min, vz_max = np.quantile(z, [0.02, 0.98])
    im4 = ax4.imshow(
        z,
        origin="lower",
        aspect="auto",
        extent=[float(dist_km[0]), float(dist_km[-1]), float(time_s[0]), float(time_s[-1])],
        cmap="jet",
        vmin=float(vz_min),
        vmax=float(vz_max),
    )
    for i in picked:
        ax4.axvline(float(dist_group[i]), color="white", linestyle="--", linewidth=1.2)
    ax4.axvline(float(usable_end_km), color="gray", linestyle="--", linewidth=1.1, alpha=0.8)
    if expected_km:
        for d in expected_km:
            ax4.axvline(float(d), color="lime", linestyle=":", linewidth=1.2)
    ax4.set_xlabel("Distance, km")
    ax4.set_ylabel("Time, s")
    ax4.set_title("Step 5: Detections over normalized waterfall")
    cb4 = fig4.colorbar(im4, ax=ax4)
    cb4.set_label("z-score")
    fig4.savefig(outdir / "step5_detections_overlay.png", dpi=200, bbox_inches="tight")
    plt.close(fig4)

    # Step 5b. Clean final figure without noisy 2D background.
    smooth_win = max(5, (len(combined) // 250) | 1)  # odd window, full-range friendly
    combined_smooth = _moving_average_1d(combined, smooth_win)
    fig5, ax5 = plt.subplots(figsize=(12, 4.0))
    ax5.plot(dist_group, combined_smooth, color="black", linewidth=1.6, label="combined score (smoothed)")
    ax5.axhline(thr, color="red", linestyle="--", linewidth=1.2, label=f"combined threshold={thr:.2f}")
    for i in picked:
        ax5.scatter(
            [float(dist_group[i])],
            [float(combined_smooth[i])],
            s=70,
            marker="o",
            color="magenta",
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        ax5.axvline(float(dist_group[i]), color="magenta", alpha=0.25, linewidth=1.0)
    ax5.axvline(float(usable_end_km), color="gray", linestyle="--", linewidth=1.1, alpha=0.8)
    if expected_km:
        for d in expected_km:
            ax5.axvline(float(d), color="green", linestyle=":", linewidth=1.2)
    ax5.set_xlabel("Distance, km")
    ax5.set_ylabel("Combined score")
    ax5.set_title("Step 5b: Final detection graph (clean)")
    ax5.grid(alpha=0.25)
    ax5.legend(loc="upper right")
    fig5.savefig(outdir / "step5b_final_detection_graph.png", dpi=220, bbox_inches="tight")
    plt.close(fig5)

    detected_rows: list[dict[str, float]] = []
    for i in picked:
        detected_rows.append(
            {
                "distance_km": float(dist_group[i]),
                "combined_score": float(combined[i]),
                "peak_spectral_z": float(peak_score[i]),
                "broad_score": float(broad_score[i]),
                "energy_z": float(energy_z[i]),
            }
        )

    res: dict[str, Any] = {
        "aligned_npz": str(aligned_npz),
        "n_traces": int(n_traces),
        "max_traces": int(max_traces) if max_traces is not None else None,
        "use_stable_segment": bool(use_stable_segment),
        "stable_segment_i0_raw": int(seg_i0_raw),
        "stable_segment_i1_raw": int(seg_i1_raw),
        "stable_segment_len_raw": int(seg_i1_raw - seg_i0_raw),
        "stable_segment_i0": int(seg_i0),
        "stable_segment_i1": int(seg_i1),
        "stable_segment_len": int(seg_i1 - seg_i0),
        "stable_corr_threshold": float(stable_corr_threshold),
        "stable_corr_median": float(corr_stats["corr_median"]),
        "stable_corr_q10": float(corr_stats["corr_q10"]),
        "stable_corr_q90": float(corr_stats["corr_q90"]),
        "stable_segment_corr_median": float(corr_stats["segment_corr_median"]),
        "stable_segment_corr_q10": float(corr_stats["segment_corr_q10"]),
        "stable_segment_corr_q90": float(corr_stats["segment_corr_q90"]),
        "n_bins": int(n_bins),
        "trace_len": int(trace_len),
        "adc_fs_hz": float(adc_fs_hz),
        "time_step_s": float(dt_trace),
        "trace_rate_hz": float(1.0 / dt_trace),
        "group_bins": int(group_bins),
        "n_groups": int(len(dist_group)),
        "freq_min_hz": float(f_view[0]),
        "freq_max_hz": float(f_view[-1]),
        "combined_threshold": float(thr),
        "peak_threshold": float(peak_thr),
        "ignore_start_km": float(ignore_start_km),
        "signal_end_km": float(signal_end_km),
        "usable_end_km": float(usable_end_km),
        "max_detections": int(max_detections),
        "detected": detected_rows,
        "expected_km": expected_km or [],
    }
    if top_usable is not None:
        res["top_candidate_within_usable"] = top_usable

    (outdir / "detection_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Post-alignment Detection Summary",
        "",
        f"- input: `{aligned_npz}`",
        f"- traces: **{n_traces}**",
        f"- use stable segment crop: **{bool(use_stable_segment)}**",
        f"- stable segment suggested by coherence: **[{seg_i0_raw}, {seg_i1_raw})**, len={seg_i1_raw - seg_i0_raw}",
        f"- processed segment: **[{seg_i0}, {seg_i1})**, len={seg_i1 - seg_i0}",
        f"- adjacent corr (full) median/q10/q90: **{corr_stats['corr_median']:.3f} / {corr_stats['corr_q10']:.3f} / {corr_stats['corr_q90']:.3f}**",
        f"- adjacent corr (stable segment) median/q10/q90: **{corr_stats['segment_corr_median']:.3f} / {corr_stats['segment_corr_q10']:.3f} / {corr_stats['segment_corr_q90']:.3f}**",
        f"- distance bins: **{n_bins}**",
        f"- trace rate: **{(1.0 / dt_trace):.3f} Hz**",
        f"- grouped bins per score point: **{group_bins}**",
        f"- combined threshold: **{thr:.3f}**",
        f"- peak threshold: **{peak_thr:.3f}**",
        f"- ignored near-start zone: **0..{ignore_start_km:.2f} km**",
        f"- auto signal end estimate: **{signal_end_km:.3f} km**",
        f"- usable detection range end: **{usable_end_km:.3f} km**",
        f"- max detections: **{max_detections}**",
        "",
        "## Detected disturbance candidates",
        "",
    ]
    if detected_rows:
        for idx, row in enumerate(detected_rows, start=1):
            md_lines.append(
                f"{idx}. {row['distance_km']:.3f} km | combined={row['combined_score']:.3f} | "
                f"peak={row['peak_spectral_z']:.3f} | broad={row['broad_score']:.3f} | energy={row['energy_z']:.3f}"
            )
    else:
        md_lines.append("No candidates above threshold.")

    if top_usable is not None:
        md_lines.extend(
            [
                "",
                "## Top usable-zone candidate (ranked, threshold-agnostic)",
                "",
                f"- distance: **{top_usable['best_distance_km_within_usable']:.3f} km**",
                f"- combined score: **{top_usable['best_combined_score_within_usable']:.3f}**",
                f"- peak spectral z: **{top_usable['best_peak_spectral_z_within_usable']:.3f}**",
                f"- broadband score: **{top_usable['best_broad_score_within_usable']:.3f}**",
                f"- time-domain energy z: **{top_usable['best_energy_z_within_usable']:.3f}**",
                "- files:",
                "  - `top_candidate_within_usable.json`",
                "  - `suspicious_zone_raw_pm3km_top_usable.png`",
            ]
        )

    if expected_km:
        md_lines.extend(["", "## Expected references", ""])
        for d in expected_km:
            nearest = min((abs(row["distance_km"] - d) for row in detected_rows), default=float("inf"))
            md_lines.append(f"- expected {d:.3f} km -> nearest detected delta {nearest:.3f} km")

    (outdir / "detection_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return res


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.parser.detect_from_aligned",
        description="Run post-alignment filtering + frequency-agnostic disturbance detection",
    )
    p.add_argument("--aligned-npz", type=Path, required=True)
    p.add_argument("--outdir", type=Path, default=Path("reports/figures/post_align_detect"))
    p.add_argument("--group-bins", type=int, default=20)
    p.add_argument("--freq-min-hz", type=float, default=5.0)
    p.add_argument("--freq-max-hz", type=float, default=500.0)
    p.add_argument("--min-sep-km", type=float, default=2.0)
    p.add_argument("--threshold-k", type=float, default=3.0)
    p.add_argument("--peak-threshold-k", type=float, default=5.0)
    p.add_argument("--ignore-start-km", type=float, default=1.0)
    p.add_argument("--max-detections", type=int, default=8)
    p.add_argument("--max-traces", type=int, default=None)
    p.add_argument("--end-guard-km", type=float, default=1.0)
    p.add_argument("--stable-corr-threshold", type=float, default=0.95)
    p.add_argument("--stable-min-traces", type=int, default=400)
    p.add_argument(
        "--use-stable-segment",
        action="store_true",
        help="Process only longest high-coherence time segment (off by default: full file is used)",
    )
    p.add_argument(
        "--expected-km",
        type=float,
        nargs="*",
        default=None,
        help="Optional reference distances to compare against",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    summary = run_detection(
        aligned_npz=args.aligned_npz.resolve(),
        outdir=outdir,
        group_bins=args.group_bins,
        freq_min_hz=args.freq_min_hz,
        freq_max_hz=args.freq_max_hz,
        min_sep_km=args.min_sep_km,
        threshold_k=args.threshold_k,
        peak_threshold_k=args.peak_threshold_k,
        ignore_start_km=args.ignore_start_km,
        max_detections=args.max_detections,
        expected_km=args.expected_km,
        max_traces=args.max_traces,
        end_guard_km=args.end_guard_km,
        stable_corr_threshold=args.stable_corr_threshold,
        stable_min_traces=args.stable_min_traces,
        use_stable_segment=args.use_stable_segment,
    )

    print(f"outdir={outdir}")
    print(f"detected_n={len(summary['detected'])}")
    for row in summary["detected"]:
        print(
            "detected_km="
            f"{row['distance_km']:.3f};combined={row['combined_score']:.3f};"
            f"peak={row['peak_spectral_z']:.3f};broad={row['broad_score']:.3f};energy={row['energy_z']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
