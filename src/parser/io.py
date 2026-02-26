"""I/O and reporting for one-file parser runs."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

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

from src.parser.config import ParseConfig
from src.parser.core import parse_reflectograms
from src.parser.templates import align_traces_cc, estimate_residual_jitter, select_alignment_window


def sanitize_tag(text: str, max_len: int = 80) -> str:
    """Build filesystem-friendly tag chunk from free text."""

    cleaned = re.sub(r"[^\w.-]+", "_", text.strip(), flags=re.UNICODE).strip("_.")
    if not cleaned:
        cleaned = "unknown"
    return cleaned[:max_len]


def build_source_tag(path: Path) -> str:
    """Stable short tag that encodes source file identity for output names."""

    parent = sanitize_tag(path.parent.name)
    stem = sanitize_tag(path.stem)
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    return f"{parent}__{stem}__{digest}"


def read_data_stream(path: Path, max_points: int | None = None) -> np.ndarray:
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


def _save_plot(path: Path, fig: plt.Figure) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_raw_segment(data: np.ndarray, starts: np.ndarray, out: Path, points: int, fs_hz: float) -> None:
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


def plot_waterfall(traces: np.ndarray, out: Path, title: str, fs_hz: float, cmap: str, exp_alpha: float) -> None:
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


def _write_parser_diagnostics(
    outdir: Path,
    path: Path,
    metrics: dict[str, Any],
    cfg: ParseConfig,
    cfg_align: ParseConfig,
    source_tag: str,
    raw_segment_name: str,
    waterfall_before_name: str,
    waterfall_after_name: str,
    trace_len_used: int,
) -> None:
    lines = [
        "# Parser Diagnostics",
        "",
        f"- file: `{path}`",
        f"- ADC sampling rate: **{cfg.adc_fs_hz:.0f} Hz**",
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
                "source_tag": source_tag,
                "trace_len_used": trace_len_used,
                "config": asdict(cfg),
                "cc_window_start_used": int(cfg_align.cc_window_start),
                "cc_window_len_used": int(cfg_align.cc_window_len),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def run_one_file(path: Path, outdir: Path, cfg: ParseConfig) -> dict[str, float | int | str]:
    data = read_data_stream(path, max_points=cfg.max_samples)
    source_tag = build_source_tag(path)

    parse_result = parse_reflectograms(data, cfg)
    trace_len = int(parse_result.trace_len)
    starts = parse_result.starts
    traces = parse_result.traces

    if cfg.auto_select_cc_window:
        cc_start = select_alignment_window(traces, cfg)
        cfg_align = replace(cfg, cc_window_start=cc_start)
    else:
        cfg_align = cfg

    residual_before = estimate_residual_jitter(traces, cfg_align)
    aligned_candidate, shifts_candidate = align_traces_cc(traces, cfg_align)
    residual_after_candidate = estimate_residual_jitter(aligned_candidate, cfg_align)

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

    plot_raw_segment(
        data,
        starts,
        outdir / raw_segment_name,
        points=cfg.raw_plot_points,
        fs_hz=cfg.adc_fs_hz,
    )
    plot_waterfall(
        traces,
        outdir / waterfall_before_name,
        "Waterfall before alignment",
        fs_hz=cfg.adc_fs_hz,
        cmap=cfg.waterfall_cmap,
        exp_alpha=cfg.waterfall_exp_alpha,
    )
    plot_waterfall(
        aligned,
        outdir / waterfall_after_name,
        "Waterfall after cross-correlation alignment",
        fs_hz=cfg.adc_fs_hz,
        cmap=cfg.waterfall_cmap,
        exp_alpha=cfg.waterfall_exp_alpha,
    )

    detected_period = int(round(float(np.median(np.diff(starts))))) if len(starts) > 1 else trace_len
    detected_period = max(1, detected_period)
    if len(starts) > 0:
        first_start = int(starts[0])
        max_start = len(data) - trace_len
        expected_traces = max(0, (max_start - first_start) // trace_len + 1) if max_start >= first_start else 0
    else:
        expected_traces = 0
    coverage = float(len(starts) / expected_traces) if expected_traces > 0 else 0.0

    metrics: dict[str, float | int | str] = {
        "source_tag": source_tag,
        "n_samples": int(len(data)),
        "n_detected_starts": int(len(starts)),
        "n_extracted_traces": int(traces.shape[0]),
        "first_start": int(starts[0]) if len(starts) else -1,
        "last_start": int(starts[-1]) if len(starts) else -1,
        "trace_len": int(traces.shape[1]),
        "trace_duration_us": float(traces.shape[1] * 1e6 / cfg.adc_fs_hz),
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
        "detected_period_us": float(detected_period * 1e6 / cfg.adc_fs_hz),
    }

    _write_parser_diagnostics(
        outdir=outdir,
        path=path,
        metrics=metrics,
        cfg=cfg,
        cfg_align=cfg_align,
        source_tag=source_tag,
        raw_segment_name=raw_segment_name,
        waterfall_before_name=waterfall_before_name,
        waterfall_after_name=waterfall_after_name,
        trace_len_used=trace_len,
    )
    return metrics
