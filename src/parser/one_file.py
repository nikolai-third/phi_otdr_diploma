"""One-file reflectogram parsing pipeline with diagnostic plots."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Iterable

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
    trace_len: int = 55_000
    max_traces: int = 200
    min_gap_factor: float = 1.15
    low_level_factor: float = 0.10
    rise_factor: float = 0.075
    max_shift: int = 300
    cc_window_start: int = 500
    cc_window_len: int = 12_000
    raw_plot_points: int = 1_000_000
    auto_tune: bool = False


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


def _detect_starts(data: np.ndarray, cfg: ParseConfig) -> np.ndarray:
    min_signal = float(np.min(data))
    max_signal = float(np.max(data))
    dyn = max_signal - min_signal

    low_thr = min_signal + cfg.low_level_factor * dyn
    low_regions = data < low_thr

    gradient = np.zeros_like(data)
    gradient[1:-1] = (data[2:] - data[:-2]) / 2.0
    gradient[0] = gradient[1]
    gradient[-1] = gradient[-2]

    rise_thr = cfg.rise_factor * dyn
    sharp_rises = gradient > rise_thr

    potential = np.where(low_regions[:-1] & sharp_rises[1:])[0] + 1
    if len(potential) == 0:
        return np.array([], dtype=np.int64)

    min_gap = int(cfg.trace_len * cfg.min_gap_factor)
    gaps = np.concatenate(([min_gap + 1], np.diff(potential)))
    starts = potential[gaps >= min_gap]
    return starts.astype(np.int64)


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


def _best_shift(ref: np.ndarray, cur: np.ndarray, max_shift: int) -> int:
    best_s = 0
    best_score = -np.inf
    for s in range(-max_shift, max_shift + 1):
        if s >= 0:
            a = ref[s:]
            b = cur[: len(a)]
        else:
            a = ref[:s]
            b = cur[-s:]
        if len(a) < 128:
            continue
        score = float(np.dot(a, b))
        if score > best_score:
            best_score = score
            best_s = s
    return best_s


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
        s = _best_shift(ref, cur, cfg.max_shift)
        shifts[i] = s

        if s >= 0:
            aligned[i, :-s or None] = traces[i, s:]
            if s > 0:
                aligned[i, -s:] = traces[i, -1]
        else:
            aligned[i, -s:] = traces[i, :s]
            aligned[i, : -s] = traces[i, 0]

    return aligned, shifts


def _save_plot(path: Path, fig: plt.Figure) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_raw_segment(data: np.ndarray, starts: np.ndarray, out: Path, points: int) -> None:
    n = min(points, len(data))
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, data[:n], linewidth=0.8)
    starts_vis = starts[starts < n]
    if len(starts_vis) > 0:
        ax.scatter(starts_vis, data[starts_vis], s=8)
    ax.set_title("Raw signal segment with detected starts")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.2)
    _save_plot(out, fig)


def _plot_waterfall(traces: np.ndarray, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(traces, aspect="auto", origin="lower")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Range bin")
    ax.set_ylabel("Trace index")
    _save_plot(out, fig)


def run_one_file(path: Path, outdir: Path, cfg: ParseConfig) -> dict[str, float | int]:
    data = _read_data_stream(path)

    if cfg.auto_tune:
        cfg = _auto_tune_config(data, cfg)

    starts = _detect_starts(data, cfg)
    traces = _extract_reflectograms(data, starts, trace_len=cfg.trace_len, max_traces=cfg.max_traces)
    aligned, shifts = _align_traces_cc(traces, cfg)

    _plot_raw_segment(data, starts, outdir / "parser_raw_segment.png", points=cfg.raw_plot_points)
    _plot_waterfall(traces, outdir / "parser_waterfall_before.png", "Waterfall before alignment")
    _plot_waterfall(aligned, outdir / "parser_waterfall_after.png", "Waterfall after cross-correlation alignment")

    metrics = {
        "n_samples": int(len(data)),
        "n_detected_starts": int(len(starts)),
        "n_extracted_traces": int(traces.shape[0]),
        "trace_len": int(traces.shape[1]),
        "shift_abs_mean": float(np.mean(np.abs(shifts))),
        "shift_abs_p95": float(np.quantile(np.abs(shifts), 0.95)),
        "shift_std": float(np.std(shifts)),
    }

    lines = [
        "# Parser Diagnostics",
        "",
        f"- file: `{path}`",
        f"- samples: **{metrics['n_samples']}**",
        f"- detected starts: **{metrics['n_detected_starts']}**",
        f"- extracted traces: **{metrics['n_extracted_traces']}**",
        f"- trace_len: **{metrics['trace_len']}**",
        f"- mean |shift|: **{metrics['shift_abs_mean']:.3f}**",
        f"- p95 |shift|: **{metrics['shift_abs_p95']:.3f}**",
        f"- shift std: **{metrics['shift_std']:.3f}**",
        "",
        "## Files",
        "",
        "- parser_raw_segment.png",
        "- parser_waterfall_before.png",
        "- parser_waterfall_after.png",
    ]
    (outdir / "parser_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (outdir / "parser_params.json").write_text(
        json.dumps(
            {
                "trace_len": cfg.trace_len,
                "max_traces": cfg.max_traces,
                "min_gap_factor": cfg.min_gap_factor,
                "low_level_factor": cfg.low_level_factor,
                "rise_factor": cfg.rise_factor,
                "max_shift": cfg.max_shift,
                "cc_window_start": cfg.cc_window_start,
                "cc_window_len": cfg.cc_window_len,
                "raw_plot_points": cfg.raw_plot_points,
                "auto_tune": cfg.auto_tune,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return metrics


def _auto_tune_config(data: np.ndarray, cfg: ParseConfig) -> ParseConfig:
    low_grid = [0.07, 0.09, 0.10, 0.12]
    rise_grid = [0.05, 0.06, 0.07, 0.075, 0.09]
    gap_grid = [1.10, 1.15, 1.20]

    best_score = np.inf
    best_cfg = cfg

    for low in low_grid:
        for rise in rise_grid:
            for gap in gap_grid:
                candidate = ParseConfig(
                    trace_len=cfg.trace_len,
                    max_traces=cfg.max_traces,
                    min_gap_factor=gap,
                    low_level_factor=low,
                    rise_factor=rise,
                    max_shift=cfg.max_shift,
                    cc_window_start=cfg.cc_window_start,
                    cc_window_len=cfg.cc_window_len,
                    raw_plot_points=cfg.raw_plot_points,
                    auto_tune=cfg.auto_tune,
                )
                starts = _detect_starts(data, candidate)
                if len(starts) < 20:
                    continue
                try:
                    traces = _extract_reflectograms(
                        data,
                        starts,
                        trace_len=candidate.trace_len,
                        max_traces=candidate.max_traces,
                    )
                    _, shifts = _align_traces_cc(traces, candidate)
                except Exception:
                    continue

                mean_abs = float(np.mean(np.abs(shifts)))
                p95 = float(np.quantile(np.abs(shifts), 0.95))
                std = float(np.std(shifts))
                n = traces.shape[0]
                score = mean_abs + 0.25 * p95 + 0.02 * std - 0.03 * n
                if score < best_score:
                    best_score = score
                    best_cfg = candidate

    return best_cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.parser.one_file", description="Run reflectogram parser on one file")
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("reports/figures/parser"))
    parser.add_argument("--trace-len", type=int, default=55_000)
    parser.add_argument("--max-traces", type=int, default=200)
    parser.add_argument("--min-gap-factor", type=float, default=1.15)
    parser.add_argument("--low-level-factor", type=float, default=0.10)
    parser.add_argument("--rise-factor", type=float, default=0.075)
    parser.add_argument("--max-shift", type=int, default=300)
    parser.add_argument("--cc-window-start", type=int, default=500)
    parser.add_argument("--cc-window-len", type=int, default=12000)
    parser.add_argument("--raw-plot-points", type=int, default=1_000_000)
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
    )

    metrics = run_one_file(path=args.file, outdir=args.outdir, cfg=cfg)
    for k, v in metrics.items():
        print(f"{k}={v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
