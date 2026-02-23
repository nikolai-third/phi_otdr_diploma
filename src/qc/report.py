"""QC report CLI: generate basic figures for one dataset record."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure non-interactive backend and writable cache in restricted environments.
cache_root = (Path("data") / "interim" / ".cache").resolve()
mpl_root = (Path("data") / "interim" / ".mplconfig").resolve()
cache_root.mkdir(parents=True, exist_ok=True)
mpl_root.mkdir(parents=True, exist_ok=True)
os.environ["XDG_CACHE_HOME"] = str(cache_root)
os.environ["MPLCONFIGDIR"] = str(mpl_root)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.types as patypes

from src.io.reader import RecordHandle, UnsupportedFormatError, open_record


@dataclass(frozen=True)
class QCParams:
    max_traces: int = 512
    max_range_bins: int = 2048
    max_bytes: int = 25_000_000
    trace_len: int = 55_000
    distance_step_m: float = 2.02
    time_step_s: float = 0.02


def _legacy_segment_1d(signal: np.ndarray, trace_len: int, max_traces: int) -> np.ndarray:
    """Split long 1D stream into reflectograms using legacy trigger heuristic."""

    data = np.asarray(signal, dtype=np.float64)
    if data.size < trace_len * 2:
        usable = (data.size // trace_len) * trace_len
        if usable == 0:
            return data.reshape(1, -1)
        return data[:usable].reshape(-1, trace_len)[:max_traces]

    min_signal = float(np.nanmin(data))
    max_signal = float(np.nanmax(data))
    dyn = max_signal - min_signal
    if dyn <= 0:
        usable = (data.size // trace_len) * trace_len
        return data[:usable].reshape(-1, trace_len)[:max_traces]

    low_signal_threshold = min_signal + 0.10 * dyn
    low_signal_regions = data < low_signal_threshold

    gradient = np.zeros_like(data)
    gradient[1:-1] = (data[2:] - data[:-2]) / 2.0
    gradient[0] = gradient[1]
    gradient[-1] = gradient[-2]

    rise_threshold = 0.075 * dyn
    sharp_rises = gradient > rise_threshold
    potential_starts = np.where(low_signal_regions[:-1] & sharp_rises[1:])[0] + 1

    if potential_starts.size == 0:
        usable = (data.size // trace_len) * trace_len
        if usable == 0:
            return data.reshape(1, -1)
        return data[:usable].reshape(-1, trace_len)[:max_traces]

    gaps = np.concatenate(([int(trace_len * 1.2) + 1000], np.diff(potential_starts)))
    starts = potential_starts[gaps >= int(trace_len * 1.2)]
    starts = starts[:max_traces]

    if starts.size == 0:
        usable = (data.size // trace_len) * trace_len
        if usable == 0:
            return data.reshape(1, -1)
        return data[:usable].reshape(-1, trace_len)[:max_traces]

    indices = starts[:, np.newaxis] + np.arange(trace_len)
    flat = indices.ravel().astype(np.int64)
    flat = flat[flat < data.size]
    usable = (flat.size // trace_len) * trace_len
    if usable == 0:
        return data.reshape(1, -1)
    return data[flat[:usable]].reshape(-1, trace_len)


def _to_2d_matrix(array: np.ndarray, params: QCParams) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = _legacy_segment_1d(arr, trace_len=params.trace_len, max_traces=params.max_traces)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    arr = arr[: params.max_traces, : params.max_range_bins]
    arr = arr.astype(np.float32, copy=False)

    if arr.nbytes > params.max_bytes:
        scale = np.sqrt(arr.nbytes / params.max_bytes)
        t_stride = max(1, int(np.ceil(scale)))
        r_stride = max(1, int(np.ceil(scale)))
        arr = arr[::t_stride, ::r_stride]

    if arr.size == 0:
        raise ValueError("Selected slice is empty after applying limits")

    return arr


def _first_numeric_from_npz(npz: Any) -> np.ndarray:
    for key in sorted(npz.files):
        arr = npz[key]
        if np.issubdtype(arr.dtype, np.number):
            return arr
    raise UnsupportedFormatError("NPZ record has no numeric arrays")


def _first_numeric_from_h5(h5_file: Any) -> np.ndarray:
    candidates: list[Any] = []

    def visitor(_: str, obj: Any) -> None:
        if hasattr(obj, "dtype") and hasattr(obj, "shape") and np.issubdtype(obj.dtype, np.number):
            candidates.append(obj)

    h5_file.visititems(visitor)
    if not candidates:
        raise UnsupportedFormatError("HDF5 record has no numeric datasets")

    ds = candidates[0]
    if len(ds.shape) == 0:
        return np.array([[ds[()]]], dtype=np.float32)
    return np.asarray(ds)


def _matrix_from_parquet(parquet_file: Any, params: QCParams) -> np.ndarray:
    schema = parquet_file.schema_arrow
    list_cols: list[str] = []
    numeric_cols: list[str] = []

    for field in schema:
        field_type = field.type
        if patypes.is_list(field_type) or patypes.is_large_list(field_type):
            list_cols.append(field.name)
        elif patypes.is_integer(field_type) or patypes.is_floating(field_type):
            numeric_cols.append(field.name)

    batch_size = min(2_000_000, max(params.max_traces * params.max_range_bins, params.trace_len * 4))

    if list_cols:
        batches = parquet_file.iter_batches(columns=[list_cols[0]], batch_size=params.max_traces)
        try:
            batch = next(batches)
        except StopIteration:
            batch = None
        if batch is not None:
            rows = batch.column(0).to_pylist()[: params.max_traces]
            dense_rows = [np.asarray(r, dtype=np.float32) for r in rows if isinstance(r, (list, tuple)) and len(r) > 0]
            if dense_rows:
                width = min(params.max_range_bins, min(len(r) for r in dense_rows))
                return np.vstack([r[:width] for r in dense_rows])

    # Legacy parquet format often stores flattened data stream in single numeric column `data`.
    preferred_cols = [c for c in numeric_cols if c.lower() == "data"] + [c for c in numeric_cols if c.lower() != "data"]
    if preferred_cols:
        batches = parquet_file.iter_batches(columns=[preferred_cols[0]], batch_size=batch_size)
        try:
            batch = next(batches)
        except StopIteration:
            batch = None
        if batch is not None:
            signal = batch.column(0).to_numpy(zero_copy_only=False)
            return _to_2d_matrix(np.asarray(signal), params)

    raise UnsupportedFormatError("Parquet record has no supported numeric representation")


def load_matrix(handle: RecordHandle, params: QCParams) -> np.ndarray:
    """Load bounded 2D matrix for QC from record handle."""

    if handle.fmt == "npy":
        return _to_2d_matrix(handle.data, params)
    if handle.fmt == "npz":
        return _to_2d_matrix(_first_numeric_from_npz(handle.data), params)
    if handle.fmt == "h5":
        return _to_2d_matrix(_first_numeric_from_h5(handle.data), params)
    if handle.fmt == "parquet":
        matrix = _matrix_from_parquet(handle.data, params)
        return _to_2d_matrix(matrix, params)

    raise UnsupportedFormatError(f"QC loader does not support format: {handle.fmt}")


def _save_plot(fig: Any, outdir: Path, stem: str) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    png_path = outdir / f"{stem}.png"
    pdf_path = outdir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def build_waterfall_figure(matrix: np.ndarray, outdir: Path, record_id: str, params: QCParams) -> list[Path]:
    """Save waterfall heatmap for one record."""

    x_km = np.arange(matrix.shape[1], dtype=np.float64) * params.distance_step_m / 1000.0
    y_s = np.arange(matrix.shape[0], dtype=np.float64) * params.time_step_s

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        extent=[float(x_km.min()), float(x_km.max()), float(y_s.min()), float(y_s.max())],
    )
    ax.set_xlabel("Distance, km")
    ax.set_ylabel("Time, s")
    ax.set_title(f"Waterfall: {record_id}")
    fig.colorbar(im, ax=ax, label="Amplitude")
    return _save_plot(fig, outdir, f"{record_id}_waterfall")


def build_qc_report(
    index_path: Path,
    record_id: str,
    outdir: Path,
    max_traces: int,
    max_range_bins: int,
    max_bytes: int,
    trace_len: int,
    distance_step_m: float,
    time_step_s: float,
) -> list[Path]:
    """Generate QC plots for a single record."""

    params = QCParams(
        max_traces=max_traces,
        max_range_bins=max_range_bins,
        max_bytes=max_bytes,
        trace_len=trace_len,
        distance_step_m=distance_step_m,
        time_step_s=time_step_s,
    )

    handle = open_record(record_id=record_id, index_path=index_path)
    try:
        matrix = load_matrix(handle, params)
    finally:
        handle.close()

    outputs: list[Path] = []
    outputs.extend(build_waterfall_figure(matrix, outdir=outdir, record_id=record_id, params=params))

    mean_dist = np.nanmean(matrix, axis=0)
    std_dist = np.nanstd(matrix, axis=0)
    x = np.arange(matrix.shape[1])

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.plot(x, mean_dist, label="mean")
    ax2.plot(x, std_dist, label="std")
    ax2.set_xlabel("Distance bin")
    ax2.set_ylabel("Value")
    ax2.set_title(f"Record {record_id}: mean/std by distance")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.2)
    outputs.extend(_save_plot(fig2, outdir, f"{record_id}_mean_std"))

    bin_idx = matrix.shape[1] // 2
    signal = matrix[:, bin_idx].astype(np.float64)
    signal = signal - np.nanmean(signal)

    fig3, (ax31, ax32) = plt.subplots(2, 1, figsize=(9, 7))
    if signal.size >= 8:
        freqs = np.fft.rfftfreq(signal.size, d=params.time_step_s)
        psd = np.abs(np.fft.rfft(signal)) ** 2 / max(1, signal.size)
        ax31.plot(freqs, psd)
        ax31.set_xlabel("Frequency, Hz")
        ax31.set_ylabel("Power")
        ax31.set_title(f"Record {record_id}: PSD (distance bin {bin_idx})")
        ax31.grid(True, alpha=0.2)

        nfft = min(256, max(16, signal.size // 4))
        noverlap = nfft // 2
        ax32.specgram(signal, NFFT=nfft, Fs=1.0 / params.time_step_s, noverlap=noverlap)
        ax32.set_xlabel("Time, s")
        ax32.set_ylabel("Frequency, Hz")
        ax32.set_title("STFT / Spectrogram")
    else:
        ax31.text(0.5, 0.5, "Signal too short for PSD", ha="center", va="center")
        ax31.axis("off")
        ax32.text(0.5, 0.5, "Signal too short for STFT", ha="center", va="center")
        ax32.axis("off")

    outputs.extend(_save_plot(fig3, outdir, f"{record_id}_psd_stft"))
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.qc.report", description="Generate QC plots for one record")
    parser.add_argument("--index", type=Path, default=Path("data/interim/dataset_index.parquet"), help="Dataset index parquet path")
    parser.add_argument("--record", type=str, required=True, help="Record id from dataset index")
    parser.add_argument("--outdir", type=Path, default=Path("reports/figures"), help="Directory for saved QC figures")
    parser.add_argument("--max-traces", type=int, default=512, help="Max number of time traces")
    parser.add_argument("--max-range-bins", type=int, default=2048, help="Max number of distance bins")
    parser.add_argument("--max-bytes", type=int, default=25_000_000, help="Max bytes for sampled QC matrix")
    parser.add_argument("--trace-len", type=int, default=55_000, help="Legacy reflectogram length for 1D stream segmentation")
    parser.add_argument("--distance-step-m", type=float, default=2.02, help="Distance step per bin in meters")
    parser.add_argument("--time-step-s", type=float, default=0.02, help="Time step per trace in seconds")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    outputs = build_qc_report(
        index_path=args.index,
        record_id=args.record,
        outdir=args.outdir,
        max_traces=args.max_traces,
        max_range_bins=args.max_range_bins,
        max_bytes=args.max_bytes,
        trace_len=args.trace_len,
        distance_step_m=args.distance_step_m,
        time_step_s=args.time_step_s,
    )

    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
