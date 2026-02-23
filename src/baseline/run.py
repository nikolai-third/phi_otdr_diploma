"""Mass baseline analytics over full dataset index."""

from __future__ import annotations

import hashlib
import logging
import math
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.types as patypes

from src.utils.cache import ensure_local

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BaselineConfig:
    cache_dir: Path = Path("cache")
    metrics_path: Path = Path("data/processed/baseline_metrics.parquet")
    errors_path: Path = Path("logs/errors.log")
    max_workers: int = 4
    max_bytes: int = 5_000_000
    checkpoint_every: int = 50


def _sample_stats(sample: np.ndarray) -> dict[str, float | int]:
    data = np.asarray(sample)
    flat = data.ravel()
    if flat.size == 0:
        return {
            "sample_mean": np.nan,
            "sample_std": np.nan,
            "sample_min": np.nan,
            "sample_max": np.nan,
            "fraction_zeros": np.nan,
            "fraction_nans": np.nan,
            "sample_count": 0,
        }

    if not np.issubdtype(flat.dtype, np.number):
        return {
            "sample_mean": np.nan,
            "sample_std": np.nan,
            "sample_min": np.nan,
            "sample_max": np.nan,
            "fraction_zeros": np.nan,
            "fraction_nans": np.nan,
            "sample_count": int(flat.size),
        }

    values = flat.astype(np.float64, copy=False)
    is_float = np.issubdtype(flat.dtype, np.floating)
    return {
        "sample_mean": float(np.nanmean(values)),
        "sample_std": float(np.nanstd(values)),
        "sample_min": float(np.nanmin(values)),
        "sample_max": float(np.nanmax(values)),
        "fraction_zeros": float(np.mean(values == 0.0)),
        "fraction_nans": float(np.mean(np.isnan(values))) if is_float else np.nan,
        "sample_count": int(flat.size),
    }


def _max_elements(dtype: np.dtype, max_bytes: int) -> int:
    return max(1, max_bytes // max(1, int(dtype.itemsize)))


def _sample_npy(path: Path, max_bytes: int) -> tuple[np.ndarray, str, str]:
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    max_elems = _max_elements(arr.dtype, max_bytes)
    sample = arr.reshape(-1)[:max_elems]
    return sample, str(tuple(int(x) for x in arr.shape)), str(arr.dtype)


def _sample_npz(path: Path, max_bytes: int) -> tuple[np.ndarray, str, str]:
    with np.load(path, allow_pickle=False) as npz:
        for key in sorted(npz.files):
            arr = npz[key]
            if np.issubdtype(arr.dtype, np.number):
                max_elems = _max_elements(arr.dtype, max_bytes)
                sample = np.asarray(arr).reshape(-1)[:max_elems]
                return sample, str(tuple(int(x) for x in arr.shape)), str(arr.dtype)
    return np.array([]), "()", "unknown"


def _sample_h5(path: Path, max_bytes: int) -> tuple[np.ndarray, str, str]:
    if h5py is None:
        return np.array([]), "()", "h5py-missing"

    with h5py.File(path, "r") as h5f:
        candidates: list[Any] = []

        def visitor(_: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset) and np.issubdtype(obj.dtype, np.number):
                candidates.append(obj)

        h5f.visititems(visitor)
        if not candidates:
            return np.array([]), "()", "unknown"

        ds = candidates[0]
        max_elems = _max_elements(np.dtype(ds.dtype), max_bytes)
        if ds.ndim == 0:
            sample = np.asarray([ds[()]])
        else:
            sample = np.asarray(ds[tuple(slice(0, min(int(dim), max_elems)) for dim in ds.shape)]).reshape(-1)[:max_elems]
        return sample, str(tuple(int(x) for x in ds.shape)), str(ds.dtype)


def _sample_parquet(path: Path, max_bytes: int) -> tuple[np.ndarray, str, str]:
    try:
        pf = pq.ParquetFile(path)
        schema = pf.schema_arrow

        list_cols: list[str] = []
        numeric_cols: list[str] = []
        for field in schema:
            if patypes.is_list(field.type) or patypes.is_large_list(field.type):
                list_cols.append(field.name)
            elif patypes.is_integer(field.type) or patypes.is_floating(field.type):
                numeric_cols.append(field.name)

        if list_cols:
            batch_iter = pf.iter_batches(columns=[list_cols[0]], batch_size=16)
            try:
                batch = next(batch_iter)
                rows = batch.column(0).to_pylist()
                arrays = [np.asarray(r, dtype=np.float64) for r in rows if isinstance(r, (list, tuple)) and len(r) > 0]
                if arrays:
                    width = min(len(a) for a in arrays)
                    matrix = np.vstack([a[:width] for a in arrays])
                    sample = matrix.reshape(-1)[: _max_elements(np.dtype(np.float64), max_bytes)]
                    return sample, str(matrix.shape), "float64"
            except StopIteration:
                pass

        preferred_cols = [c for c in numeric_cols if c.lower() == "data"] + [c for c in numeric_cols if c.lower() != "data"]
        if preferred_cols:
            batch_size = min(1_000_000, _max_elements(np.dtype(np.float64), max_bytes) * 2)
            batch_iter = pf.iter_batches(columns=[preferred_cols[0]], batch_size=batch_size)
            try:
                batch = next(batch_iter)
                values = batch.column(0).to_numpy(zero_copy_only=False)
                sample = np.asarray(values).reshape(-1)[: _max_elements(np.asarray(values).dtype, max_bytes)]
                return sample, str((int(len(values)),)), str(np.asarray(values).dtype)
            except StopIteration:
                pass
    except Exception as exc:
        LOGGER.warning("parquet fallback for %s: %s", path, exc)

    # Fallback for malformed .parquet-like binaries: treat leading bytes as int16 stream.
    with path.open("rb") as fp:
        raw = fp.read(max_bytes)
    if not raw:
        return np.array([]), "()", "bytes"
    arr = np.frombuffer(raw, dtype=np.int16)
    return arr, str((int(arr.size),)), "int16-fallback"


def _extract_metrics(local_path: Path, inferred_format: str, max_bytes: int) -> dict[str, Any]:
    fmt = inferred_format.lower()

    if fmt == "npy":
        sample, shape, dtype = _sample_npy(local_path, max_bytes=max_bytes)
    elif fmt == "npz":
        sample, shape, dtype = _sample_npz(local_path, max_bytes=max_bytes)
    elif fmt in {"h5", "hdf5"}:
        sample, shape, dtype = _sample_h5(local_path, max_bytes=max_bytes)
    elif fmt == "parquet":
        sample, shape, dtype = _sample_parquet(local_path, max_bytes=max_bytes)
    else:
        sample, shape, dtype = np.array([]), "()", "unsupported"

    metrics = _sample_stats(sample)
    metrics.update({"shape": shape, "dtype": dtype})
    return metrics


def _jitter_seconds(record_id: str) -> float:
    b = int(hashlib.sha1(record_id.encode("utf-8")).hexdigest()[:2], 16)
    return 0.1 + (b / 255.0) * 0.2


def _write_error(errors_path: Path, record_id: str, path: str, message: str) -> None:
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    with errors_path.open("a", encoding="utf-8") as fp:
        fp.write(f"{record_id}\t{path}\t{message}\n")


def _load_processed_ids(metrics_path: Path) -> set[str]:
    if not metrics_path.exists():
        return set()
    df = pd.read_parquet(metrics_path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    return set(df["record_id"].astype(str).tolist())


def _flush_metrics(metrics_path: Path, buffered_rows: list[dict[str, Any]]) -> None:
    if not buffered_rows:
        return

    new_df = pd.DataFrame(buffered_rows)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    if metrics_path.exists():
        prev = pd.read_parquet(metrics_path)
        merged = pd.concat([prev, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["record_id"], keep="first")
    else:
        merged = new_df

    merged = merged.sort_values("record_id").reset_index(drop=True)
    merged.to_parquet(metrics_path, index=False)


def _process_one(row: dict[str, Any], config: BaselineConfig) -> dict[str, Any]:
    record_id = str(row["record_id"])
    abs_path = Path(str(row["abs_path"]))
    inferred_format = str(row["inferred_format"])

    local_path = ensure_local(abs_path, cache_dir=config.cache_dir)
    metrics = _extract_metrics(local_path, inferred_format=inferred_format, max_bytes=config.max_bytes)

    result: dict[str, Any] = {
        "record_id": record_id,
        "path": str(row["path"]),
        "abs_path": str(abs_path),
        "local_path": str(local_path),
        "file_group": str(row["file_group"]),
        "inferred_format": inferred_format,
        "size_bytes": int(row["size_bytes"]),
        "file_size_mb": float(row["file_size_mb"]),
        "has_metadata": bool(row["has_metadata"]),
        "status": "ok",
        "error": None,
    }
    result.update(metrics)

    time.sleep(_jitter_seconds(record_id))
    return result


def run_baseline(
    index_path: str | Path = "data/interim/index.parquet",
    cache_dir: str | Path = "cache",
    metrics_path: str | Path = "data/processed/baseline_metrics.parquet",
    errors_path: str | Path = "logs/errors.log",
    max_workers: int = 4,
    max_bytes: int = 5_000_000,
    checkpoint_every: int = 50,
    progress_log_minutes: int = 10,
) -> Path:
    """Run resumable baseline analysis on all indexed records."""

    config = BaselineConfig(
        cache_dir=Path(cache_dir),
        metrics_path=Path(metrics_path),
        errors_path=Path(errors_path),
        max_workers=min(4, max(1, int(max_workers))),
        max_bytes=max_bytes,
        checkpoint_every=checkpoint_every,
    )

    index_df = pd.read_parquet(index_path)
    if index_df.empty:
        raise ValueError(f"index is empty: {index_path}")

    processed_ids = _load_processed_ids(config.metrics_path)
    pending_df = index_df[~index_df["record_id"].astype(str).isin(processed_ids)].copy()
    pending_df = pending_df.sort_values("record_id").reset_index(drop=True)

    total = len(pending_df)
    LOGGER.info("baseline pending records: %d (already processed: %d)", total, len(processed_ids))
    if total == 0:
        return config.metrics_path

    buffered: list[dict[str, Any]] = []
    done_count = 0
    pending: dict[Future[dict[str, Any]], dict[str, Any]] = {}
    max_pending = max(config.max_workers * 4, 8)

    next_progress_ts = time.time() + progress_log_minutes * 60

    with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
        for row in pending_df.to_dict(orient="records"):
            while len(pending) >= max_pending:
                done_set, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                for fut in done_set:
                    src_row = pending.pop(fut)
                    try:
                        buffered.append(fut.result())
                    except Exception as exc:  # pragma: no cover
                        _write_error(config.errors_path, str(src_row["record_id"]), str(src_row["path"]), str(exc))
                        buffered.append(
                            {
                                "record_id": str(src_row["record_id"]),
                                "path": str(src_row["path"]),
                                "abs_path": str(src_row["abs_path"]),
                                "local_path": None,
                                "file_group": str(src_row["file_group"]),
                                "inferred_format": str(src_row["inferred_format"]),
                                "size_bytes": int(src_row["size_bytes"]),
                                "file_size_mb": float(src_row["file_size_mb"]),
                                "has_metadata": bool(src_row["has_metadata"]),
                                "status": "error",
                                "error": str(exc),
                                "shape": "()",
                                "dtype": "unknown",
                                "sample_mean": np.nan,
                                "sample_std": np.nan,
                                "sample_min": np.nan,
                                "sample_max": np.nan,
                                "fraction_zeros": np.nan,
                                "fraction_nans": np.nan,
                                "sample_count": 0,
                            }
                        )
                    done_count += 1

                    if done_count % config.checkpoint_every == 0:
                        _flush_metrics(config.metrics_path, buffered)
                        LOGGER.info("checkpoint: %d/%d records", done_count, total)
                        buffered.clear()

                    now = time.time()
                    if now >= next_progress_ts:
                        pct = (done_count / total) * 100.0 if total else 100.0
                        LOGGER.info("progress: %d/%d (%.2f%%)", done_count, total, pct)
                        next_progress_ts = now + progress_log_minutes * 60

            fut = pool.submit(_process_one, row, config)
            pending[fut] = row

        while pending:
            done_set, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
            for fut in done_set:
                src_row = pending.pop(fut)
                try:
                    buffered.append(fut.result())
                except Exception as exc:  # pragma: no cover
                    _write_error(config.errors_path, str(src_row["record_id"]), str(src_row["path"]), str(exc))
                    buffered.append(
                        {
                            "record_id": str(src_row["record_id"]),
                            "path": str(src_row["path"]),
                            "abs_path": str(src_row["abs_path"]),
                            "local_path": None,
                            "file_group": str(src_row["file_group"]),
                            "inferred_format": str(src_row["inferred_format"]),
                            "size_bytes": int(src_row["size_bytes"]),
                            "file_size_mb": float(src_row["file_size_mb"]),
                            "has_metadata": bool(src_row["has_metadata"]),
                            "status": "error",
                            "error": str(exc),
                            "shape": "()",
                            "dtype": "unknown",
                            "sample_mean": np.nan,
                            "sample_std": np.nan,
                            "sample_min": np.nan,
                            "sample_max": np.nan,
                            "fraction_zeros": np.nan,
                            "fraction_nans": np.nan,
                            "sample_count": 0,
                        }
                    )
                done_count += 1
                if done_count % config.checkpoint_every == 0:
                    _flush_metrics(config.metrics_path, buffered)
                    LOGGER.info("checkpoint: %d/%d records", done_count, total)
                    buffered.clear()

                now = time.time()
                if now >= next_progress_ts:
                    pct = (done_count / total) * 100.0 if total else 100.0
                    LOGGER.info("progress: %d/%d (%.2f%%)", done_count, total, pct)
                    next_progress_ts = now + progress_log_minutes * 60

    _flush_metrics(config.metrics_path, buffered)
    LOGGER.info("baseline completed: %d records processed", done_count)
    return config.metrics_path
