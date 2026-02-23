"""Unified lazy reader for dataset records."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    import h5py
except ImportError:  # pragma: no cover - optional dependency in runtime
    h5py = None  # type: ignore[assignment]


class RecordNotFoundError(KeyError):
    """Record id does not exist in dataset index."""


class UnsupportedFormatError(ValueError):
    """Record format cannot be opened by current reader."""


@dataclass
class RecordHandle:
    """Handle for a lazily opened record."""

    record_id: str
    fmt: str
    source_path: str
    data: Any

    def close(self) -> None:
        """Close underlying resource if supported."""

        close_fn = getattr(self.data, "close", None)
        if callable(close_fn):
            close_fn()


def _to_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, tuple):
        return [str(x) for x in value]
    if isinstance(value, np.ndarray):
        return [str(x) for x in value.tolist()]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                loaded = json.loads(stripped)
                if isinstance(loaded, list):
                    return [str(x) for x in loaded]
            except Exception:
                pass
        if stripped:
            return [stripped]
    return []


def _select_supported_path(paths: list[str]) -> tuple[str, str]:
    priority = [".npy", ".npz", ".h5", ".hdf5", ".parquet"]
    p_objs = [Path(p) for p in paths]
    for ext in priority:
        candidates = sorted([p for p in p_objs if p.suffix.lower() == ext])
        if candidates:
            return ext, str(candidates[0])
    observed = sorted(set(p.suffix.lower() for p in p_objs))
    raise UnsupportedFormatError(
        "No supported files in record. Supported: npy/npz/h5/hdf5/parquet. "
        f"Observed extensions: {observed}"
    )


def open_record(record_id: str, index_path: str | Path) -> RecordHandle:
    """Open record lazily from index parquet.

    Returns a RecordHandle with underlying lazy object:
    - npy: numpy.memmap
    - npz: numpy.lib.npyio.NpzFile
    - h5/hdf5: h5py.File
    - parquet: pyarrow.parquet.ParquetFile
    """

    index_df = pd.read_parquet(index_path)
    matches = index_df[index_df["record_id"] == record_id]
    if matches.empty:
        raise RecordNotFoundError(f"Record not found: {record_id}")

    row = matches.iloc[0]
    abs_paths = _to_list(row.get("abs_paths"))
    paths = abs_paths or _to_list(row.get("paths"))
    if not paths:
        raise UnsupportedFormatError(f"Record {record_id} has empty path list")

    ext, path = _select_supported_path(paths)

    if ext == ".npy":
        return RecordHandle(record_id=record_id, fmt="npy", source_path=path, data=np.load(path, mmap_mode="r", allow_pickle=False))
    if ext == ".npz":
        return RecordHandle(record_id=record_id, fmt="npz", source_path=path, data=np.load(path, allow_pickle=False))
    if ext in {".h5", ".hdf5"}:
        if h5py is None:
            raise UnsupportedFormatError("h5py is required for h5/hdf5 records but is not installed")
        return RecordHandle(record_id=record_id, fmt="h5", source_path=path, data=h5py.File(path, "r"))
    if ext == ".parquet":
        return RecordHandle(record_id=record_id, fmt="parquet", source_path=path, data=pq.ParquetFile(path))

    raise UnsupportedFormatError(f"Unsupported format for {path}")
