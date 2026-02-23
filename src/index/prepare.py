"""Build lightweight processing index from catalog."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


def _file_group(path: str) -> str:
    parts = Path(path).parts
    return parts[0] if parts else "root"


def _infer_format(ext: str) -> str:
    value = (ext or "").strip().lower().lstrip(".")
    return value if value else "unknown"


def _record_id(path: str) -> str:
    # deterministic uuid for resume/reproducibility
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"phi-otdr::{path}"))


def build_processing_index(catalog_path: str | Path, index_path: str | Path) -> Path:
    """Create processing index parquet from file-level catalog parquet."""

    catalog = Path(catalog_path)
    out = Path(index_path)

    if not catalog.exists():
        raise FileNotFoundError(f"catalog not found: {catalog}")

    if catalog.suffix.lower() == ".csv":
        df = pd.read_csv(catalog)
    else:
        df = pd.read_parquet(catalog)

    if "path" not in df.columns:
        raise ValueError("catalog must include 'path' column")

    work = df.copy()
    work["path"] = work["path"].astype(str)
    if "abs_path" in work.columns:
        work["abs_path"] = work["abs_path"].astype(str)
    else:
        work["abs_path"] = work["path"].map(lambda p: str((Path("data/raw") / str(p)).absolute()))
    work["size_bytes"] = pd.to_numeric(work.get("size_bytes", 0), errors="coerce").fillna(0).astype("int64")
    work["ext"] = work.get("ext", "").fillna("").astype(str)

    has_metadata_col = work.get("metadata_detected")
    if has_metadata_col is None:
        has_metadata = pd.Series(False, index=work.index)
    else:
        has_metadata = has_metadata_col.fillna(False).astype(bool)

    out_df = pd.DataFrame(
        {
            "record_id": work["path"].map(_record_id),
            "path": work["path"],
            "abs_path": work["abs_path"],
            "file_group": work["path"].map(_file_group),
            "file_size_mb": work["size_bytes"] / (1024.0 * 1024.0),
            "size_bytes": work["size_bytes"],
            "inferred_format": work["ext"].map(_infer_format),
            "has_metadata": has_metadata,
            "mtime": pd.to_datetime(work.get("mtime"), errors="coerce"),
        }
    )

    out_df = out_df.sort_values(["file_group", "path"]).reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)
    LOGGER.info("processing index saved: %s (%d rows)", out, len(out_df))
    return out


def ensure_processing_index(
    catalog_path: str | Path = "data/interim/catalog.parquet",
    index_path: str | Path = "data/interim/index.parquet",
) -> Path:
    """Build processing index only if missing."""

    out = Path(index_path)
    if out.exists():
        LOGGER.info("processing index already exists: %s", out)
        return out
    return build_processing_index(catalog_path=catalog_path, index_path=index_path)
