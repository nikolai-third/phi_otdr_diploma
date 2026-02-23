"""Build dataset index from audit catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroupKey:
    """Grouping key for dataset record aggregation."""

    top_dir: str


def _init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _safe_load_json(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, str) or not payload.strip():
        return {}
    try:
        value = json.loads(payload)
        if isinstance(value, dict):
            return value
    except Exception:
        return {}
    return {}


def _top_dir(rel_path: str) -> str:
    parts = Path(rel_path).parts
    if not parts:
        return "root"
    return parts[0]


def _infer_format(ext_counts: dict[str, int]) -> str:
    priority = [".npy", ".npz", ".h5", ".hdf5", ".parquet", ".csv", ".json", ".txt"]
    for ext in priority:
        if ext_counts.get(ext, 0) > 0:
            return ext.lstrip(".")
    if ext_counts:
        return max(ext_counts.items(), key=lambda item: item[1])[0].lstrip(".")
    return "unknown"


def _build_notes(meta_samples: list[dict[str, Any]]) -> str:
    notes: list[str] = []
    for meta in meta_samples:
        if "text_encoding" in meta:
            notes.append(f"txt:{meta.get('text_encoding')}")
        if "dataset_count" in meta:
            notes.append(f"h5_datasets={meta.get('dataset_count')}")
        if "shape" in meta:
            notes.append(f"shape={meta.get('shape')}")
        if "array_count" in meta:
            notes.append(f"npz_arrays={meta.get('array_count')}")
    if not notes:
        return "grouped by top-level directory"
    # deterministic and compact notes
    unique_notes = sorted(set(notes))
    return "; ".join(unique_notes[:8])


def _deterministic_record_id(key: GroupKey, paths: list[str]) -> str:
    canonical = f"{key.top_dir}|{len(paths)}|{min(paths)}|{max(paths)}"
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]
    return f"rec_{digest}"


def build_index(catalog: Path, out: Path, group_by: str = "topdir") -> Path:
    """Build dataset index parquet from file-level audit catalog."""

    if not catalog.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog}")

    df = pd.read_parquet(catalog)
    if df.empty:
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            columns=[
                "record_id",
                "file_count",
                "total_size_bytes",
                "paths",
                "abs_paths",
                "ext_counts",
                "mtime_min",
                "mtime_max",
                "inferred_format",
                "notes",
            ]
        ).to_parquet(out, index=False)
        return out

    for col in ("path", "abs_path", "ext", "size_bytes", "mtime", "metadata_json"):
        if col not in df.columns:
            if col == "metadata_json":
                df[col] = "{}"
            elif col == "mtime":
                df[col] = pd.NaT
            else:
                raise ValueError(f"Catalog missing required column: {col}")

    df = df.copy()
    df["path"] = df["path"].astype(str)
    df["abs_path"] = df["abs_path"].astype(str)
    df["ext"] = df["ext"].fillna("").astype(str).str.lower()
    df["size_bytes"] = pd.to_numeric(df["size_bytes"], errors="coerce").fillna(0).astype(np.int64)
    df["mtime"] = pd.to_datetime(df["mtime"], errors="coerce")
    if group_by not in {"topdir", "file"}:
        raise ValueError(f"Unsupported group_by={group_by}. Expected one of: topdir,file")

    if group_by == "topdir":
        df["group_key"] = df["path"].map(_top_dir)
    else:
        df["group_key"] = df["path"]

    rows: list[dict[str, Any]] = []
    grouped = df.sort_values("path").groupby("group_key", sort=True)

    for key_str, group in grouped:
        key = GroupKey(top_dir=str(key_str))
        paths = group["path"].tolist()
        abs_paths = group["abs_path"].tolist()

        ext_counts_series = group["ext"].value_counts().sort_index()
        ext_counts = {str(ext): int(cnt) for ext, cnt in ext_counts_series.items() if str(ext)}

        meta_samples = [_safe_load_json(item) for item in group["metadata_json"].head(10).tolist()]
        mtime = pd.to_datetime(group["mtime"], errors="coerce")

        row = {
            "record_id": _deterministic_record_id(key, paths),
            "file_count": int(len(group)),
            "total_size_bytes": int(group["size_bytes"].sum()),
            "paths": paths,
            "abs_paths": abs_paths,
            "ext_counts": ext_counts,
            "mtime_min": mtime.min(),
            "mtime_max": mtime.max(),
            "inferred_format": _infer_format(ext_counts),
            "notes": f"group_by={group_by}; {_build_notes(meta_samples)}",
        }
        rows.append(row)

    index_df = pd.DataFrame(rows).sort_values(["record_id"]).reset_index(drop=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_parquet(out, index=False)
    LOGGER.info("Saved dataset index: %s", out)
    LOGGER.info("Records: %d", len(index_df))
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.index", description="Build grouped dataset index")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build dataset-level index from file catalog")
    p_build.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/interim/catalog.parquet"),
        help="Input file-level catalog parquet",
    )
    p_build.add_argument(
        "--out",
        type=Path,
        default=Path("data/interim/dataset_index.parquet"),
        help="Output dataset index parquet",
    )
    p_build.add_argument(
        "--group-by",
        choices=["topdir", "file"],
        default="topdir",
        help="Grouping strategy for record construction",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _init_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "build":
        build_index(args.catalog, args.out, group_by=args.group_by)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
