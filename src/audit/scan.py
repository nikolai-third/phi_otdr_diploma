"""Dataset audit scanner for phi-OTDR project."""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import os
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import zipfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

try:
    import h5py
except ImportError:  # pragma: no cover - dependency may be optional in some environments
    h5py = None  # type: ignore[assignment]

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - dependency may be optional in some environments
    class tqdm:  # type: ignore[no-redef]
        """Minimal fallback progress wrapper when tqdm is unavailable."""

        def __init__(self, iterable: Any = None, total: int | None = None, **_: Any) -> None:
            self.total = total
            self.count = 0
            self.iterable = iterable

        def __enter__(self) -> "tqdm":
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def update(self, n: int = 1) -> None:
            self.count += n

        def __iter__(self) -> Any:
            if self.iterable is None:
                return iter(())
            return iter(self.iterable)

LOGGER = logging.getLogger(__name__)
MAX_H5_DATASETS = 200
TEXT_PREVIEW_CHARS = 2_000
TEXT_SAMPLE_LINES = 20


@dataclass(frozen=True)
class ScanConfig:
    """Configuration for file scanning."""

    sample_bytes: int = 64
    qc_max_bytes: int = 5_000_000
    parquet_schema_max_bytes: int = 2_000_000


def _init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _iter_files(root: Path, max_files: int | None) -> Any:
    count = 0
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            yield Path(dirpath) / filename
            count += 1
            if max_files is not None and count >= max_files:
                return


def _npy_header_from_fp(fp: Any) -> tuple[tuple[int, ...], str]:
    version = np.lib.format.read_magic(fp)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(fp)
    else:
        shape, _, dtype = np.lib.format.read_array_header_2_0(fp)
    return tuple(int(dim) for dim in shape), str(dtype)


def _compute_qc(sample: np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(sample)
    flat = arr.ravel()
    if flat.size == 0:
        return {
            "qc_mean": np.nan,
            "qc_std": np.nan,
            "qc_min": np.nan,
            "qc_max": np.nan,
            "qc_fraction_zeros": np.nan,
            "qc_fraction_nans": np.nan,
            "qc_sample_count": 0,
        }

    is_float = np.issubdtype(flat.dtype, np.floating)
    flat_float = flat.astype(np.float64, copy=False)
    zero_fraction = float(np.mean(flat_float == 0.0))
    if is_float:
        nan_fraction = float(np.mean(np.isnan(flat_float)))
    else:
        nan_fraction = np.nan

    return {
        "qc_mean": float(np.nanmean(flat_float)),
        "qc_std": float(np.nanstd(flat_float)),
        "qc_min": float(np.nanmin(flat_float)),
        "qc_max": float(np.nanmax(flat_float)),
        "qc_fraction_zeros": zero_fraction,
        "qc_fraction_nans": nan_fraction,
        "qc_sample_count": int(flat.size),
    }


def _sample_npy(path: Path, qc_max_bytes: int) -> np.ndarray:
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    itemsize = max(1, int(arr.dtype.itemsize))
    max_elems = max(1, qc_max_bytes // itemsize)
    return arr.reshape(-1)[:max_elems]


def _sample_h5_dataset(dataset: h5py.Dataset, qc_max_bytes: int) -> np.ndarray:
    itemsize = max(1, int(dataset.dtype.itemsize))
    max_elems = max(1, qc_max_bytes // itemsize)

    if dataset.ndim == 0:
        return np.asarray([dataset[()]])

    slices: list[slice] = []
    remaining = max_elems
    for dim in dataset.shape:
        take = min(int(dim), max(1, remaining))
        slices.append(slice(0, take))
        remaining = max(1, remaining // max(1, take))

    sample = np.asarray(dataset[tuple(slices)]).ravel()[:max_elems]
    return sample


def _extract_h5_metadata(path: Path, config: ScanConfig) -> tuple[dict[str, Any], list[str]]:
    metadata: dict[str, Any] = {}
    warnings: list[str] = []
    datasets: list[dict[str, Any]] = []

    if h5py is None:
        warnings.append("h5py is not installed; HDF5 metadata skipped")
        metadata.update(_compute_qc(np.array([])))
        return metadata, warnings

    with h5py.File(path, "r") as h5_file:
        root_attrs = {key: str(value) for key, value in h5_file.attrs.items()}
        truncated = False

        def visitor(name: str, obj: Any) -> None:
            nonlocal truncated
            if isinstance(obj, h5py.Dataset):
                if len(datasets) >= MAX_H5_DATASETS:
                    truncated = True
                    return True
                datasets.append(
                    {
                        "name": name,
                        "shape": tuple(int(x) for x in obj.shape),
                        "dtype": str(obj.dtype),
                    }
                )
            return None

        h5_file.visititems(visitor)
        metadata["root_attrs"] = root_attrs
        metadata["datasets"] = datasets
        metadata["dataset_count"] = len(datasets)
        metadata["dataset_list_truncated"] = truncated
        if truncated:
            warnings.append(f"HDF5 dataset list truncated at {MAX_H5_DATASETS} entries")

        qc_done = False
        for ds_meta in datasets:
            ds = h5_file[ds_meta["name"]]
            if not isinstance(ds, h5py.Dataset):
                continue
            if ds.dtype.kind not in {"i", "u", "f", "b"}:
                continue
            try:
                metadata.update(_compute_qc(_sample_h5_dataset(ds, config.qc_max_bytes)))
                metadata["qc_dataset"] = ds_meta["name"]
                qc_done = True
                break
            except Exception as exc:  # pragma: no cover - defensive
                warnings.append(f"QC failed for {ds_meta['name']}: {exc}")

        if not qc_done:
            metadata.update(_compute_qc(np.array([])))

    return metadata, warnings


def _extract_npy_metadata(path: Path, config: ScanConfig) -> tuple[dict[str, Any], list[str]]:
    metadata: dict[str, Any] = {}
    warnings: list[str] = []

    with path.open("rb") as fp:
        shape, dtype = _npy_header_from_fp(fp)
    metadata["shape"] = shape
    metadata["dtype"] = dtype

    try:
        metadata.update(_compute_qc(_sample_npy(path, config.qc_max_bytes)))
    except Exception as exc:  # pragma: no cover - defensive
        warnings.append(f"QC failed: {exc}")
        metadata.update(_compute_qc(np.array([])))

    return metadata, warnings


def _extract_npz_metadata(path: Path) -> tuple[dict[str, Any], list[str]]:
    metadata: dict[str, Any] = {"arrays": []}
    warnings: list[str] = []

    with zipfile.ZipFile(path, "r") as zf:
        for member in zf.infolist():
            if not member.filename.endswith(".npy"):
                continue
            try:
                with zf.open(member.filename, "r") as fp:
                    shape, dtype = _npy_header_from_fp(fp)
                metadata["arrays"].append(
                    {"name": member.filename, "shape": shape, "dtype": dtype}
                )
            except Exception as exc:
                warnings.append(f"Could not parse {member.filename}: {exc}")

    metadata["array_count"] = len(metadata["arrays"])
    metadata.update(_compute_qc(np.array([])))
    return metadata, warnings


def _extract_json_metadata(path: Path, parse_limit: int) -> tuple[dict[str, Any], list[str]]:
    metadata: dict[str, Any] = {}
    warnings: list[str] = []

    if path.stat().st_size > parse_limit:
        warnings.append(f"JSON parse skipped: file larger than {parse_limit} bytes")
        return metadata, warnings

    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    if isinstance(payload, dict):
        metadata["json_keys"] = list(payload.keys())
    elif isinstance(payload, list):
        metadata["json_type"] = "list"
        metadata["json_length"] = len(payload)
        if payload and isinstance(payload[0], dict):
            metadata["json_item_keys"] = list(payload[0].keys())
    else:
        metadata["json_type"] = type(payload).__name__

    return metadata, warnings


def _extract_yaml_metadata(path: Path, parse_limit: int) -> tuple[dict[str, Any], list[str]]:
    metadata: dict[str, Any] = {}
    warnings: list[str] = []

    if path.stat().st_size > parse_limit:
        warnings.append(f"YAML parse skipped: file larger than {parse_limit} bytes")
        return metadata, warnings

    with path.open("r", encoding="utf-8") as fp:
        payload = yaml.safe_load(fp)

    if isinstance(payload, dict):
        metadata["yaml_keys"] = list(payload.keys())
    elif isinstance(payload, list):
        metadata["yaml_type"] = "list"
        metadata["yaml_length"] = len(payload)

    return metadata, warnings


def _extract_csv_metadata(path: Path) -> tuple[dict[str, Any], list[str]]:
    metadata: dict[str, Any] = {}
    warnings: list[str] = []

    columns = pd.read_csv(path, nrows=0).columns.tolist()
    metadata["columns"] = columns

    return metadata, warnings


def _extract_parquet_metadata(
    path: Path, parquet_schema_max_bytes: int
) -> tuple[dict[str, Any], list[str]]:
    metadata: dict[str, Any] = {}
    warnings: list[str] = []

    if path.stat().st_size > parquet_schema_max_bytes:
        warnings.append(
            f"Parquet schema skipped: file larger than {parquet_schema_max_bytes} bytes"
        )
        return metadata, warnings

    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    metadata["columns"] = schema.names
    metadata["column_types"] = [str(schema.field(name).type) for name in schema.names]
    metadata["num_row_groups"] = parquet.num_row_groups

    return metadata, warnings


def _decode_text_sample(raw: bytes) -> tuple[str, str, float]:
    """Decode raw text bytes with a best-effort encoding guess."""

    encodings = ("utf-8-sig", "cp1251", "cp866", "koi8-r", "utf-16", "latin-1")
    best_text = ""
    best_encoding = "unknown"
    best_score = -1.0

    for encoding in encodings:
        try:
            decoded = raw.decode(encoding, errors="replace")
        except Exception:
            continue
        replacement_ratio = decoded.count("\ufffd") / max(1, len(decoded))
        printable_ratio = sum(ch.isprintable() or ch in "\r\n\t" for ch in decoded) / max(
            1, len(decoded)
        )
        score = printable_ratio - replacement_ratio
        if score > best_score:
            best_text = decoded
            best_encoding = encoding
            best_score = score

    return best_encoding, best_text, best_score


def _extract_txt_metadata(path: Path, parse_limit: int) -> tuple[dict[str, Any], list[str]]:
    metadata: dict[str, Any] = {}
    warnings: list[str] = []

    read_bytes = min(parse_limit, max(4_096, TEXT_PREVIEW_CHARS * 4))
    with path.open("rb") as fp:
        raw = fp.read(read_bytes)

    encoding, decoded, decode_score = _decode_text_sample(raw)
    if not decoded:
        warnings.append("TXT decode failed")
        return metadata, warnings

    lines = [line.strip() for line in decoded.splitlines() if line.strip()]
    kv_pairs: dict[str, str] = {}
    for line in lines[:TEXT_SAMPLE_LINES]:
        for sep in (":", "=", "\t"):
            if sep in line:
                key, value = line.split(sep, 1)
                key = key.strip()
                value = value.strip()
                if key:
                    kv_pairs[key] = value
                break

    metadata["text_encoding"] = encoding
    metadata["text_decode_score"] = round(float(decode_score), 4)
    metadata["text_line_count_sample"] = len(lines[:TEXT_SAMPLE_LINES])
    metadata["text_preview"] = decoded[:TEXT_PREVIEW_CHARS]
    if kv_pairs:
        metadata["text_kv_pairs"] = kv_pairs

    if path.stat().st_size > read_bytes:
        warnings.append(f"TXT partially parsed: sampled first {read_bytes} bytes only")

    return metadata, warnings


def _extract_metadata(
    path: Path, ext: str, config: ScanConfig, size_bytes: int
) -> tuple[dict[str, Any], list[str]]:
    parse_limit = min(config.qc_max_bytes, 2_000_000)

    if ext in {".h5", ".hdf5"}:
        return _extract_h5_metadata(path, config)
    if ext == ".npy":
        return _extract_npy_metadata(path, config)
    if ext == ".npz":
        return _extract_npz_metadata(path)
    if ext == ".json":
        return _extract_json_metadata(path, parse_limit)
    if ext in {".yaml", ".yml"}:
        return _extract_yaml_metadata(path, parse_limit)
    if ext == ".csv":
        return _extract_csv_metadata(path)
    if ext == ".parquet":
        return _extract_parquet_metadata(path, config.parquet_schema_max_bytes)
    if ext == ".txt":
        return _extract_txt_metadata(path, parse_limit)

    return {}, []


def _scan_file(path: Path, root: Path, config: ScanConfig) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": str(path.relative_to(root)),
        "abs_path": str(path.absolute()),
        "size_bytes": np.nan,
        "mtime": pd.NaT,
        "ext": path.suffix.lower(),
        "is_symlink": path.is_symlink(),
        "is_file": path.is_file(),
        "metadata_detected": False,
        "file_type_guess": None,
        "magic_hex": None,
        "metadata_json": "{}",
        "warning": None,
        "error": None,
        "qc_mean": np.nan,
        "qc_std": np.nan,
        "qc_min": np.nan,
        "qc_max": np.nan,
        "qc_fraction_zeros": np.nan,
        "qc_fraction_nans": np.nan,
        "qc_sample_count": np.nan,
    }

    warnings: list[str] = []

    try:
        stat = path.stat()
        row["size_bytes"] = int(stat.st_size)
        row["mtime"] = datetime.fromtimestamp(stat.st_mtime)
        row["file_type_guess"] = mimetypes.guess_type(path.name)[0]

        with path.open("rb") as fp:
            row["magic_hex"] = fp.read(config.sample_bytes).hex()

        metadata, meta_warnings = _extract_metadata(path, row["ext"], config, int(row["size_bytes"]))
        warnings.extend(meta_warnings)

        if metadata:
            row["metadata_detected"] = True
            row["metadata_json"] = json.dumps(metadata, ensure_ascii=True, default=str)

            for qc_key in (
                "qc_mean",
                "qc_std",
                "qc_min",
                "qc_max",
                "qc_fraction_zeros",
                "qc_fraction_nans",
                "qc_sample_count",
            ):
                if qc_key in metadata:
                    row[qc_key] = metadata[qc_key]

    except Exception as exc:  # pragma: no cover - defensive
        row["error"] = str(exc)

    if warnings:
        row["warning"] = " | ".join(warnings)

    return row


def _build_summary(df: pd.DataFrame) -> str:
    file_count = int(len(df))
    total_size = int(df["size_bytes"].fillna(0).sum())
    top_ext = (
        df["ext"].fillna("(none)").value_counts().head(10).rename_axis("ext").reset_index(name="count")
    )
    top_big = (
        df[["path", "size_bytes"]]
        .sort_values("size_bytes", ascending=False)
        .head(20)
        .fillna({"size_bytes": 0})
    )
    mtime_series = pd.to_datetime(df["mtime"], errors="coerce")
    min_date = mtime_series.min()
    max_date = mtime_series.max()
    detected_ratio = float(df["metadata_detected"].mean()) if file_count > 0 else 0.0
    issue_count = int(df["error"].notna().sum() + df["warning"].notna().sum())

    lines = [
        "# Dataset Audit Summary",
        "",
        f"- Files scanned: **{file_count}**",
        f"- Total size (bytes): **{total_size}**",
        f"- Metadata detected ratio: **{detected_ratio:.2%}**",
        f"- Files with warnings/errors: **{issue_count}**",
        f"- Date range (mtime): **{min_date}** .. **{max_date}**",
        "",
        "## Top Extensions",
        "",
    ]

    if not top_ext.empty:
        lines.append("| ext | count |")
        lines.append("|---|---:|")
        for row in top_ext.itertuples(index=False):
            lines.append(f"| {row.ext} | {row.count} |")
    else:
        lines.append("No files found.")

    lines.extend(["", "## Top 20 Largest Files", ""])

    if not top_big.empty:
        lines.append("| path | size_bytes |")
        lines.append("|---|---:|")
        for row in top_big.itertuples(index=False):
            lines.append(f"| {row.path} | {int(row.size_bytes)} |")

    if issue_count > 0:
        lines.extend(["", "## Problems", ""])
        issue_df = df[df["error"].notna() | df["warning"].notna()][["path", "warning", "error"]]
        for row in issue_df.head(100).itertuples(index=False):
            lines.append(f"- `{row.path}` | warning: {row.warning} | error: {row.error}")

    lines.append("")
    return "\n".join(lines)


def run_scan(
    root: Path,
    out: Path,
    max_files: int | None,
    sample_bytes: int,
    qc_max_bytes: int,
    parquet_schema_max_bytes: int,
    workers: int,
) -> tuple[Path, Path, Path]:
    """Run dataset audit and write parquet/csv/markdown outputs."""

    if not root.exists():
        raise FileNotFoundError(f"Root path does not exist: {root}")

    out.parent.mkdir(parents=True, exist_ok=True)
    config = ScanConfig(
        sample_bytes=sample_bytes,
        qc_max_bytes=qc_max_bytes,
        parquet_schema_max_bytes=parquet_schema_max_bytes,
    )

    records: list[dict[str, Any]] = []
    pending: dict[Future[dict[str, Any]], Path] = {}
    max_pending = max(workers * 4, 8)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        with tqdm(total=max_files, unit="file", desc="Scanning") as pbar:
            for path in _iter_files(root, max_files=max_files):
                while len(pending) >= max_pending:
                    done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                    for future in done:
                        records.append(future.result())
                        pending.pop(future, None)
                        pbar.update(1)

                future = executor.submit(_scan_file, path, root, config)
                pending[future] = path

            if pending:
                for future in wait(pending.keys()).done:
                    records.append(future.result())
                    pbar.update(1)

    df = pd.DataFrame(records)
    if not df.empty:
        df["mtime"] = pd.to_datetime(df["mtime"], errors="coerce")

    df.to_parquet(out, index=False)
    csv_path = out.with_suffix(".csv")
    df.to_csv(csv_path, index=False)

    summary_path = out.parent / "summary.md"
    summary_path.write_text(_build_summary(df), encoding="utf-8")

    LOGGER.info("Saved parquet: %s", out)
    LOGGER.info("Saved csv: %s", csv_path)
    LOGGER.info("Saved summary: %s", summary_path)

    return out, csv_path, summary_path


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for audit tools."""

    parser = argparse.ArgumentParser(prog="python -m src.audit", description="Audit dataset files")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="Scan data root and build file catalog")
    scan_parser.add_argument("--root", type=Path, default=Path("data/raw"), help="Root directory to scan")
    scan_parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/interim/catalog.parquet"),
        help="Parquet output path",
    )
    scan_parser.add_argument("--max-files", type=int, default=None, help="Maximum number of files to scan")
    scan_parser.add_argument("--sample-bytes", type=int, default=64, help="Bytes to read for magic header")
    scan_parser.add_argument(
        "--qc-max-bytes",
        type=int,
        default=5_000_000,
        help="Max bytes allowed for QC sample",
    )
    scan_parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads for metadata extraction",
    )
    scan_parser.add_argument(
        "--parquet-schema-max-bytes",
        type=int,
        default=2_000_000,
        help="Skip parquet schema read for files larger than this limit",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for audit module."""

    _init_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "scan":
        run_scan(
            root=args.root,
            out=args.out,
            max_files=args.max_files,
            sample_bytes=args.sample_bytes,
            qc_max_bytes=args.qc_max_bytes,
            parquet_schema_max_bytes=args.parquet_schema_max_bytes,
            workers=args.workers,
        )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
