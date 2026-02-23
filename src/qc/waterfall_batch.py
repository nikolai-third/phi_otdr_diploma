"""Batch waterfall generation for all records in dataset index."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import pandas as pd

from src.io.reader import open_record
from src.qc.report import QCParams, build_waterfall_figure, load_matrix


def run_batch(
    index_path: Path,
    outdir: Path,
    max_traces: int,
    max_range_bins: int,
    max_bytes: int,
    trace_len: int,
    distance_step_m: float,
    time_step_s: float,
    max_records: int | None,
) -> tuple[list[Path], Path]:
    """Generate waterfall plots for all eligible records."""

    index_df = pd.read_parquet(index_path)
    if index_df.empty:
        raise ValueError(f"Index is empty: {index_path}")

    params = QCParams(
        max_traces=max_traces,
        max_range_bins=max_range_bins,
        max_bytes=max_bytes,
        trace_len=trace_len,
        distance_step_m=distance_step_m,
        time_step_s=time_step_s,
    )

    outputs: list[Path] = []
    rows: list[dict[str, str]] = []

    records = index_df["record_id"].tolist()
    if max_records is not None:
        records = records[:max_records]

    for record_id in records:
        handle = None
        try:
            handle = open_record(record_id, index_path)
            matrix = load_matrix(handle, params)
            saved = build_waterfall_figure(matrix, outdir=outdir, record_id=record_id, params=params)
            outputs.extend(saved)
            rows.append({"record_id": record_id, "status": "ok", "detail": str(saved[0])})
            print(f"[ok] {record_id}")
        except Exception as exc:
            rows.append({"record_id": record_id, "status": "error", "detail": f"{type(exc).__name__}: {exc}"})
            print(f"[error] {record_id}: {exc}")
            print(traceback.format_exc(limit=1))
        finally:
            if handle is not None:
                handle.close()

    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "waterfall_batch_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    return outputs, summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.qc.waterfall_batch", description="Build waterfall plots for all records")
    parser.add_argument("--index", type=Path, default=Path("data/interim/dataset_index.parquet"), help="Dataset index parquet path")
    parser.add_argument("--outdir", type=Path, default=Path("reports/figures"), help="Output directory for waterfall figures")
    parser.add_argument("--max-traces", type=int, default=512)
    parser.add_argument("--max-range-bins", type=int, default=2048)
    parser.add_argument("--max-bytes", type=int, default=25_000_000)
    parser.add_argument("--trace-len", type=int, default=55_000)
    parser.add_argument("--distance-step-m", type=float, default=2.02)
    parser.add_argument("--time-step-s", type=float, default=0.02)
    parser.add_argument("--max-records", type=int, default=None, help="Optional cap for debugging")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    outputs, summary = run_batch(
        index_path=args.index,
        outdir=args.outdir,
        max_traces=args.max_traces,
        max_range_bins=args.max_range_bins,
        max_bytes=args.max_bytes,
        trace_len=args.trace_len,
        distance_step_m=args.distance_step_m,
        time_step_s=args.time_step_s,
        max_records=args.max_records,
    )

    for path in outputs:
        print(path)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
