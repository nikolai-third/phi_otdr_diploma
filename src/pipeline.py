"""Top-level pipeline CLI."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.baseline.artifacts import build_artifacts
from src.baseline.run import run_baseline
from src.index.prepare import ensure_processing_index
from src.utils.logging_config import setup_logging

LOGGER = logging.getLogger(__name__)


def _ensure_dirs() -> None:
    for path in [
        Path("cache"),
        Path("logs"),
        Path("data/processed"),
        Path("reports/figures"),
        Path("reports/tables"),
    ]:
        path.mkdir(parents=True, exist_ok=True)


def run_all(args: argparse.Namespace) -> int:
    setup_logging("logs/pipeline.log", level=logging.INFO)
    _ensure_dirs()

    LOGGER.info("run-all started")
    index_path = ensure_processing_index(catalog_path=args.catalog, index_path=args.index)

    metrics_path = run_baseline(
        index_path=index_path,
        cache_dir=args.cache_dir,
        metrics_path=args.metrics,
        errors_path=args.errors,
        max_workers=args.max_workers,
        max_bytes=args.max_bytes,
        checkpoint_every=args.checkpoint_every,
        progress_log_minutes=10,
    )

    summary_path = build_artifacts(
        metrics_path=metrics_path,
        figures_dir=args.figures,
        tables_dir=args.tables,
        summary_path=args.summary,
    )

    LOGGER.info("run-all completed")
    LOGGER.info("index: %s", index_path)
    LOGGER.info("metrics: %s", metrics_path)
    LOGGER.info("summary: %s", summary_path)
    print("Pipeline finished")
    print(f"index: {index_path}")
    print(f"metrics: {metrics_path}")
    print(f"summary: {summary_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.pipeline", description="Run full phi-OTDR baseline pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run-all", help="Run index + baseline + artifact generation")
    p.add_argument("--catalog", type=Path, default=Path("data/interim/catalog.parquet"))
    p.add_argument("--index", type=Path, default=Path("data/interim/index.parquet"))
    p.add_argument("--cache-dir", type=Path, default=Path("cache"))
    p.add_argument("--metrics", type=Path, default=Path("data/processed/baseline_metrics.parquet"))
    p.add_argument("--errors", type=Path, default=Path("logs/errors.log"))
    p.add_argument("--figures", type=Path, default=Path("reports/figures"))
    p.add_argument("--tables", type=Path, default=Path("reports/tables"))
    p.add_argument("--summary", type=Path, default=Path("reports/summary_baseline.md"))
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--max-bytes", type=int, default=5_000_000)
    p.add_argument("--checkpoint-every", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run-all":
        return run_all(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
