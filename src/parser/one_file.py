"""CLI entrypoint for one-file reflectogram parsing."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.parser.config import ParseConfig
from src.parser.io import run_one_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.parser.one_file",
        description="Run reflectogram parser on one file (auto mode).",
    )
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("reports/figures/parser"))
    parser.add_argument("--max-samples", type=int, default=50_000_000)
    parser.add_argument("--max-traces", type=int, default=2_000)
    parser.add_argument("--adc-fs-hz", type=float, default=50_000_000.0)
    parser.add_argument("--raw-plot-points", type=int, default=1_000_000)
    parser.add_argument("--waterfall-cmap", type=str, default="jet")
    parser.add_argument("--waterfall-exp-alpha", type=float, default=4.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = ParseConfig(
        max_traces=args.max_traces,
        raw_plot_points=args.raw_plot_points,
        adc_fs_hz=args.adc_fs_hz,
        max_samples=args.max_samples,
        waterfall_cmap=args.waterfall_cmap,
        waterfall_exp_alpha=args.waterfall_exp_alpha,
    )

    metrics = run_one_file(path=args.file, outdir=args.outdir, cfg=cfg)
    for key, value in metrics.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
