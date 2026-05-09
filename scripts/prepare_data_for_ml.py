"""Convert data_for_ml/parsed_*.parquet (already-extracted reflectograms)
into the aligned.npz format that detect_from_aligned consumes.

Each parquet file is laid out as [n_distance_bins, n_traces] (rows=distance, columns=traces).
We transpose to [n_traces, n_bins], optionally cross-correlation align, save to
``data/processed_usb/parser_cache/records/data_for_ml/<file>/aligned.npz`` plus
a meta.json + waterfall_raw.png compatible with the rest of the pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

# Set up matplotlib cache before importing
_DEFAULT_CACHE_ROOT = Path("/Volumes/data/phi-OTDR/cache")
_cache_root = Path(os.environ.get("PHI_OTDR_CACHE_ROOT", str(_DEFAULT_CACHE_ROOT))).resolve()
_mpl_root = (_cache_root / ".mplconfig").resolve()
_mpl_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.parser.config import ParseConfig
from src.parser.templates import (
    align_traces_cc,
    estimate_residual_jitter,
    select_alignment_window,
    should_apply_alignment,
)
from src.utils.logging_config import setup_logging

LOG = logging.getLogger("data_for_ml.prepare")

# Default ADC sampling rate used across the project.
ADC_FS_HZ = 50_000_000.0
# Standard 1 ms between traces (1 kHz repetition) for the data_for_ml acquisition.
TRACE_PERIOD_SAMPLES = 50_000


def _list_files(raw_root: Path) -> list[Path]:
    files: list[Path] = []
    for p in (raw_root / "data_for_ml").glob("parsed_*.parquet"):
        if p.name.startswith("._"):
            continue
        files.append(p)
    return sorted(files)


def _is_done(out_root: Path, src: Path) -> bool:
    rdir = out_root / "records" / "data_for_ml" / src.stem
    return (rdir / "aligned.npz").exists() and (rdir / "meta.json").exists()


def _save_waterfall(aligned: np.ndarray, out_path: Path, max_rows: int = 1200, max_cols: int = 4000) -> None:
    n_traces, n_bins = aligned.shape
    row_step = max(1, n_traces // max_rows)
    col_step = max(1, n_bins // max_cols)
    img = aligned[::row_step, ::col_step]
    fig, ax = plt.subplots(figsize=(11, 6))
    img_p = ax.imshow(img, aspect="auto", cmap="jet", origin="lower",
                      extent=[0, n_bins, 0, n_traces])
    ax.set_xlabel("distance bin")
    ax.set_ylabel("trace #")
    fig.colorbar(img_p, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _load_traces(src: Path) -> np.ndarray:
    """Load parsed_*.parquet into a [n_traces, n_bins] float32 array.

    The on-disk layout is rows=distance_bin, cols=trace, so we transpose.
    """
    table = pq.read_table(src)
    # Read all columns at once via pandas to avoid per-column conversion overhead.
    df = table.to_pandas(zero_copy_only=False)
    arr = df.to_numpy(dtype=np.float32, copy=False)  # [n_bins, n_traces]
    return np.ascontiguousarray(arr.T)  # [n_traces, n_bins]


def _run_one(src: Path, out_root: Path, cfg: ParseConfig, do_align: bool) -> dict[str, Any]:
    t0 = time.time()
    rdir = out_root / "records" / "data_for_ml" / src.stem
    rdir.mkdir(parents=True, exist_ok=True)

    traces = _load_traces(src)
    n_traces, n_bins = traces.shape
    starts = np.arange(n_traces, dtype=np.int64) * TRACE_PERIOD_SAMPLES

    # Cross-correlation alignment is applied only when explicitly requested.
    if do_align:
        cc_start = select_alignment_window(traces, cfg) if cfg.auto_select_cc_window else int(cfg.cc_window_start)
        cfg_align = replace(cfg, cc_window_start=cc_start)
        before = estimate_residual_jitter(traces, cfg_align)
        aligned_cc, shifts = align_traces_cc(traces, cfg_align)
        after = estimate_residual_jitter(aligned_cc, cfg_align)
        apply = should_apply_alignment(
            before=before, after=after,
            traces_before=traces, traces_after=aligned_cc,
            start=int(cfg_align.cc_window_start),
            end=int(min(traces.shape[1], cfg_align.cc_window_start + cfg_align.cc_window_len)),
        )
        aligned = aligned_cc if apply else traces
        eff = after if apply else before
        cc_window_used = [int(cfg_align.cc_window_start), int(cfg_align.cc_window_len)]
    else:
        before = estimate_residual_jitter(traces, cfg)
        aligned = traces
        eff = before
        apply = False
        cc_window_used = [int(cfg.cc_window_start), int(cfg.cc_window_len)]

    # Save in float16 to match other records on disk (saves ~2x).
    aligned_out = aligned.astype(np.float16, copy=False)
    np.savez_compressed(
        rdir / "aligned.npz",
        aligned=aligned_out,
        starts=starts,
        trace_len=np.int32(n_bins),
        adc_fs_hz=np.float64(ADC_FS_HZ),
        source_path=str(src),
    )

    _save_waterfall(aligned, rdir / "waterfall_raw.png")

    meta = {
        "source_rel": f"data_for_ml/{src.name}",
        "source_abs": str(src.resolve()),
        "elapsed_sec": float(time.time() - t0),
        "n_traces": int(n_traces),
        "n_bins": int(n_bins),
        "trace_len": int(n_bins),
        "adc_fs_hz": float(ADC_FS_HZ),
        "trace_period_samples_assumed": int(TRACE_PERIOD_SAMPLES),
        "alignment_applied": bool(apply),
        "residual_before_abs_mean": float(before[0]),
        "residual_before_abs_p95": float(before[1]),
        "residual_after_abs_mean": float(eff[0]),
        "residual_after_abs_p95": float(eff[1]),
        "cc_window_start_used": cc_window_used[0],
        "cc_window_len_used": cc_window_used[1],
        "config": asdict(cfg),
    }
    (rdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare data_for_ml/ as aligned.npz records")
    p.add_argument("--raw-root", type=Path, default=Path("/Volumes/data/phi-OTDR/raw"))
    p.add_argument("--out-root", type=Path, default=Path("data/processed_usb/parser_cache"))
    p.add_argument("--align", action="store_true", help="Run optional CC alignment on top of already-parsed traces")
    p.add_argument("--max-shift", type=int, default=300)
    p.add_argument("--align-iters", type=int, default=3)
    p.add_argument("--align-decimation", type=int, default=2)
    p.add_argument("--limit", type=int, default=None, help="Process at most N files (smoke check)")
    p.add_argument("--force", action="store_true", help="Reprocess even if cache exists")
    p.add_argument("--log", type=Path, default=Path("logs/prepare_data_for_ml.log"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    setup_logging(args.log)
    LOG.info("prepare_data_for_ml started")

    raw_root = args.raw_root.resolve()
    out_root = args.out_root.resolve()
    cfg = ParseConfig(
        max_shift=args.max_shift,
        align_iters=args.align_iters,
        align_decimation=args.align_decimation,
        cc_window_start=500,
        cc_window_len=12_000,
        auto_select_cc_window=True,
    )

    files = _list_files(raw_root)
    if args.limit is not None:
        files = files[: args.limit]
    todo = files if args.force else [p for p in files if not _is_done(out_root, p)]
    LOG.info("Candidates=%d, todo=%d, skipped_done=%d", len(files), len(todo), len(files) - len(todo))
    if not todo:
        LOG.info("Nothing to do.")
        return 0

    manifest_ok = out_root / "manifest_ml_ok.jsonl"
    manifest_err = out_root / "manifest_ml_err.jsonl"

    ok = err = 0
    for src in tqdm(todo, desc="prepare-ml", unit="file"):
        try:
            meta = _run_one(src=src, out_root=out_root, cfg=cfg, do_align=args.align)
            _append_jsonl(manifest_ok, meta)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            row = {"source_rel": f"data_for_ml/{src.name}", "error": f"{type(exc).__name__}: {exc}"}
            _append_jsonl(manifest_err, row)
            LOG.exception("Failed: %s", src)
            err += 1

    LOG.info("Done: ok=%d err=%d total=%d", ok, err, len(todo))
    return 0 if err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
