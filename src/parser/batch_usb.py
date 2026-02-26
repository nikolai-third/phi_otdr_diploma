"""Batch parser run for USB-backed raw data.

Processes all parquet files under ``raw_root`` (except known exclusions),
saves aligned traces and raw (non-normalized) waterfall plots under
``out_root``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

_cache_root = (Path("data") / "interim" / ".cache").resolve()
_mpl_root = (Path("data") / "interim" / ".mplconfig").resolve()
_cache_root.mkdir(parents=True, exist_ok=True)
_mpl_root.mkdir(parents=True, exist_ok=True)
os.environ["XDG_CACHE_HOME"] = str(_cache_root)
os.environ["MPLCONFIGDIR"] = str(_mpl_root)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.parser.config import ParseConfig
from src.parser.core import collect_start_candidates, extract_with_fallbacks, infer_trace_len
from src.parser.io import read_data_stream
from src.parser.templates import align_traces_cc, estimate_residual_jitter, select_alignment_window
from src.utils.logging_config import setup_logging


LOG = logging.getLogger("parser.batch_usb")


def _list_candidates(raw_root: Path) -> list[Path]:
    files: list[Path] = []
    for p in raw_root.rglob("*.parquet"):
        # Skip AppleDouble sidecar files.
        if p.name.startswith("._"):
            continue
        parts_l = [x.lower() for x in p.parts]
        # data_for_ml already contains historical parsed artifacts.
        if "data_for_ml" in parts_l:
            continue
        if p.name.startswith("parsed_"):
            continue
        files.append(p)
    return sorted(files)


def _record_dir(out_root: Path, raw_root: Path, src: Path) -> Path:
    rel = src.relative_to(raw_root).with_suffix("")
    return out_root / "records" / rel


def _is_done(out_root: Path, raw_root: Path, src: Path) -> bool:
    rdir = _record_dir(out_root, raw_root, src)
    return (rdir / "aligned.npz").exists() and (rdir / "waterfall_raw.png").exists() and (rdir / "meta.json").exists()


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save_waterfall_raw(
    aligned: np.ndarray,
    starts: np.ndarray,
    trace_len: int,
    adc_fs_hz: float,
    out_path: Path,
    refractive_index: float = 1.468,
) -> None:
    c0 = 299_792_458.0
    dist_km = (np.arange(trace_len, dtype=np.float64) * (c0 / (2.0 * refractive_index * adc_fs_hz))) / 1000.0
    if len(starts) > 1:
        t_sec = (starts[: aligned.shape[0]] - starts[0]) / adc_fs_hz
        t_max = float(t_sec[-1])
    else:
        t_max = float(aligned.shape[0] / max(1.0, adc_fs_hz / max(1, trace_len)))

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(
        aligned,
        origin="lower",
        aspect="auto",
        extent=[float(dist_km[0]), float(dist_km[-1]), 0.0, t_max],
        cmap="jet",
    )
    fig.colorbar(im, ax=ax, label="Raw amplitude")
    ax.set_xlabel("Distance, km")
    ax.set_ylabel("Time, s")
    ax.set_title("Aligned reflectograms waterfall (no normalization)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _run_one(src: Path, raw_root: Path, out_root: Path, cfg: ParseConfig, save_dtype: str) -> dict[str, Any]:
    t0 = time.time()
    rdir = _record_dir(out_root, raw_root, src)
    rdir.mkdir(parents=True, exist_ok=True)

    data = read_data_stream(src, max_points=cfg.max_samples)
    raw_candidates = collect_start_candidates(data)
    trace_len = infer_trace_len(data, raw_candidates)
    starts, traces = extract_with_fallbacks(data, trace_len=trace_len, max_traces=cfg.max_traces)

    if cfg.auto_select_cc_window:
        cc_start = select_alignment_window(traces, cfg)
        cfg_align = replace(cfg, cc_window_start=cc_start)
    else:
        cfg_align = cfg

    residual_before = estimate_residual_jitter(traces, cfg_align)
    aligned_cc, shifts = align_traces_cc(traces, cfg_align)
    residual_after = estimate_residual_jitter(aligned_cc, cfg_align)
    apply_alignment = residual_after[0] < residual_before[0]
    aligned = aligned_cc if apply_alignment else traces

    if save_dtype == "float16":
        aligned_out = aligned.astype(np.float16, copy=False)
    else:
        aligned_out = aligned.astype(np.float32, copy=False)

    np.savez_compressed(
        rdir / "aligned.npz",
        aligned=aligned_out,
        starts=starts.astype(np.int64, copy=False),
        trace_len=np.int32(trace_len),
        adc_fs_hz=np.float64(cfg.adc_fs_hz),
        source_path=str(src),
    )

    _save_waterfall_raw(
        aligned=aligned,
        starts=starts,
        trace_len=int(trace_len),
        adc_fs_hz=float(cfg.adc_fs_hz),
        out_path=rdir / "waterfall_raw.png",
    )

    meta = {
        "source_rel": str(src.relative_to(raw_root)),
        "source_abs": str(src.resolve()),
        "elapsed_sec": float(time.time() - t0),
        "n_samples": int(len(data)),
        "n_detected_starts": int(len(starts)),
        "n_extracted_traces": int(traces.shape[0]),
        "trace_len": int(trace_len),
        "alignment_applied": bool(apply_alignment),
        "residual_before_abs_mean": float(residual_before[0]),
        "residual_before_abs_p95": float(residual_before[1]),
        "residual_after_abs_mean": float(residual_after[0]),
        "residual_after_abs_p95": float(residual_after[1]),
        "shift_abs_mean": float(np.mean(np.abs(shifts))),
        "cc_window_start_used": int(cfg_align.cc_window_start),
        "cc_window_len_used": int(cfg_align.cc_window_len),
        "config": asdict(cfg_align),
        "save_dtype": save_dtype,
    }
    (rdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.parser.batch_usb", description="Batch parse all parquet files to USB cache")
    p.add_argument("--raw-root", type=Path, default=Path("data/raw_usb"))
    p.add_argument("--out-root", type=Path, default=Path("data/processed_usb/parser_cache"))
    p.add_argument("--max-samples", type=int, default=50_000_000)
    p.add_argument("--max-traces", type=int, default=2_000)
    p.add_argument("--max-shift", type=int, default=450)
    p.add_argument("--align-iters", type=int, default=3)
    p.add_argument("--align-decimation", type=int, default=2)
    p.add_argument("--save-dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--limit", type=int, default=None, help="Optional limit for quick smoke run")
    p.add_argument("--log", type=Path, default=Path("logs/parser_batch_usb.log"))
    p.add_argument("--daemon", action="store_true", help="Run detached in background")
    p.add_argument("--daemon-out", type=Path, default=Path("logs/parser_batch_usb.out"))
    return p


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if "--daemon" in argv:
        out_path = Path("logs/parser_batch_usb.out")
        if "--daemon-out" in argv:
            idx = argv.index("--daemon-out")
            if idx + 1 < len(argv):
                out_path = Path(argv[idx + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        child_argv = [x for x in argv if x != "--daemon"]
        cmd = [sys.executable, "-m", "src.parser.batch_usb", *child_argv]
        with out_path.open("ab") as out:
            proc = subprocess.Popen(  # noqa: S603
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=os.environ.copy(),
            )
        print(proc.pid)
        return 0

    args = build_parser().parse_args(argv)
    setup_logging(args.log)
    LOG.info("Batch USB parser started")
    LOG.info("raw_root=%s out_root=%s", args.raw_root, args.out_root)

    raw_root = args.raw_root.resolve()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_ok = out_root / "manifest_ok.jsonl"
    manifest_err = out_root / "manifest_err.jsonl"

    cfg = ParseConfig(
        max_traces=args.max_traces,
        max_shift=args.max_shift,
        align_iters=args.align_iters,
        align_decimation=args.align_decimation,
        max_samples=args.max_samples,
        auto_select_cc_window=True,
    )

    all_files = _list_candidates(raw_root)
    if args.limit is not None:
        all_files = all_files[: args.limit]
    todo = [p for p in all_files if not _is_done(out_root, raw_root, p)]

    LOG.info("Candidates=%d, todo=%d, skipped_done=%d", len(all_files), len(todo), len(all_files) - len(todo))
    if not todo:
        LOG.info("Nothing to do.")
        return 0

    ok = 0
    err = 0
    for src in tqdm(todo, desc="batch-parse", unit="file"):
        try:
            meta = _run_one(src=src, raw_root=raw_root, out_root=out_root, cfg=cfg, save_dtype=args.save_dtype)
            _append_jsonl(manifest_ok, meta)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            row = {"source_rel": str(src.relative_to(raw_root)), "error": f"{type(exc).__name__}: {exc}"}
            _append_jsonl(manifest_err, row)
            LOG.exception("Failed: %s", src)
            err += 1

    LOG.info("Done: ok=%d err=%d total=%d", ok, err, len(todo))
    return 0 if err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
