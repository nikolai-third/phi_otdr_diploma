"""Sweep alignment parameters on a single problem file to find better config."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.parser.config import ParseConfig
from src.parser.core import collect_start_candidates, extract_with_fallbacks, infer_trace_len
from src.parser.io import read_data_stream
from src.parser.templates import (
    align_traces_cc,
    estimate_residual_jitter,
    select_alignment_window,
    should_apply_alignment,
)


def run(traces: np.ndarray, cfg: ParseConfig) -> dict:
    t0 = time.time()
    if cfg.auto_select_cc_window:
        cc_start = select_alignment_window(traces, cfg)
        cfg_align = replace(cfg, cc_window_start=cc_start)
    else:
        cfg_align = cfg

    before = estimate_residual_jitter(traces, cfg_align)
    aligned, shifts = align_traces_cc(traces, cfg_align)
    after = estimate_residual_jitter(aligned, cfg_align)
    apply = should_apply_alignment(
        before=before, after=after,
        traces_before=traces, traces_after=aligned,
        start=int(cfg_align.cc_window_start),
        end=int(min(traces.shape[1], cfg_align.cc_window_start + cfg_align.cc_window_len)),
    )
    return {
        "cc_window": [int(cfg_align.cc_window_start), int(cfg_align.cc_window_start + cfg_align.cc_window_len)],
        "before_mean": float(before[0]),
        "before_p95": float(before[1]),
        "after_mean": float(after[0]),
        "after_p95": float(after[1]),
        "apply_alignment": bool(apply),
        "elapsed_sec": float(time.time() - t0),
        "max_abs_shift": int(np.max(np.abs(shifts))),
        "shift_std": float(np.std(shifts)),
    }


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/Volumes/data/phi-OTDR/raw/some_test/2024-12-28_13_37.parquet")
    print(f"file: {src}")
    print("loading...")
    t0 = time.time()
    data = read_data_stream(src)
    raw_candidates = collect_start_candidates(data)
    trace_len = infer_trace_len(data, raw_candidates)
    starts, traces = extract_with_fallbacks(data, trace_len=trace_len, max_traces=None)
    print(f"  loaded in {time.time()-t0:.1f}s, n_samples={len(data):_}, traces={traces.shape}, trace_len={trace_len}")

    configs = [
        ("baseline (current)",      ParseConfig(max_shift=450, align_iters=3, align_decimation=2, cc_window_start=500, cc_window_len=12000, auto_select_cc_window=True, cc_scan_step=2000)),
        ("more_iters",              ParseConfig(max_shift=450, align_iters=6, align_decimation=2, cc_window_start=500, cc_window_len=12000, auto_select_cc_window=True, cc_scan_step=2000)),
        ("wider_window",            ParseConfig(max_shift=450, align_iters=3, align_decimation=2, cc_window_start=500, cc_window_len=24000, auto_select_cc_window=True, cc_scan_step=2000)),
        ("more_iters+wider_window", ParseConfig(max_shift=450, align_iters=6, align_decimation=2, cc_window_start=500, cc_window_len=24000, auto_select_cc_window=True, cc_scan_step=2000)),
        ("max_shift_900+iters6",    ParseConfig(max_shift=900, align_iters=6, align_decimation=2, cc_window_start=500, cc_window_len=24000, auto_select_cc_window=True, cc_scan_step=2000)),
        ("dec1_full_res+iters6",    ParseConfig(max_shift=450, align_iters=6, align_decimation=1, cc_window_start=500, cc_window_len=12000, auto_select_cc_window=True, cc_scan_step=2000)),
        ("finer_scan",              ParseConfig(max_shift=450, align_iters=6, align_decimation=2, cc_window_start=500, cc_window_len=12000, auto_select_cc_window=True, cc_scan_step=500)),
        ("fixed_early_window",      ParseConfig(max_shift=450, align_iters=6, align_decimation=2, cc_window_start=2000, cc_window_len=12000, auto_select_cc_window=False)),
    ]
    print(f"\n{'name':<28}  {'cc_window':<14}  {'before':>8}  {'after':>8}  {'p95_aft':>8}  {'apply':>5}  {'time':>5}  {'max|s|':>6}")
    print("-" * 120)
    for name, cfg in configs:
        res = run(traces, cfg)
        cc = f"[{res['cc_window'][0]}:{res['cc_window'][1]}]"
        print(f"{name:<28}  {cc:<14}  {res['before_mean']:>8.2f}  {res['after_mean']:>8.2f}  {res['after_p95']:>8.2f}  {str(res['apply_alignment']):>5}  {res['elapsed_sec']:>5.1f}  {res['max_abs_shift']:>6d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
