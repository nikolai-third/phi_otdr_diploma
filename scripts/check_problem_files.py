"""Re-run alignment on the worst files from old manifest to check if current code already fixes them."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
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


def run_baseline(src: Path) -> dict:
    cfg = ParseConfig(max_shift=450, align_iters=3, align_decimation=2, cc_window_start=500, cc_window_len=12000, auto_select_cc_window=True, cc_scan_step=2000)
    t0 = time.time()
    data = read_data_stream(src)
    raw_candidates = collect_start_candidates(data)
    trace_len = infer_trace_len(data, raw_candidates)
    starts, traces = extract_with_fallbacks(data, trace_len=trace_len, max_traces=None)
    n_samples = len(data)
    n_extracted = int(traces.shape[0])
    cc_start = select_alignment_window(traces, cfg)
    cfg_align = replace(cfg, cc_window_start=cc_start)

    before = estimate_residual_jitter(traces, cfg_align)
    aligned, shifts = align_traces_cc(traces, cfg_align)
    after = estimate_residual_jitter(aligned, cfg_align)
    apply = should_apply_alignment(
        before=before, after=after,
        traces_before=traces, traces_after=aligned,
        start=int(cfg_align.cc_window_start),
        end=int(min(traces.shape[1], cfg_align.cc_window_start + cfg_align.cc_window_len)),
    )
    eff = after if apply else before
    return {
        "n_samples": n_samples,
        "trace_len": int(trace_len),
        "n_extracted": n_extracted,
        "cc_start": int(cc_start),
        "before_mean": float(before[0]),
        "before_p95": float(before[1]),
        "after_mean": float(after[0]),
        "after_p95": float(after[1]),
        "effective_mean": float(eff[0]),
        "apply_alignment": bool(apply),
        "elapsed_sec": float(time.time() - t0),
    }


def main() -> int:
    targets = [
        ("some_test/2024-12-28_13_37.parquet",          272.7, 209.1),
        ("08_10_2024/2024-10-08_08_45.parquet",         253.9, 171.6),
        ("измерение_возмущение/2024-10-15_14_41.parquet", 164.6, 154.0),
        ("some_test/2024-11-06_14_02.parquet",          307.2, 140.3),
        ("измерение_возмущение/2024-10-21_17_08.parquet", 150.0, 134.0),
        # also 2 from "alignment didn't apply with non-trivial before"
        ("30_09_2024/2024-09-30_08_45.parquet",         50.8, 0.0),  # group p90 case
    ]
    raw_root = Path("/Volumes/data/phi-OTDR/raw")
    print(f"{'file':<55}  {'n_extr':>6}  {'before':>8}  {'before_p95':>10}  {'after':>8}  {'after_p95':>9}  {'apply':>5}  {'old_after':>9}  {'time':>5}")
    print("-" * 140)
    for rel, old_before, old_after in targets:
        src = raw_root / rel
        if not src.exists():
            print(f"{rel:<55}  MISSING")
            continue
        try:
            r = run_baseline(src)
            print(f"{rel:<55}  {r['n_extracted']:>6}  {r['before_mean']:>8.2f}  {r['before_p95']:>10.2f}  {r['after_mean']:>8.2f}  {r['after_p95']:>9.2f}  {str(r['apply_alignment']):>5}  {old_after:>9.2f}  {r['elapsed_sec']:>5.1f}")
        except Exception as e:
            print(f"{rel:<55}  ERROR: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
