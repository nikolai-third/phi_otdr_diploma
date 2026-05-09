"""Evaluate the post-alignment detector on labeled data_for_ml records.

Loads ground truth from ``data_for_ml/data_description.json`` (event_start/end
in distance-bin indices), runs the existing detector CLI on every labeled
record, then aggregates precision / recall / F1 / localization error.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# matplotlib config dir (avoid spam in stderr)
_DEFAULT_CACHE_ROOT = Path("/Volumes/data/phi-OTDR/cache")
_cache_root = Path(os.environ.get("PHI_OTDR_CACHE_ROOT", str(_DEFAULT_CACHE_ROOT))).resolve()
_mpl_root = (_cache_root / ".mplconfig").resolve()
_mpl_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_root))

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.parser import detect_from_aligned as detector
from src.utils.logging_config import setup_logging

LOG = logging.getLogger("eval_detector")

ADC_FS_HZ = 50_000_000.0
N_FIBER = 1.468
C_LIGHT = 299_792_458.0
DIST_PER_BIN_KM = (C_LIGHT / (2.0 * N_FIBER * ADC_FS_HZ)) / 1000.0  # ~0.00204 km/bin


def bin_to_km(b: int) -> float:
    return float(b) * DIST_PER_BIN_KM


def run_detector_on(
    npz_path: Path,
    outdir: Path,
    max_detections: int = 8,
    ignore_start_km: float = 1.0,
    threshold_k: float | None = None,
    peak_threshold_k: float | None = None,
    freq_min_hz: float | None = None,
    freq_max_hz: float | None = None,
    min_sep_km: float | None = None,
    score_mode: str | None = None,
) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    argv = [
        "--aligned-npz", str(npz_path),
        "--outdir", str(outdir),
        "--max-detections", str(max_detections),
        "--ignore-start-km", str(ignore_start_km),
    ]
    if threshold_k is not None:
        argv += ["--threshold-k", str(threshold_k)]
    if peak_threshold_k is not None:
        argv += ["--peak-threshold-k", str(peak_threshold_k)]
    if freq_min_hz is not None:
        argv += ["--freq-min-hz", str(freq_min_hz)]
    if freq_max_hz is not None:
        argv += ["--freq-max-hz", str(freq_max_hz)]
    if min_sep_km is not None:
        argv += ["--min-sep-km", str(min_sep_km)]
    if score_mode is not None:
        argv += ["--score-mode", score_mode]
    detector.main(argv)
    summary = json.loads((outdir / "detection_summary.json").read_text(encoding="utf-8"))
    return summary


def evaluate_file(
    rec_dir: Path,
    out_root: Path,
    label: dict | None,
    tolerance_km: float,
    ignore_start_km: float,
    threshold_k: float | None = None,
    peak_threshold_k: float | None = None,
    freq_min_hz: float | None = None,
    freq_max_hz: float | None = None,
    min_sep_km: float | None = None,
    score_mode: str | None = None,
) -> dict:
    """Run detector on one record and compute hit / FP info given (optional) ground truth."""
    npz_path = rec_dir / "aligned.npz"
    eval_outdir = out_root / rec_dir.name
    summary = run_detector_on(
        npz_path, eval_outdir,
        ignore_start_km=ignore_start_km,
        threshold_k=threshold_k, peak_threshold_k=peak_threshold_k,
        freq_min_hz=freq_min_hz, freq_max_hz=freq_max_hz,
        min_sep_km=min_sep_km, score_mode=score_mode,
    )

    detected = [d["distance_km"] for d in summary.get("detected", [])]
    top_usable = summary.get("top_candidate_within_usable")
    top_usable_km = top_usable["best_distance_km_within_usable"] if top_usable else None
    top_usable_score = top_usable["best_combined_score_within_usable"] if top_usable else None

    row = {
        "file": rec_dir.name,
        "n_bins": summary["n_bins"],
        "trace_len": summary["trace_len"],
        "n_traces": summary.get("stable_segment_len") or summary.get("n_groups"),
        "max_distance_km": summary["n_bins"] * DIST_PER_BIN_KM,
        "trace_rate_hz": summary["trace_rate_hz"],
        "combined_threshold": summary["combined_threshold"],
        "n_detected": len(detected),
        "detected_km": ";".join(f"{d:.3f}" for d in detected),
        "top_usable_km": top_usable_km,
        "top_usable_score": top_usable_score,
        "usable_end_km": summary.get("usable_end_km"),
        "ignore_start_km": summary.get("ignore_start_km"),
    }

    if label is None:
        row.update({
            "label": None,
            "event_start_km": None,
            "event_end_km": None,
            "hit_threshold": None,
            "hit_top_usable": None,
            "loc_err_threshold_km": None,
            "loc_err_top_usable_km": None,
            "n_false_positive": None,
        })
        return row

    if label.get("event") == 1:
        es_km = bin_to_km(label["event_start"])
        ee_km = bin_to_km(label["event_end"])
        ec_km = 0.5 * (es_km + ee_km)

        # Hit by threshold-based detection: any detected within tolerance of event
        in_zone = [
            d for d in detected
            if (es_km - tolerance_km) <= d <= (ee_km + tolerance_km)
        ]
        hit_thr = len(in_zone) > 0
        loc_err_thr = (min(abs(d - ec_km) for d in in_zone) if hit_thr else None)
        n_fp = len([d for d in detected if not ((es_km - tolerance_km) <= d <= (ee_km + tolerance_km))])

        # Hit by top-usable candidate
        if top_usable_km is not None:
            hit_top = (es_km - tolerance_km) <= top_usable_km <= (ee_km + tolerance_km)
            loc_err_top = abs(top_usable_km - ec_km)
        else:
            hit_top = False
            loc_err_top = None

        row.update({
            "label": "positive",
            "event_start_km": es_km,
            "event_end_km": ee_km,
            "hit_threshold": hit_thr,
            "hit_top_usable": hit_top,
            "loc_err_threshold_km": loc_err_thr,
            "loc_err_top_usable_km": loc_err_top,
            "n_false_positive": n_fp,
        })
    else:
        # Negative: any threshold detection is a FP; top-usable is informational only
        row.update({
            "label": "negative",
            "event_start_km": None,
            "event_end_km": None,
            "hit_threshold": None,
            "hit_top_usable": None,
            "loc_err_threshold_km": None,
            "loc_err_top_usable_km": None,
            "n_false_positive": len(detected),
        })

    # Pulse duration parsed from comment if available
    import re
    cm = label.get("comment", "")
    m = re.search(r"импульс[ыа]?\s+(\d+)\s*нс", cm)
    row["pulse_ns"] = int(m.group(1)) if m else None
    row["id_line"] = label.get("id_line")

    return row


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--description", type=Path, default=Path("/Volumes/data/phi-OTDR/raw/data_for_ml/data_description.json"))
    p.add_argument("--records-root", type=Path, default=Path("data/processed_usb/parser_cache/records/data_for_ml"))
    p.add_argument("--out-root", type=Path, default=Path("data/processed_usb/parser_cache/records/data_for_ml/_eval"))
    p.add_argument("--csv", type=Path, default=Path("reports/tables/data_for_ml_eval.csv"))
    p.add_argument("--md", type=Path, default=Path("reports/data_for_ml_eval_summary.md"))
    p.add_argument("--tolerance-km", type=float, default=0.5)
    p.add_argument("--ignore-start-km", type=float, default=1.0)
    p.add_argument("--threshold-k", type=float, default=None)
    p.add_argument("--peak-threshold-k", type=float, default=None)
    p.add_argument("--freq-min-hz", type=float, default=None)
    p.add_argument("--freq-max-hz", type=float, default=None)
    p.add_argument("--min-sep-km", type=float, default=None)
    p.add_argument("--score-mode", choices=["combined", "energy_only", "peak_only"], default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--log", type=Path, default=Path("logs/evaluate_detector.log"))
    args = p.parse_args(argv)
    setup_logging(args.log)

    description = json.loads(args.description.read_text(encoding="utf-8"))
    records_root = args.records_root.resolve()

    # gather labeled records that have aligned.npz prepared
    labeled = []
    for fname, lbl in description.items():
        rec_name = fname.replace(".parquet", "")
        rec_dir = records_root / rec_name
        if (rec_dir / "aligned.npz").exists():
            labeled.append((rec_dir, lbl))
    LOG.info("labeled records prepared: %d", len(labeled))
    if args.limit is not None:
        labeled = labeled[: args.limit]

    rows = []
    for rec_dir, lbl in tqdm(labeled, desc="evaluate", unit="file"):
        try:
            row = evaluate_file(
                rec_dir=rec_dir,
                out_root=args.out_root.resolve(),
                label=lbl,
                tolerance_km=args.tolerance_km,
                ignore_start_km=args.ignore_start_km,
                threshold_k=args.threshold_k,
                peak_threshold_k=args.peak_threshold_k,
                freq_min_hz=args.freq_min_hz,
                freq_max_hz=args.freq_max_hz,
                min_sep_km=args.min_sep_km,
                score_mode=args.score_mode,
            )
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            LOG.exception("eval failed: %s", rec_dir.name)
            rows.append({"file": rec_dir.name, "error": f"{type(exc).__name__}: {exc}"})

    df = pd.DataFrame(rows)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)
    LOG.info("wrote %s (%d rows)", args.csv, len(df))

    # Aggregate stats
    pos = df[df["label"] == "positive"]
    neg = df[df["label"] == "negative"]
    total_pos = len(pos)
    total_neg = len(neg)
    hits_thr = int(pos["hit_threshold"].fillna(False).sum())
    hits_top = int(pos["hit_top_usable"].fillna(False).sum())
    fp_thr = int(pos["n_false_positive"].fillna(0).sum() + neg["n_false_positive"].fillna(0).sum())

    recall_thr = hits_thr / total_pos if total_pos else float("nan")
    recall_top = hits_top / total_pos if total_pos else float("nan")
    precision_thr = (hits_thr / (hits_thr + fp_thr)) if (hits_thr + fp_thr) else float("nan")
    f1_thr = (2 * precision_thr * recall_thr / (precision_thr + recall_thr)) if (precision_thr + recall_thr) else float("nan")
    med_loc_thr = float(pos["loc_err_threshold_km"].dropna().median()) if hits_thr else float("nan")
    med_loc_top = float(pos["loc_err_top_usable_km"].dropna().median()) if hits_top else float("nan")

    md_lines = [
        f"# Detector evaluation on data_for_ml",
        "",
        f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- tolerance: ±{args.tolerance_km} km around event range",
        f"- positives evaluated: **{total_pos}**, negatives: **{total_neg}**",
        "",
        "## Threshold-based detection",
        f"- recall:    **{recall_thr:.3f}** ({hits_thr}/{total_pos})",
        f"- precision: **{precision_thr:.3f}**  (FP={fp_thr})",
        f"- F1:        **{f1_thr:.3f}**",
        f"- median localization error (hits only): **{med_loc_thr:.3f} km**",
        "",
        "## Top-usable-candidate (single, threshold-agnostic)",
        f"- recall: **{recall_top:.3f}** ({hits_top}/{total_pos})",
        f"- median localization error: **{med_loc_top:.3f} km**",
        "",
    ]

    # by pulse duration
    if "pulse_ns" in pos.columns and pos["pulse_ns"].notna().any():
        md_lines.append("## Recall by pulse duration")
        md_lines.append("")
        md_lines.append("| pulse_ns | n | recall_thr | recall_top | median_loc_err_top_km |")
        md_lines.append("|---|---:|---:|---:|---:|")
        for pulse, sub in pos.groupby("pulse_ns"):
            n = len(sub)
            r_thr = sub["hit_threshold"].fillna(False).sum() / n
            r_top = sub["hit_top_usable"].fillna(False).sum() / n
            med = float(sub["loc_err_top_usable_km"].dropna().median()) if r_top > 0 else float("nan")
            md_lines.append(f"| {int(pulse)} | {n} | {r_thr:.3f} | {r_top:.3f} | {med:.3f} |")
        md_lines.append("")

    if "id_line" in pos.columns and pos["id_line"].notna().any():
        md_lines.append("## Recall by id_line")
        md_lines.append("")
        md_lines.append("| id_line | n | recall_thr | recall_top |")
        md_lines.append("|---|---:|---:|---:|")
        for line_id, sub in pos.groupby("id_line"):
            n = len(sub)
            r_thr = sub["hit_threshold"].fillna(False).sum() / n
            r_top = sub["hit_top_usable"].fillna(False).sum() / n
            md_lines.append(f"| {int(line_id)} | {n} | {r_thr:.3f} | {r_top:.3f} |")
        md_lines.append("")

    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text("\n".join(md_lines), encoding="utf-8")
    LOG.info("wrote %s", args.md)

    print(f"\nEvaluated: {total_pos} positives, {total_neg} negatives")
    print(f"Threshold:    recall={recall_thr:.3f} precision={precision_thr:.3f} F1={f1_thr:.3f} med_err={med_loc_thr:.3f}km")
    print(f"Top-usable:   recall={recall_top:.3f} med_err={med_loc_top:.3f}km")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
