"""Sensitivity analysis of the combined-score weights S(g) = w_peak*peak + w_broad*broad + w_energy*energy.

Idea: the three per-group score components (peak, broad, energy_z) do not depend on the
weights. Only the linear combination S(g) and its threshold thr = med + 3*MAD do. The peak
guard P_thr = med + 5*MAD(peak) is independent of the weights. So we extract the components
once for every labeled record, then evaluate Precision/Recall/F1 for thousands of weight
vectors in pure numpy, exactly reproducing the production combined-mode detector.

Outputs:
- ternary heatmap of F1 over the weight simplex (chosen point + single-component vertices),
- histogram of F1 over random simplex samples (chosen point marked),
- a small markdown/CSV summary with the key numbers for the slide.

Usage:
    .venv/bin/python scripts/sweep_score_weights.py            # extract (cached) + sweep + plot
    .venv/bin/python scripts/sweep_score_weights.py --reuse    # reuse component cache
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import time
from pathlib import Path

# Keep matplotlib/cache writable regardless of the data mount layout.
_SCRATCH = Path(
    os.environ.get(
        "PHI_OTDR_CACHE_ROOT",
        "/private/tmp/claude-501/-Users-nort-Desktop------8---------phi-otdr-diploma/"
        "92029b4f-60bc-4396-873e-1209f41d6a18/scratchpad/phi_cache",
    )
)
_SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ["PHI_OTDR_CACHE_ROOT"] = str(_SCRATCH)

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# Reuse the exact numeric helpers from the production detector to avoid drift.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.parser.detect_from_aligned import (  # noqa: E402
    _distance_axis_km,
    _estimate_signal_end_km,
    _group_by_distance,
    _limit_by_score,
    _mad,
    _pick_peaks,
    _time_axis_s,
)

# ---- fixed detector hyperparameters (production defaults) ----
GROUP_BINS = 20
FREQ_MIN_HZ = 5.0
FREQ_MAX_HZ = 500.0
MIN_SEP_KM = 2.0
THRESHOLD_K = 3.0
PEAK_THRESHOLD_K = 5.0
IGNORE_START_KM = 1.0
END_GUARD_KM = 1.0
MAX_DETECTIONS = 8
TOLERANCE_KM = 0.5

# chosen weights in the thesis
W_CHOSEN = (0.55, 0.30, 0.15)

ADC_FS_HZ = 50_000_000.0
N_FIBER = 1.468
C_LIGHT = 299_792_458.0
DIST_PER_BIN_KM = (C_LIGHT / (2.0 * N_FIBER * ADC_FS_HZ)) / 1000.0

DESCRIPTION = Path("/Volumes/data/phi-OTDR/raw/data_for_ml/data_description.json")
RECORDS_ROOT = Path("data/processed_usb/parser_cache/records/data_for_ml")

CACHE_PKL = _SCRATCH / "score_components_cache.pkl"
ASSETS = Path("slides/assets")
THESIS_FIG = Path("thesis/figures")
REPORTS = Path("reports")


def bin_to_km(b: float) -> float:
    return float(b) * DIST_PER_BIN_KM


def extract_components(npz_path: Path) -> dict:
    """Replicate the numeric steps 1-5 of run_detection (no plotting) to get score components."""
    data = np.load(npz_path)
    aligned = np.asarray(data["aligned"], dtype=np.float32)
    starts = np.asarray(data["starts"], dtype=np.int64)
    trace_len = int(data["trace_len"])
    adc_fs_hz = float(data["adc_fs_hz"])

    n_traces, n_bins = aligned.shape
    dist_km = _distance_axis_km(trace_len=n_bins, adc_fs_hz=adc_fs_hz)
    time_s = _time_axis_s(starts=starts, n_traces=n_traces, trace_len=trace_len, adc_fs_hz=adc_fs_hz)

    # Step 1: robust background + residual, usable-zone end
    background = np.median(aligned, axis=0)
    residual = aligned - background[None, :]
    signal_end_km = _estimate_signal_end_km(background=background, dist_km=dist_km)
    usable_end_km = max(float(dist_km[0]), signal_end_km - END_GUARD_KM)

    # Step 2: MAD normalization per distance bin
    scale = np.asarray(_mad(residual, axis=0), dtype=np.float32) + np.float32(1e-9)
    z = residual / scale[None, :]

    # Step 3: group by distance, FFT over time
    grouped, g_idx = _group_by_distance(z, group_bins=GROUP_BINS)
    dist_group = dist_km[g_idx * GROUP_BINS]
    dt_trace = float(np.median(np.diff(time_s))) if len(time_s) > 1 else float(trace_len / adc_fs_hz)
    dt_trace = max(dt_trace, 1e-9)

    spec = np.fft.rfft(grouped, axis=0)
    freq = np.fft.rfftfreq(grouped.shape[0], d=dt_trace)
    power_db_raw = 10.0 * np.log10((np.abs(spec) ** 2) / max(1, grouped.shape[0]) + 1e-24)
    fmask = (freq >= FREQ_MIN_HZ) & (freq <= min(FREQ_MAX_HZ, freq[-1]))
    if not np.any(fmask):
        raise ValueError("No frequencies left after masking")
    p_view = power_db_raw[fmask, :]

    # Step 4: robust normalization per frequency + components
    row_med = np.median(p_view, axis=1, keepdims=True)
    row_mad = np.asarray(_mad(p_view, axis=1, keepdims=True), dtype=np.float32) + np.float32(1e-9)
    zf = (p_view - row_med) / row_mad

    peak_score = np.max(zf, axis=0).astype(np.float64)
    broad_score = np.mean(np.clip(zf - 1.0, 0.0, None), axis=0).astype(np.float64)
    energy_rms = np.sqrt(np.mean(grouped**2, axis=0))
    energy_z = ((energy_rms - np.median(energy_rms)) / (float(_mad(energy_rms)) + 1e-9)).astype(np.float64)

    return {
        "peak": peak_score,
        "broad": broad_score,
        "energy": energy_z,
        "dist_group": dist_group.astype(np.float64),
        "usable_end_km": float(usable_end_km),
        "ignore_start_km": float(IGNORE_START_KM),
    }


def build_cache(reuse: bool) -> list[dict]:
    if reuse and CACHE_PKL.exists():
        with open(CACHE_PKL, "rb") as f:
            recs = pickle.load(f)
        print(f"reused component cache: {len(recs)} records from {CACHE_PKL}")
        return recs

    description = json.loads(DESCRIPTION.read_text(encoding="utf-8"))
    recs: list[dict] = []
    t0 = time.time()
    for fname, lbl in description.items():
        rec_name = fname.replace(".parquet", "")
        npz = RECORDS_ROOT / rec_name / "aligned.npz"
        if not npz.exists():
            continue
        try:
            comp = extract_components(npz)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {rec_name}: {type(exc).__name__}: {exc}")
            continue

        cm = lbl.get("comment", "")
        m = re.search(r"импульс[ыа]?\s+(\d+)\s*нс", cm)
        pulse_ns = int(m.group(1)) if m else None

        rec = {
            "file": rec_name,
            "label": "positive" if lbl.get("event") == 1 else "negative",
            "pulse_ns": pulse_ns,
            **comp,
        }
        if lbl.get("event") == 1:
            rec["es_km"] = bin_to_km(lbl["event_start"])
            rec["ee_km"] = bin_to_km(lbl["event_end"])
        recs.append(rec)
        if len(recs) % 25 == 0:
            print(f"  extracted {len(recs)} ... ({time.time() - t0:.1f}s)")

    with open(CACHE_PKL, "wb") as f:
        pickle.dump(recs, f)
    print(f"extracted {len(recs)} records in {time.time() - t0:.1f}s -> {CACHE_PKL}")
    return recs


def detect_one(rec: dict, w: tuple[float, float, float]) -> list[float]:
    """Reproduce the production combined-mode candidate selection for a weight vector."""
    peak = rec["peak"]
    broad = rec["broad"]
    energy = rec["energy"]
    dist_group = rec["dist_group"]
    wp, wb, we = w

    combined = wp * peak + wb * broad + we * energy
    combined_med = float(np.median(combined))
    combined_mad = float(_mad(combined))
    thr = combined_med + THRESHOLD_K * combined_mad

    peak_med = float(np.median(peak))
    peak_mad = float(_mad(peak))
    peak_thr = peak_med + PEAK_THRESHOLD_K * peak_mad
    peak_pass = peak >= peak_thr

    candidate_score = combined.copy()
    candidate_mask = (
        peak_pass
        & (dist_group >= rec["ignore_start_km"])
        & (dist_group <= rec["usable_end_km"])
    )
    candidate_score[~candidate_mask] = -np.inf

    picked = _pick_peaks(score=candidate_score, dist_km=dist_group, threshold=thr, min_sep_km=MIN_SEP_KM)
    picked = _limit_by_score(picked, score=combined, max_detections=MAX_DETECTIONS)
    return [float(dist_group[i]) for i in picked]


def evaluate(recs: list[dict], w: tuple[float, float, float]) -> dict:
    hits = 0
    fp = 0
    n_pos = 0
    n_det = 0
    for rec in recs:
        detected = detect_one(rec, w)
        n_det += len(detected)
        if rec["label"] == "positive":
            n_pos += 1
            es, ee = rec["es_km"], rec["ee_km"]
            in_zone = [d for d in detected if (es - TOLERANCE_KM) <= d <= (ee + TOLERANCE_KM)]
            if in_zone:
                hits += 1
            fp += len(detected) - len(in_zone)
        else:
            fp += len(detected)
    recall = hits / n_pos if n_pos else float("nan")
    precision = hits / (hits + fp) if (hits + fp) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "hits": hits,
        "fp": fp,
        "n_det": n_det,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true", help="reuse component cache if present")
    ap.add_argument("--grid-step", type=float, default=0.02)
    ap.add_argument("--n-random", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260624)
    args = ap.parse_args()

    recs = build_cache(reuse=args.reuse)
    n_pos = sum(r["label"] == "positive" for r in recs)
    n_neg = sum(r["label"] == "negative" for r in recs)
    print(f"records: {len(recs)} (pos={n_pos}, neg={n_neg})")

    # ---- validation against the known reference at the chosen weights ----
    ref = evaluate(recs, W_CHOSEN)
    print(
        f"VALIDATION @ {W_CHOSEN}: hits={ref['hits']} fp={ref['fp']} "
        f"recall={ref['recall']:.3f} precision={ref['precision']:.3f} f1={ref['f1']:.3f}"
    )
    if not (ref["hits"] == 35 and ref["fp"] == 20):
        print("  WARNING: does not match reference (expected hits=35, fp=20). Check extraction fidelity.")

    # ---- single-component vertices (same machinery, all weight on one component) ----
    vertices = {
        "peak": (1.0, 0.0, 0.0),
        "broad": (0.0, 1.0, 0.0),
        "energy": (0.0, 0.0, 1.0),
    }
    vert_res = {k: evaluate(recs, v) for k, v in vertices.items()}
    for k, r in vert_res.items():
        print(f"vertex {k:7s}: f1={r['f1']:.3f} recall={r['recall']:.3f} precision={r['precision']:.3f} fp={r['fp']}")

    # ---- dense grid over the simplex ----
    step = args.grid_step
    grid_w, grid_f1 = [], []
    n = int(round(1.0 / step))
    for i in range(n + 1):
        for j in range(n + 1 - i):
            wp = i * step
            wb = j * step
            we = 1.0 - wp - wb
            if we < -1e-9:
                continue
            we = max(0.0, we)
            res = evaluate(recs, (wp, wb, we))
            grid_w.append((wp, wb, we))
            grid_f1.append(res["f1"])
    grid_w = np.array(grid_w)
    grid_f1 = np.array(grid_f1)
    best_idx = int(np.argmax(grid_f1))
    best_w = grid_w[best_idx]
    best_f1 = grid_f1[best_idx]
    print(f"grid best F1={best_f1:.3f} at w(peak,broad,energy)=({best_w[0]:.2f},{best_w[1]:.2f},{best_w[2]:.2f})")

    # ---- random simplex samples (Dirichlet uniform) ----
    rng = np.random.default_rng(args.seed)
    rand_w = rng.dirichlet(np.ones(3), size=args.n_random)
    rand_f1 = np.array([evaluate(recs, tuple(w))["f1"] for w in rand_w])
    chosen_f1 = ref["f1"]
    pct_better = float(np.mean(rand_f1 <= chosen_f1) * 100.0)
    print(
        f"random combos: mean F1={rand_f1.mean():.3f}, median={np.median(rand_f1):.3f}, "
        f"max={rand_f1.max():.3f}; chosen beats {pct_better:.1f}% of random combos"
    )

    _plot_ternary(grid_w, grid_f1, best_w, best_f1, ASSETS / "weight_sweep_ternary.png")
    _plot_histogram(rand_f1, chosen_f1, vert_res, best_f1, ASSETS / "weight_sweep_hist.png")

    # ---- summary table for the slide / thesis ----
    REPORTS.mkdir(parents=True, exist_ok=True)
    summary = {
        "chosen": {"w": list(W_CHOSEN), **ref},
        "grid_best": {"w": [float(x) for x in best_w], "f1": float(best_f1)},
        "vertices": {k: vert_res[k] for k in vertices},
        "random": {
            "n": int(args.n_random),
            "mean_f1": float(rand_f1.mean()),
            "median_f1": float(np.median(rand_f1)),
            "max_f1": float(rand_f1.max()),
            "chosen_percentile": pct_better,
        },
    }
    (REPORTS / "weight_sweep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {REPORTS / 'weight_sweep_summary.json'}")
    print(f"wrote {ASSETS / 'weight_sweep_ternary.png'}")
    print(f"wrote {ASSETS / 'weight_sweep_hist.png'}")
    return 0


def _bary_to_xy(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """peak -> top, broad -> bottom-left, energy -> bottom-right."""
    wp, wb, we = w[:, 0], w[:, 1], w[:, 2]
    x = 0.5 * wp + 1.0 * we
    y = (np.sqrt(3.0) / 2.0) * wp
    return x, y


def _plot_ternary(grid_w: np.ndarray, grid_f1: np.ndarray, best_w: np.ndarray, best_f1: float, out: Path) -> None:
    x, y = _bary_to_xy(grid_w)
    triang = mtri.Triangulation(x, y)
    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    tcf = ax.tricontourf(triang, grid_f1, levels=14, cmap="viridis")
    cb = fig.colorbar(tcf, ax=ax, shrink=0.74, pad=0.02)
    cb.set_label("F1-мера", fontsize=12)

    # triangle outline
    corners = np.array([[0.5, np.sqrt(3) / 2], [0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    ax.plot(corners[:, 0], corners[:, 1], color="white", linewidth=1.0, alpha=0.6)

    # grid-best point (draw first, under the star)
    bx, by = _bary_to_xy(best_w.reshape(1, 3))
    ax.scatter(bx, by, s=130, marker="o", facecolor="none", edgecolor="white", linewidth=2.0, zorder=5,
               label=f"максимум F1 на сетке ({best_f1:.3f})")
    # chosen point
    cx, cy = _bary_to_xy(np.array([list(W_CHOSEN)]))
    ax.scatter(cx, cy, s=260, marker="*", color="#ffd166", edgecolor="black", linewidth=1.1, zorder=6,
               label="выбранные веса (0,55; 0,30; 0,15)")

    # corner labels
    ax.text(0.5, np.sqrt(3) / 2 + 0.055, "только\npeak", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.text(-0.04, -0.035, "только\nbroad", ha="right", va="top", fontsize=11, fontweight="bold")
    ax.text(1.04, -0.035, "только\nenergy", ha="left", va="top", fontsize=11, fontweight="bold")

    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.16, np.sqrt(3) / 2 + 0.20)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("F1-мера на симплексе весов\n$S(g)=w_1\\,peak+w_2\\,broad+w_3\\,energy$", fontsize=13, pad=14)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.10), fontsize=10, framealpha=0.92, ncol=1)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_histogram(rand_f1: np.ndarray, chosen_f1: float, vert_res: dict, best_f1: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    ax.hist(rand_f1, bins=28, color="#4D77A8", alpha=0.82, edgecolor="white", linewidth=0.4)
    ax.axvline(rand_f1.mean(), color="gray", linestyle="--", linewidth=1.5,
               label=f"среднее по случайным комбинациям ({rand_f1.mean():.3f})")
    ax.axvline(vert_res["energy"]["f1"], color="#e76f51", linestyle=":", linewidth=1.8,
               label=f"только energy ({vert_res['energy']['f1']:.3f})")
    ax.axvline(vert_res["peak"]["f1"], color="#2a9d8f", linestyle=":", linewidth=1.8,
               label=f"только peak ({vert_res['peak']['f1']:.3f})")
    ax.axvline(chosen_f1, color="#e63946", linewidth=2.4,
               label=f"выбранные веса ({chosen_f1:.3f})")
    ax.axvline(best_f1, color="black", linestyle="-.", linewidth=1.4,
               label=f"максимум на сетке ({best_f1:.3f})")
    ax.set_xlabel("F1-мера", fontsize=12)
    ax.set_ylabel("Число комбинаций весов", fontsize=12)
    ax.set_title(f"Распределение F1 по {len(rand_f1)} случайным комбинациям весов", fontsize=13)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.92)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
