"""Render every figure used in the bachelor thesis under a single, strict
matplotlib style. Outputs to ``thesis/figures/``.

Inputs:
    * Aligned reflectograms for the reference disturbance case
      ``parser_cache/records/измерение_возмущение/2024-10-11_18_59/aligned.npz``.
    * ``reports/tables/data_for_ml_eval{,_peak,_energy,_relaxed}.csv`` with
      the labelled-set evaluation results for the four score variants.

Style: Times New Roman, 12 pt main text, 11 pt ticks, viridis for waterfalls,
RdBu_r for signed contrast maps.  Captions are produced in Russian to match
the thesis text.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import Normalize

# ---------------------------------------------------------------------------
# Style - single source of truth for every thesis figure.
# ---------------------------------------------------------------------------

_TIMES_FONTS = [
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman Bold Italic.ttf",
]
for _p in _TIMES_FONTS:
    if Path(_p).exists():
        font_manager.fontManager.addfont(_p)

mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.titleweight": "regular",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "figure.dpi": 150,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.1,
    "image.cmap": "viridis",
    "image.aspect": "auto",
    "image.interpolation": "nearest",
    "axes.formatter.use_locale": False,
    "axes.unicode_minus": True,
})

PROJECT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT / "thesis" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

REF_NPZ = (
    Path("/Volumes/data/phi-OTDR/processed/parser_cache/records")
    / "измерение_возмущение/2024-10-11_18_59/aligned.npz"
)
TABLES = PROJECT / "reports" / "tables"

C0 = 299_792_458.0
N_FIBER = 1.468

# ---------------------------------------------------------------------------
# Detector pipeline (replicates src/parser/detect_from_aligned.run_detection).
# ---------------------------------------------------------------------------


def _mad(a, axis=None, keepdims=False):
    med = np.median(a, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(a - med), axis=axis, keepdims=keepdims)


def _quantile_clip(a, lo=0.01, hi=0.99):
    qlo, qhi = np.quantile(a, [lo, hi])
    return float(qlo), float(qhi)


def _signed_clip(a, q_abs=0.995):
    v = float(np.quantile(np.abs(a), q_abs))
    return -v, v


def compute_detector(npz_path: Path, group_bins: int = 20, freq_min_hz: float = 5.0):
    data = np.load(npz_path)
    aligned = np.asarray(data["aligned"], dtype=np.float32)
    starts = np.asarray(data["starts"], dtype=np.int64)
    trace_len = int(data["trace_len"])
    adc_fs_hz = float(data["adc_fs_hz"])

    n_traces, n_bins = aligned.shape
    dist_km = np.arange(n_bins) * (C0 / (2.0 * N_FIBER * adc_fs_hz)) / 1000.0
    dt_trace = float(np.median(np.diff(starts))) / adc_fs_hz
    time_s = np.arange(n_traces) * dt_trace

    bg = np.median(aligned, axis=0)
    residual = aligned - bg[None, :]

    mad_d = 1.4826 * np.median(np.abs(residual - np.median(residual, axis=0)[None, :]), axis=0)
    mad_d = np.maximum(mad_d, 1e-9)
    z = residual / mad_d[None, :]

    n_groups = n_bins // group_bins
    grouped = z[:, : n_groups * group_bins].reshape(n_traces, n_groups, group_bins).mean(axis=2)
    dist_group = dist_km[np.arange(n_groups) * group_bins]

    spec = np.fft.rfft(grouped, axis=0)
    freq = np.fft.rfftfreq(n_traces, d=dt_trace)
    P_db = 10.0 * np.log10(np.abs(spec) ** 2 / max(1, n_traces) + 1e-24)

    fmask = freq >= freq_min_hz
    f_view = freq[fmask]
    p_view = P_db[fmask, :]

    row_med = np.median(p_view, axis=1, keepdims=True)
    row_mad = _mad(p_view, axis=1, keepdims=True) + 1e-9
    zf = (p_view - row_med) / row_mad

    peak_score = np.max(zf, axis=0)
    broad_score = np.mean(np.clip(zf - 1.0, 0.0, None), axis=0)
    energy_rms = np.sqrt(np.mean(grouped**2, axis=0))
    energy_z = (energy_rms - np.median(energy_rms)) / (_mad(energy_rms) + 1e-9)

    combined = 0.55 * peak_score + 0.30 * broad_score + 0.15 * energy_z

    combined_thr = float(np.median(combined) + 3.0 * _mad(combined))
    peak_thr = float(np.median(peak_score) + 5.0 * _mad(peak_score))

    return {
        "aligned": aligned,
        "trace_len": trace_len,
        "adc_fs_hz": adc_fs_hz,
        "dist_km": dist_km,
        "time_s": time_s,
        "background": bg,
        "residual": residual,
        "mad_d": mad_d,
        "z": z,
        "grouped": grouped,
        "dist_group": dist_group,
        "freq": f_view,
        "P_db": p_view,
        "peak_score": peak_score,
        "broad_score": broad_score,
        "energy_z": energy_z,
        "combined": combined,
        "combined_thr": combined_thr,
        "peak_thr": peak_thr,
        "trace_period_s": dt_trace,
    }


# ---------------------------------------------------------------------------
# Helpers - small wrappers that enforce uniform layout across waterfalls.
# ---------------------------------------------------------------------------


def _waterfall(arr, *, x, y, xlabel, ylabel, cbar_label, title, q_lo=0.01, q_hi=0.99,
               cmap="viridis", figsize=(10, 4.2)):
    fig, ax = plt.subplots(figsize=figsize)
    vmin, vmax = _quantile_clip(arr, q_lo, q_hi)
    img = ax.imshow(
        arr,
        extent=[float(x[0]), float(x[-1]), float(y[0]), float(y[-1])],
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(False)
    cbar = fig.colorbar(img, ax=ax, pad=0.015, fraction=0.05)
    cbar.set_label(cbar_label)
    return fig, ax


def _waterfall_signed(arr, *, x, y, xlabel, ylabel, cbar_label, title,
                      q_abs=0.995, cmap="RdBu_r", figsize=(10, 4.2)):
    fig, ax = plt.subplots(figsize=figsize)
    vmin, vmax = _signed_clip(arr, q_abs)
    img = ax.imshow(
        arr,
        extent=[float(x[0]), float(x[-1]), float(y[0]), float(y[-1])],
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(False)
    cbar = fig.colorbar(img, ax=ax, pad=0.015, fraction=0.05)
    cbar.set_label(cbar_label)
    return fig, ax


# ---------------------------------------------------------------------------
# Reference-case figures.
# ---------------------------------------------------------------------------


def fig_reflectogram(d):
    aligned = d["aligned"]
    dist = d["dist_km"]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(dist, aligned[0], color="#1f4e79", linewidth=0.9)
    ax.set_xlabel("Дистанция, км")
    ax.set_ylabel("Амплитуда, отн. ед.")
    ax.set_title("Пример одной рефлектограммы")
    ax.set_xlim(dist[0], dist[-1])
    fig.savefig(FIG_DIR / "fig_reflectogram.png")
    plt.close(fig)


def fig_waterfall_aligned(d):
    fig, _ = _waterfall(
        d["aligned"],
        x=d["dist_km"],
        y=d["time_s"],
        xlabel="Дистанция, км",
        ylabel="Время, с",
        cbar_label="Амплитуда, отн. ед.",
        title="Водопадная диаграмма выровненных рефлектограмм",
    )
    fig.savefig(FIG_DIR / "fig_waterfall_aligned.png")
    plt.close(fig)


def fig_residual_waterfall(d):
    fig, _ = _waterfall_signed(
        d["residual"],
        x=d["dist_km"],
        y=d["time_s"],
        xlabel="Дистанция, км",
        ylabel="Время, с",
        cbar_label="Остаток, отн. ед.",
        title="Остаток после вычитания медианного фона",
    )
    fig.savefig(FIG_DIR / "fig_residual_waterfall.png")
    plt.close(fig)


def fig_background_and_mad(d):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True)
    axes[0].plot(d["dist_km"], d["background"], color="#1f4e79", linewidth=0.9)
    axes[0].set_ylabel("Фон, отн. ед.")
    axes[0].set_title("Медианный фон по времени и MAD-нормировка по дистанции")
    axes[1].plot(d["dist_km"], d["mad_d"], color="#a6324a", linewidth=0.9)
    axes[1].set_ylabel("MAD, отн. ед.")
    axes[1].set_xlabel("Дистанция, км")
    axes[1].set_xlim(d["dist_km"][0], d["dist_km"][-1])
    fig.savefig(FIG_DIR / "fig_background_mad.png")
    plt.close(fig)


def fig_normalized_waterfall(d):
    fig, _ = _waterfall_signed(
        d["z"],
        x=d["dist_km"],
        y=d["time_s"],
        xlabel="Дистанция, км",
        ylabel="Время, с",
        cbar_label="z-оценка",
        title="Нормированный остаток (z-оценка по дистанционным бинам)",
        q_abs=0.99,
    )
    fig.savefig(FIG_DIR / "fig_normalized_waterfall.png")
    plt.close(fig)


def fig_fft_map(d):
    fig, _ = _waterfall(
        d["P_db"],
        x=d["dist_group"],
        y=d["freq"],
        xlabel="Дистанция, км",
        ylabel="Частота, Гц",
        cbar_label="Спектральная плотность, дБ",
        title="Спектральная плотность сгруппированного сигнала",
        q_lo=0.02,
        q_hi=0.995,
        figsize=(10, 4.4),
    )
    fig.savefig(FIG_DIR / "fig_fft_map.png")
    plt.close(fig)


def fig_score_components(d):
    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.plot(d["dist_group"], d["peak_score"], color="#1f4e79",
            linewidth=1.0, label="спектральный пик")
    ax.plot(d["dist_group"], d["broad_score"], color="#c98a1d",
            linewidth=1.0, label="широкополосная активность")
    ax.plot(d["dist_group"], d["energy_z"], color="#3a8a3a",
            linewidth=1.0, alpha=0.8, label="энергия (RMS, z-оценка)")
    ax.plot(d["dist_group"], d["combined"], color="black",
            linewidth=1.4, label="комбинированный score")
    ax.axhline(d["combined_thr"], color="#a6324a", linestyle="--",
               linewidth=1.0, label=f"порог combined = {d['combined_thr']:.2f}")
    ax.axhline(d["peak_thr"], color="#7a3aa6", linestyle=":",
               linewidth=1.0, label=f"порог peak = {d['peak_thr']:.2f}")
    ax.set_xlabel("Дистанция, км")
    ax.set_ylabel("Score")
    ax.set_title("Спектральный score детектора по дистанции")
    ax.set_xlim(d["dist_group"][0], d["dist_group"][-1])
    ax.legend(ncol=2, loc="upper left", framealpha=0.92)
    fig.savefig(FIG_DIR / "fig_score_components.png")
    plt.close(fig)


def fig_final_detection(d, expected_km=(32.0, 42.0)):
    dg = d["dist_group"]
    cs = d["combined"]
    thr = d["combined_thr"]
    pthr = d["peak_thr"]
    ps = d["peak_score"]

    # find local maxima above thr & passing peak gate, separated by 2 km
    mask = (cs >= thr) & (ps >= pthr)
    candidates = []
    for i in np.where(mask)[0]:
        if i == 0 or i == len(cs) - 1:
            continue
        if cs[i] >= cs[i - 1] and cs[i] >= cs[i + 1]:
            candidates.append((float(dg[i]), float(cs[i])))
    candidates.sort(key=lambda t: -t[1])
    picked = []
    for d_km, s in candidates:
        if all(abs(d_km - p[0]) >= 2.0 for p in picked):
            picked.append((d_km, s))
        if len(picked) >= 6:
            break

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(dg, cs, color="black", linewidth=1.0, label="комбинированный score")
    ax.axhline(thr, color="#a6324a", linestyle="--",
               linewidth=1.0, label=f"порог = {thr:.2f}")
    for ek in expected_km:
        ax.axvspan(ek - 0.5, ek + 0.5, color="#c98a1d", alpha=0.18,
                   label="эталонная зона ±0.5 км" if ek == expected_km[0] else None)
    if picked:
        xs, ys = zip(*picked)
        ax.scatter(xs, ys, marker="o", s=70,
                   facecolor="#a6324a", edgecolor="black", zorder=5,
                   label="детекции")
        for x, y in picked:
            ax.annotate(f"{x:.1f}", (x, y), textcoords="offset points",
                        xytext=(6, 6), fontsize=10)
    ax.set_xlabel("Дистанция, км")
    ax.set_ylabel("Score")
    ax.set_title("Финальный детект на эталонной размеченной записи")
    ax.set_xlim(dg[0], dg[-1])
    ax.legend(loc="upper right", framealpha=0.92)
    fig.savefig(FIG_DIR / "fig_final_detection.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Eval-set figures.
# ---------------------------------------------------------------------------


def _read_eval(name):
    return pd.read_csv(TABLES / name)


def fig_loc_err_hist():
    df = _read_eval("data_for_ml_eval.csv")
    pos = df[df["label"] == "positive"]
    err = pos["loc_err_threshold_km"].dropna()
    err_top = pos["loc_err_top_usable_km"].dropna()

    fig, ax = plt.subplots(figsize=(10, 4.0))
    bins = np.linspace(0.0, 5.0, 26)
    ax.hist(err.values, bins=bins, color="#1f4e79", alpha=0.85,
            edgecolor="white", label=f"пороговый детект ($n={len(err)}$)")
    ax.hist(np.clip(err_top.values, 0.0, 5.0), bins=bins,
            color="#c98a1d", alpha=0.55, edgecolor="white",
            label=f"топ-кандидат в usable ($n={len(err_top)}$)")
    ax.axvline(0.5, color="#a6324a", linestyle="--", linewidth=1.0,
               label="допуск ±0.5 км")
    ax.set_xlabel("Ошибка локализации, км")
    ax.set_ylabel("Число файлов")
    ax.set_title("Распределение ошибки локализации (положительные кейсы)")
    ax.legend(loc="upper right", framealpha=0.92)
    fig.savefig(FIG_DIR / "fig_loc_err_hist.png")
    plt.close(fig)


def fig_scatter():
    df = _read_eval("data_for_ml_eval.csv")
    pos = df[df["label"] == "positive"].copy()
    pos["truth_km"] = 0.5 * (pos["event_start_km"] + pos["event_end_km"])
    pos["pred_km"] = pos["detected_km"].fillna(pos["top_usable_km"])
    pos = pos.dropna(subset=["pred_km"])

    fig, ax = plt.subplots(figsize=(6.5, 6.2))
    hits = pos[pos["hit_threshold"] == True]
    misses = pos[pos["hit_threshold"] != True]

    ax.scatter(misses["truth_km"], misses["pred_km"],
               s=42, marker="x", color="#a6324a", linewidth=1.2,
               label=f"промах ($n={len(misses)}$)")
    ax.scatter(hits["truth_km"], hits["pred_km"],
               s=42, marker="o", facecolor="#1f4e79", edgecolor="black",
               linewidth=0.6, label=f"попадание ($n={len(hits)}$)")

    lim_lo = min(pos["truth_km"].min(), pos["pred_km"].min()) - 0.3
    lim_hi = max(pos["truth_km"].max(), pos["pred_km"].max()) + 0.3
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
            color="black", linestyle=":", linewidth=1.0, label="идеальная диагональ")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Истинная середина зоны, км")
    ax.set_ylabel("Предсказанная позиция, км")
    ax.set_title("Сравнение детекций с разметкой (положительные кейсы)")
    ax.legend(loc="upper left", framealpha=0.92)
    fig.savefig(FIG_DIR / "fig_scatter.png")
    plt.close(fig)


def _f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _metrics(df):
    pos = df[df["label"] == "positive"]
    neg = df[df["label"] == "negative"]
    n_pos = len(pos)
    tp = int((pos["hit_threshold"] == True).sum())
    fp_pos = int(pos["n_false_positive"].sum())
    fp_neg = int(neg["n_detected"].sum()) if "n_detected" in df else 0
    fp = fp_pos + fp_neg
    recall = tp / max(n_pos, 1)
    precision = tp / max(tp + fp, 1)
    f1 = _f1(precision, recall)
    return recall, precision, f1, fp


def fig_f1_bars():
    runs = [
        ("combined\n(предложенный)", _read_eval("data_for_ml_eval.csv")),
        ("combined relaxed", _read_eval("data_for_ml_eval_relaxed.csv")),
        ("peak only", _read_eval("data_for_ml_eval_peak.csv")),
        ("energy only", _read_eval("data_for_ml_eval_energy.csv")),
    ]
    labels = []
    rec, prc, f1s = [], [], []
    for label, df in runs:
        labels.append(label)
        r, p, f1, _ = _metrics(df)
        rec.append(r); prc.append(p); f1s.append(f1)

    x = np.arange(len(labels))
    w = 0.27
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.bar(x - w, rec, w, color="#1f4e79", label="Recall")
    ax.bar(x,     prc, w, color="#c98a1d", label="Precision")
    ax.bar(x + w, f1s, w, color="#3a8a3a", label="F1")
    for i, (r, p, f) in enumerate(zip(rec, prc, f1s)):
        ax.text(i - w, r + 0.012, f"{r:.2f}", ha="center", fontsize=10)
        ax.text(i,     p + 0.012, f"{p:.2f}", ha="center", fontsize=10)
        ax.text(i + w, f + 0.012, f"{f:.2f}", ha="center", fontsize=10)

    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(0.75, max(prc) + 0.12))
    ax.set_ylabel("Значение метрики")
    ax.set_title("Сравнение детекторов на размеченном наборе (130 файлов)")
    ax.legend(loc="upper right", framealpha=0.92)
    fig.savefig(FIG_DIR / "fig_f1_bars.png")
    plt.close(fig)


def fig_recall_by_pulse():
    df = _read_eval("data_for_ml_eval.csv")
    pos = df[df["label"] == "positive"].copy()
    grp = pos.groupby("pulse_ns")
    pulses = sorted(grp.groups)
    rec_thr = [grp.get_group(p)["hit_threshold"].mean() for p in pulses]
    rec_top = [grp.get_group(p)["hit_top_usable"].mean() for p in pulses]
    n = [len(grp.get_group(p)) for p in pulses]

    x = np.arange(len(pulses))
    w = 0.4
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - w/2, rec_thr, w, color="#1f4e79", label="пороговый детект")
    ax.bar(x + w/2, rec_top, w, color="#c98a1d", label="топ-кандидат в usable")
    for i, (rt, rl) in enumerate(zip(rec_thr, rec_top)):
        ax.text(i - w/2, rt + 0.02, f"{rt:.2f}", ha="center", fontsize=10)
        ax.text(i + w/2, rl + 0.02, f"{rl:.2f}", ha="center", fontsize=10)
    ax.set_xticks(x, [f"{p}\n($n={ni}$)" for p, ni in zip(pulses, n)])
    ax.set_xlabel("Длительность импульса, нс")
    ax.set_ylabel("Recall")
    ax.set_title("Recall в зависимости от длительности зондирующего импульса")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", framealpha=0.92)
    fig.savefig(FIG_DIR / "fig_recall_by_pulse.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    print(f"writing figures to {FIG_DIR}")
    print(f"loading reference case from {REF_NPZ}")
    d = compute_detector(REF_NPZ)
    print(f"  aligned shape: {d['aligned'].shape}")
    print(f"  combined thr={d['combined_thr']:.3f}, peak thr={d['peak_thr']:.3f}")

    fig_reflectogram(d)
    fig_waterfall_aligned(d)
    fig_residual_waterfall(d)
    fig_background_and_mad(d)
    fig_normalized_waterfall(d)
    fig_fft_map(d)
    fig_score_components(d)
    fig_final_detection(d)
    fig_loc_err_hist()
    fig_scatter()
    fig_f1_bars()
    fig_recall_by_pulse()

    print("done.")


if __name__ == "__main__":
    main()
