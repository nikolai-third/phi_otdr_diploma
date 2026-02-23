"""Generate baseline analytics artifacts (figures, tables, summary)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

matplotlib.use("Agg")


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_artifacts(
    metrics_path: str | Path = "data/processed/baseline_metrics.parquet",
    figures_dir: str | Path = "reports/figures",
    tables_dir: str | Path = "reports/tables",
    summary_path: str | Path = "reports/summary_baseline.md",
) -> Path:
    """Create figures/tables/summary from baseline metrics."""

    metrics_file = Path(metrics_path)
    if not metrics_file.exists():
        raise FileNotFoundError(f"baseline metrics not found: {metrics_file}")

    figures = Path(figures_dir)
    tables = Path(tables_dir)
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(metrics_file)
    if df.empty:
        raise ValueError("baseline metrics parquet is empty")

    ok_df = df[df["status"] == "ok"].copy()
    err_df = df[df["status"] != "ok"].copy()

    # 1) file sizes histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["file_size_mb"].dropna(), bins=40)
    ax.set_xlabel("File size, MB")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of file sizes")
    ax.grid(True, alpha=0.2)
    _save_fig(fig, figures / "hist_file_sizes.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    size_positive = df["file_size_mb"].dropna()
    size_positive = size_positive[size_positive > 0]
    ax.hist(size_positive, bins=40, log=True)
    ax.set_xscale("log")
    ax.set_xlabel("File size, MB (log scale)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Histogram of file sizes (log)")
    ax.grid(True, alpha=0.2)
    _save_fig(fig, figures / "hist_file_sizes_log.png")

    # 2) std distribution histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ok_df["sample_std"].replace([np.inf, -np.inf], np.nan).dropna(), bins=40)
    ax.set_xlabel("Sample std")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of sample std")
    ax.grid(True, alpha=0.2)
    _save_fig(fig, figures / "hist_std_distribution.png")

    # 3) format distribution
    format_counts = df["inferred_format"].fillna("unknown").value_counts().rename_axis("format").reset_index(name="count")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(format_counts["format"].astype(str), format_counts["count"].astype(int))
    ax.set_xlabel("Format")
    ax.set_ylabel("Count")
    ax.set_title("Distribution by format")
    ax.tick_params(axis="x", rotation=45)
    _save_fig(fig, figures / "format_distribution.png")

    # 4) top 20 largest files table
    top20 = df.sort_values("size_bytes", ascending=False).head(20)[
        ["record_id", "path", "inferred_format", "size_bytes", "file_size_mb", "sample_std", "status"]
    ]
    top20.to_csv(tables / "top20_largest_files.csv", index=False)

    # 5) scatter size vs std
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter_df = ok_df[["file_size_mb", "sample_std"]].replace([np.inf, -np.inf], np.nan).dropna()
    ax.scatter(scatter_df["file_size_mb"], scatter_df["sample_std"], s=10, alpha=0.6)
    ax.set_xlabel("File size, MB")
    ax.set_ylabel("Sample std")
    ax.set_title("File size vs sample std")
    ax.grid(True, alpha=0.2)
    _save_fig(fig, figures / "scatter_size_vs_std.png")

    # std by format for major groups
    top_formats = (
        ok_df["inferred_format"].fillna("unknown").value_counts().head(6).index.tolist()
        if not ok_df.empty
        else []
    )
    if top_formats:
        box_data = []
        labels = []
        for fmt in top_formats:
            vals = (
                ok_df.loc[ok_df["inferred_format"] == fmt, "sample_std"]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .values
            )
            if len(vals) > 0:
                box_data.append(vals)
                labels.append(fmt)
        if box_data:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.boxplot(box_data, labels=labels, showfliers=False)
            ax.set_xlabel("Format")
            ax.set_ylabel("Sample std")
            ax.set_title("Sample std by format")
            ax.grid(True, axis="y", alpha=0.2)
            _save_fig(fig, figures / "box_std_by_format.png")

    # tables
    format_counts.to_csv(tables / "format_distribution.csv", index=False)
    df[["record_id", "path", "status", "error"]].to_csv(tables / "baseline_status.csv", index=False)
    df.describe(include="all").to_csv(tables / "baseline_describe.csv")

    summary_by_format = (
        df.groupby("inferred_format", dropna=False)
        .agg(
            files=("record_id", "count"),
            ok=("status", lambda s: int((s == "ok").sum())),
            errors=("status", lambda s: int((s != "ok").sum())),
            total_size_mb=("file_size_mb", "sum"),
            mean_std=("sample_std", "mean"),
            p90_std=("sample_std", lambda x: float(pd.Series(x).quantile(0.9))),
        )
        .reset_index()
        .sort_values("files", ascending=False)
    )
    summary_by_format.to_csv(tables / "summary_by_format.csv", index=False)

    summary_by_group = (
        df.groupby("file_group", dropna=False)
        .agg(
            files=("record_id", "count"),
            ok=("status", lambda s: int((s == "ok").sum())),
            errors=("status", lambda s: int((s != "ok").sum())),
            total_size_mb=("file_size_mb", "sum"),
            mean_std=("sample_std", "mean"),
        )
        .reset_index()
        .sort_values("total_size_mb", ascending=False)
    )
    summary_by_group.to_csv(tables / "summary_by_group.csv", index=False)

    noise_quantiles = ok_df["sample_std"].replace([np.inf, -np.inf], np.nan).dropna().quantile(
        [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    )
    noise_quantiles_df = noise_quantiles.rename_axis("quantile").reset_index(name="sample_std")
    noise_quantiles_df.to_csv(tables / "noise_quantiles.csv", index=False)

    top_noisy = (
        ok_df.sort_values("sample_std", ascending=False)
        .head(20)[["record_id", "path", "inferred_format", "file_size_mb", "sample_std"]]
        .copy()
    )
    top_noisy.to_csv(tables / "top20_noisiest_files.csv", index=False)

    # summary markdown
    total_files = int(len(df))
    total_size_bytes = float(df["size_bytes"].fillna(0).sum())
    avg_size_mb = float(df["file_size_mb"].dropna().mean()) if total_files else 0.0

    dtype_series = ok_df["dtype"].fillna("unknown").astype(str)
    float_count = int(dtype_series.str.contains("float", case=False, regex=False).sum())
    int_count = int(dtype_series.str.contains("int", case=False, regex=False).sum())

    std_values = ok_df["sample_std"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(std_values) > 0:
        std_q50 = float(std_values.quantile(0.5))
        std_q90 = float(std_values.quantile(0.9))
        noise_observation = (
            f"Медианный std={std_q50:.6g}, 90-перцентиль std={std_q90:.6g}. "
            "Есть выраженный разброс шумовых уровней между файлами."
        )
    else:
        noise_observation = "Недостаточно валидных numeric-сэмплов для оценки шумовых характеристик."

    lines = [
        "# Baseline Summary",
        "",
        f"- Files processed: **{total_files}**",
        f"- Total volume (bytes): **{int(total_size_bytes)}**",
        f"- Average file size (MB): **{avg_size_mb:.3f}**",
        f"- Float dtype share: **{(float_count / max(1, len(ok_df))):.2%}**",
        f"- Int dtype share: **{(int_count / max(1, len(ok_df))):.2%}**",
        f"- Error count: **{int(len(err_df))}**",
        "",
        "## Coverage",
        "",
        f"- Successful records: **{int((df['status'] == 'ok').sum())}**",
        f"- Failed records: **{int((df['status'] != 'ok').sum())}**",
        f"- Cache volume (approx): **{Path('cache').exists() and 'see `du -sh cache`' or 'n/a'}**",
        "",
        "## Noise Observations",
        "",
        f"- {noise_observation}",
        "",
        "## Top Noisy Files",
        "",
    ]
    if not top_noisy.empty:
        lines.append("| record_id | path | format | sample_std |")
        lines.append("|---|---|---|---:|")
        for row in top_noisy.head(10).itertuples(index=False):
            lines.append(f"| {row.record_id} | {row.path} | {row.inferred_format} | {float(row.sample_std):.6g} |")
    else:
        lines.append("- No noisy-file ranking available.")

    lines.extend(
        [
        "",
        "## Generated Artifacts",
        "",
        "- reports/figures/hist_file_sizes.png",
        "- reports/figures/hist_file_sizes_log.png",
        "- reports/figures/hist_std_distribution.png",
        "- reports/figures/format_distribution.png",
        "- reports/figures/box_std_by_format.png",
        "- reports/figures/scatter_size_vs_std.png",
        "- reports/tables/top20_largest_files.csv",
        "- reports/tables/top20_noisiest_files.csv",
        "- reports/tables/format_distribution.csv",
        "- reports/tables/summary_by_format.csv",
        "- reports/tables/summary_by_group.csv",
        "- reports/tables/noise_quantiles.csv",
        "- reports/tables/baseline_status.csv",
    ])

    out_summary = Path(summary_path)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("baseline summary saved: %s", out_summary)
    return out_summary
