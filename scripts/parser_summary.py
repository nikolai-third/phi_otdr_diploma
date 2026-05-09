"""Build a per-group summary of parser quality across all USB-cached records.

Reads ``manifest_ok.jsonl`` (legacy parser run, 552 raw files) plus
``manifest_ml_ok.jsonl`` (data_for_ml records prepared without parsing) and
emits a CSV + Markdown table suitable for the thesis "Data" chapter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ok", type=Path, default=Path("data/processed_usb/parser_cache/manifest_ok.jsonl"))
    p.add_argument("--err", type=Path, default=Path("data/processed_usb/parser_cache/manifest_err.jsonl"))
    p.add_argument("--ml-ok", type=Path, default=Path("data/processed_usb/parser_cache/manifest_ml_ok.jsonl"))
    p.add_argument("--csv", type=Path, default=Path("reports/tables/parser_summary.csv"))
    p.add_argument("--md", type=Path, default=Path("reports/parser_summary.md"))
    args = p.parse_args()

    ok = load_manifest(args.ok)
    err = load_manifest(args.err)
    ml_ok = load_manifest(args.ml_ok)

    # Keep only the *latest* row per source_rel (in case manifest grew over multiple runs).
    if not ok.empty:
        ok = ok.drop_duplicates(subset=["source_rel"], keep="last").reset_index(drop=True)
    if not err.empty:
        err = err.drop_duplicates(subset=["source_rel"], keep="last").reset_index(drop=True)
        # If a file is now in ok, drop it from err
        if not ok.empty:
            err = err[~err["source_rel"].isin(ok["source_rel"])].reset_index(drop=True)

    ok["group"] = ok["source_rel"].str.split("/").str[0] if not ok.empty else None
    if not err.empty:
        err["group"] = err["source_rel"].str.split("/").str[0]

    rows = []
    groups = sorted(set((ok["group"].tolist() if not ok.empty else []) + (err["group"].tolist() if not err.empty else [])))
    for g in groups:
        g_ok = ok[ok["group"] == g]
        g_err = err[err["group"] == g]
        n_ok = len(g_ok)
        n_err = len(g_err)
        n_total = n_ok + n_err
        if n_ok > 0:
            align_pct = 100.0 * g_ok["alignment_applied"].sum() / n_ok
            cov = g_ok["n_extracted_traces"] / g_ok["n_detected_starts"].clip(lower=1)
            cov_med = float(np.median(cov))
            res_b_med = float(g_ok["residual_before_abs_mean"].median())
            res_a_med = float(g_ok["residual_after_abs_mean"].median())
            res_a_p90 = float(g_ok["residual_after_abs_mean"].quantile(0.9))
            high_after = int((g_ok["residual_after_abs_mean"] > 5).sum())
        else:
            align_pct = cov_med = res_b_med = res_a_med = res_a_p90 = float("nan")
            high_after = 0
        rows.append({
            "group": g,
            "n_total": n_total,
            "n_ok": n_ok,
            "n_err": n_err,
            "ok_rate_%": round(100.0 * n_ok / n_total, 1) if n_total else float("nan"),
            "align_applied_%": round(align_pct, 1),
            "coverage_median": round(cov_med, 3),
            "residual_before_med": round(res_b_med, 3),
            "residual_after_med": round(res_a_med, 3),
            "residual_after_p90": round(res_a_p90, 3),
            "files_with_after>5": high_after,
        })

    # data_for_ml row from ml_ok manifest
    if not ml_ok.empty:
        ml = ml_ok.drop_duplicates(subset=["source_rel"], keep="last")
        align_pct = 100.0 * ml["alignment_applied"].sum() / len(ml)
        res_b_med = float(ml["residual_before_abs_mean"].median())
        res_a_med = float(ml["residual_after_abs_mean"].median())
        rows.append({
            "group": "data_for_ml (no parser, already-extracted)",
            "n_total": len(ml),
            "n_ok": len(ml),
            "n_err": 0,
            "ok_rate_%": 100.0,
            "align_applied_%": round(align_pct, 1),
            "coverage_median": float("nan"),
            "residual_before_med": round(res_b_med, 3),
            "residual_after_med": round(res_a_med, 3),
            "residual_after_p90": float("nan"),
            "files_with_after>5": int((ml["residual_after_abs_mean"] > 5).sum()),
        })

    df = pd.DataFrame(rows)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)

    # also overall totals
    n_ok_total = int(df["n_ok"].sum())
    n_err_total = int(df["n_err"].sum())
    n_total = n_ok_total + n_err_total

    md = [
        "# Parser quality summary",
        "",
        f"- total files indexed: **{n_total}**, ok: **{n_ok_total}**, err: **{n_err_total}**, ok rate: **{100*n_ok_total/n_total:.1f}%**",
        "",
        "| group | total | ok | err | ok% | align% | cov_med | res_before_med | res_after_med | res_after_p90 | after>5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        md.append(
            f"| {r['group']} | {r['n_total']} | {r['n_ok']} | {r['n_err']} | "
            f"{r['ok_rate_%']:.1f} | {r['align_applied_%']} | {r['coverage_median']} | "
            f"{r['residual_before_med']} | {r['residual_after_med']} | {r['residual_after_p90']} | "
            f"{r['files_with_after>5']} |"
        )

    # error breakdown
    if not err.empty:
        md.extend(["", "## Error type breakdown", ""])
        ec = err["error"].value_counts()
        for msg, c in ec.head(8).items():
            md.append(f"- {c}: `{str(msg)[:100]}`")

    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(df.to_string(index=False))
    print(f"\nwrote {args.csv} and {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
