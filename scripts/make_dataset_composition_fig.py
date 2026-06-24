"""Dataset composition by disturbance type (for the presentation backup slide).

Counts parquet records per top-level directory of the raw dataset and maps each
directory to a disturbance type (the mapping matches the parser-summary categories
in the thesis). Date-named directories are background monitoring; data_for_ml is the
footstep campaign (130 of these are labeled and used for validation); the other
directories are harmonic excitation, heating / long-line tests, and static stretch.

Output: slides/assets/dataset_composition.png
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW = Path("/Volumes/data/phi-OTDR/raw")
OUT = Path("slides/assets/dataset_composition.png")

# directory -> (disturbance type label, color, is_validated)
DATE_RE = re.compile(r"^\d{2}_\d{2}_\d{4}$")


def count_parquet_by_dir() -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in glob.glob(str(RAW / "**" / "*.parquet"), recursive=True):
        base = os.path.basename(p)
        if base.startswith("._"):
            continue
        top = os.path.relpath(p, RAW).split(os.sep)[0]
        counts[top] = counts.get(top, 0) + 1
    return counts


def aggregate(counts: dict[str, int]) -> list[dict]:
    background = sum(v for d, v in counts.items() if DATE_RE.match(d))
    rows = [
        {"label": "Шаги по грунту (0,57 Гц)", "n": counts.get("data_for_ml", 0),
         "color": "#0F6E8C", "note": "из них 130 размечено -> валидация (F1=0,48)",
         "line": "длинная ~500 км (10-й пролёт);\nанализ в окне ~6-9 км"},
        {"label": "Фоновый мониторинг (без воздействия)", "n": background,
         "color": "#9AA5AD", "note": "", "line": "длинная: окно ~100 км"},
        {"label": "Гармоника (динамик / встряхивание\nкатушки, 11-1000 Гц)", "n": counts.get("измерение_возмущение", 0),
         "color": "#4D77A8", "note": "", "line": "длинная: окно ~100 км\n(воздействие 1,3-12,1 км)"},
        {"label": "Нагрев участка (длинные линии)", "n": counts.get("some_test", 0),
         "color": "#C8853A", "note": "", "line": "реальная: 50-950 км"},
        {"label": "Статическая растяжка (грузы 0,4-1 кг)", "n": counts.get("растяжение", 0),
         "color": "#7A6A9B", "note": "", "line": "длинная: окно ~100 км"},
    ]
    return rows


def main() -> int:
    counts = count_parquet_by_dir()
    rows = aggregate(counts)
    total = sum(r["n"] for r in rows)
    rows.sort(key=lambda r: r["n"], reverse=True)

    labels = [r["label"] for r in rows]
    vals = [r["n"] for r in rows]
    colors = [r["color"] for r in rows]
    y = list(range(len(rows)))[::-1]  # top-to-bottom largest first

    maxv = max(vals)
    len_x = maxv * 1.34  # fixed x for the line-length column
    fig, ax = plt.subplots(figsize=(11.2, 4.7))
    bars = ax.barh(y, vals, color=colors, edgecolor="white", height=0.66)

    for yi, r in zip(y, rows):
        pct = 100.0 * r["n"] / total if total else 0.0
        ax.text(r["n"] + maxv * 0.012, yi, f"{r['n']}  ({pct:.0f} %)",
                va="center", ha="left", fontsize=11, fontweight="bold")
        if r["note"]:
            ax.text(r["n"] - maxv * 0.012, yi, r["note"],
                    va="center", ha="right", fontsize=9, color="white", fontstyle="italic")
        ax.text(len_x, yi, r["line"], va="center", ha="left", fontsize=9.5, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlabel("Число записей", fontsize=12)
    ax.set_xlim(0, maxv * 1.92)
    ax.set_title("Состав датасета phi-OTDR: тип воздействия и длина линии", fontsize=13, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"total records: {total}")
    for r in rows:
        print(f"  {r['n']:4d}  {r['label'].splitlines()[0]}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
