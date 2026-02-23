from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.index.prepare import build_processing_index


def test_build_processing_index(tmp_path: Path) -> None:
    cat = tmp_path / "catalog.parquet"
    pd.DataFrame(
        [
            {
                "path": "group1/f1.npy",
                "abs_path": str((tmp_path / "group1" / "f1.npy")),
                "size_bytes": 1024,
                "ext": ".npy",
                "metadata_detected": True,
                "mtime": "2024-01-01",
            }
        ]
    ).to_parquet(cat, index=False)

    out = tmp_path / "index.parquet"
    build_processing_index(cat, out)

    df = pd.read_parquet(out)
    assert len(df) == 1
    assert df.iloc[0]["file_group"] == "group1"
    assert df.iloc[0]["inferred_format"] == "npy"
    assert bool(df.iloc[0]["has_metadata"]) is True
