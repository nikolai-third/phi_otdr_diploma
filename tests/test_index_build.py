from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.index.build import build_index


def test_build_index_groups_by_top_directory(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.parquet"

    df = pd.DataFrame(
        [
            {
                "path": "session_a/trace_01.npy",
                "abs_path": str((tmp_path / "session_a" / "trace_01.npy").resolve()),
                "size_bytes": 100,
                "mtime": pd.Timestamp("2024-01-01T00:00:00"),
                "ext": ".npy",
                "metadata_json": '{"shape": [10, 5]}'
            },
            {
                "path": "session_a/config.txt",
                "abs_path": str((tmp_path / "session_a" / "config.txt").resolve()),
                "size_bytes": 20,
                "mtime": pd.Timestamp("2024-01-01T00:01:00"),
                "ext": ".txt",
                "metadata_json": '{"text_encoding": "cp1251"}'
            },
            {
                "path": "session_b/raw.npz",
                "abs_path": str((tmp_path / "session_b" / "raw.npz").resolve()),
                "size_bytes": 200,
                "mtime": pd.Timestamp("2024-01-02T00:00:00"),
                "ext": ".npz",
                "metadata_json": '{"array_count": 2}'
            },
        ]
    )
    df.to_parquet(catalog_path, index=False)

    out_path = tmp_path / "dataset_index.parquet"
    build_index(catalog_path, out_path)

    index_df = pd.read_parquet(out_path)
    assert len(index_df) == 2
    assert set(index_df["inferred_format"]) == {"npy", "npz"}

    # Determinism check: second run yields same ids
    out_path2 = tmp_path / "dataset_index_second.parquet"
    build_index(catalog_path, out_path2)
    index_df2 = pd.read_parquet(out_path2)
    assert sorted(index_df["record_id"].tolist()) == sorted(index_df2["record_id"].tolist())

    row_a = index_df[index_df["inferred_format"] == "npy"].iloc[0]
    assert row_a["file_count"] == 2
    assert row_a["total_size_bytes"] == 120
