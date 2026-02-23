from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.audit.scan import run_scan

h5py = pytest.importorskip("h5py")


def _create_sample_files(root: Path) -> None:
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.save(root / "sample.npy", arr)

    np.savez(root / "sample.npz", first=arr, second=np.ones((2, 2), dtype=np.int16))

    with h5py.File(root / "sample.h5", "w") as h5_file:
        h5_file.attrs["source"] = "unit-test"
        h5_file.create_dataset("trace", data=arr)

    text = "Название: Тестовый файл\nОператор=Иванов\n"
    (root / "meta.txt").write_bytes(text.encode("cp1251"))


def test_scan_collects_metadata_and_writes_outputs(tmp_path: Path) -> None:
    data_root = tmp_path / "raw"
    data_root.mkdir(parents=True)
    _create_sample_files(data_root)

    out = tmp_path / "interim" / "catalog.parquet"
    parquet_path, csv_path, summary_path = run_scan(
        root=data_root,
        out=out,
        max_files=None,
        sample_bytes=64,
        qc_max_bytes=5_000_000,
        parquet_schema_max_bytes=2_000_000,
        workers=2,
    )

    assert parquet_path.exists()
    assert csv_path.exists()
    assert summary_path.exists()

    df = pd.read_parquet(parquet_path)
    assert len(df) == 4

    by_ext = {ext: row for ext, row in zip(df["ext"], df.to_dict(orient="records"), strict=False)}

    assert ".npy" in by_ext
    npy_meta = json.loads(by_ext[".npy"]["metadata_json"])
    assert tuple(npy_meta["shape"]) == (3, 4)
    assert "dtype" in npy_meta

    assert ".npz" in by_ext
    npz_meta = json.loads(by_ext[".npz"]["metadata_json"])
    assert npz_meta["array_count"] == 2

    assert ".h5" in by_ext
    h5_meta = json.loads(by_ext[".h5"]["metadata_json"])
    assert h5_meta["dataset_count"] == 1
    assert "root_attrs" in h5_meta

    assert ".txt" in by_ext
    txt_meta = json.loads(by_ext[".txt"]["metadata_json"])
    assert "text_encoding" in txt_meta
    assert "Тестовый файл" in txt_meta["text_preview"]
