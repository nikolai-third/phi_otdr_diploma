from __future__ import annotations

from pathlib import Path

from src.utils.cache import ensure_local


def test_ensure_local_copies_and_reuses(tmp_path: Path) -> None:
    src = tmp_path / "a.bin"
    src.write_bytes(b"12345")

    cache_dir = tmp_path / "cache"
    first = ensure_local(src, cache_dir)
    second = ensure_local(src, cache_dir)

    assert first.exists()
    assert second.exists()
    assert first == second
    assert first.read_bytes() == b"12345"
