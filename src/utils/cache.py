"""Local file caching for unstable remote storage (e.g., WebDAV)."""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path

LOGGER = logging.getLogger(__name__)

SIZE_THRESHOLD_BYTES = 200 * 1024 * 1024  # 200MB
COPY_CHUNK_BYTES = 8 * 1024 * 1024


def _cache_key(path: Path) -> str:
    stat = path.stat()
    payload = f"{path.resolve()}|{stat.st_size}|{int(stat.st_mtime_ns)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


def ensure_local(path: str | Path, cache_dir: str | Path = "cache") -> Path:
    """Return local cached path, copying source file only once.

    Strategy:
    - File < 200MB: copy to local cache if missing.
    - File >= 200MB: copy to local cache on first access and reuse on subsequent calls.
    """

    src = Path(path)
    if not src.exists() or not src.is_file():
        raise FileNotFoundError(f"Source file not found: {src}")

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    key = _cache_key(src)
    dst = cache_root / f"{key}_{src.name}"

    if dst.exists() and dst.is_file():
        LOGGER.info("cache hit: %s -> %s", src, dst)
        return dst

    size = src.stat().st_size
    LOGGER.info("cache miss: copying %s (%.2f MB) -> %s", src, size / (1024**2), dst)

    if size < SIZE_THRESHOLD_BYTES:
        shutil.copy2(src, dst)
    else:
        # Streamed copy for large files to avoid RAM spikes.
        with src.open("rb") as rfp, dst.open("wb") as wfp:
            shutil.copyfileobj(rfp, wfp, length=COPY_CHUNK_BYTES)

    return dst
