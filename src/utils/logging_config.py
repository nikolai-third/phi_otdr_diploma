"""Centralized logging configuration for the pipeline."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_path: str | Path = "logs/pipeline.log", level: int = logging.INFO) -> logging.Logger:
    """Configure root logger with console and file handlers.

    Reuses existing handlers on repeated calls.
    """

    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers when pipeline is resumed/reimported.
    existing_files = {
        getattr(h, "baseFilename", None) for h in root.handlers if hasattr(h, "baseFilename")
    }
    has_console = any(isinstance(h, logging.StreamHandler) and not hasattr(h, "baseFilename") for h in root.handlers)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    if str(log_file.resolve()) not in existing_files:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    if not has_console:
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(formatter)
        root.addHandler(sh)

    logger = logging.getLogger(__name__)
    logger.info("Logging initialized: %s", log_file)
    return logger
