"""Lightweight logging setup."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from .io import ensure_dir


def setup_logger(name: str, log_dir: Optional[Union[str, Path]] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        ensure_dir(log_dir)
        file_handler = logging.FileHandler(Path(log_dir) / f"{name}.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
