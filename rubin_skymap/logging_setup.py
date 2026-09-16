"""Logging configuration for rubin-skymap."""

from __future__ import annotations

import logging


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with a standard format.

    Parameters
    ----------
    level:
        A logging level string such as ``"DEBUG"``, ``"INFO"``, ``"WARNING"``.
        Case-insensitive.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # Silence noisy third-party loggers.
    for noisy in ("httpx", "httpcore", "urllib3", "lightgbm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
