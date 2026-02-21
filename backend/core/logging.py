"""
OSRS Flipping AI — Structured logging configuration.

Call ``configure_logging()`` once at application startup (before any
``logging.getLogger`` calls) to install the shared handler configuration.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional


_CONFIGURED = False

# Map string log-level names to logging constants
_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def configure_logging(
    level: Optional[str] = None,
    fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    """Configure the root logger exactly once.

    Parameters
    ----------
    level:
        Log level string (DEBUG, INFO, WARNING, …).  Falls back to the
        ``LOG_LEVEL`` environment variable, then ``INFO``.
    fmt:
        Log format string.
    datefmt:
        Date format string for the formatter.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    env_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    resolved = _LEVEL_MAP.get(level.upper() if level else env_level, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    root = logging.getLogger()
    root.setLevel(resolved)

    # Avoid adding duplicate handlers when uvicorn / gunicorn have already
    # installed their own.
    if not root.handlers:
        root.addHandler(handler)

    # Quieten noisy third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("pymongo").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper — equivalent to ``logging.getLogger(name)``."""
    return logging.getLogger(name)
