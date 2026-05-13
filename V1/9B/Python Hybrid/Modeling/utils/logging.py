# DracoAI V1 — modeling/utils/logging.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Structured logging helpers. No print() anywhere in the codebase."""
from __future__ import annotations
import logging
import sys
from typing import Optional

__all__ = ["get_logger", "configure_logging", "log_section"]

_FMT  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_DATE = "%H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a named logger prefixed with 'dracoai.'."""
    if not name.startswith("dracoai."):
        name = "dracoai." + name
    return logging.getLogger(name)


def configure_logging(
    level:   int  = logging.INFO,
    stream         = None,
    fmt:     str  = _FMT,
    datefmt: str  = _DATE,
    force:   bool = False,
) -> None:
    """Configure root DracoAI logger. Call once at startup."""
    root = logging.getLogger("dracoai")
    if root.handlers and not force:
        return
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False


class log_section:
    """Context manager that logs entry/exit of a named section."""

    def __init__(self, name: str,
                 logger: Optional[logging.Logger] = None,
                 level: int = logging.DEBUG):
        self._name   = name
        self._logger = logger or get_logger("utils.logging")
        self._level  = level

    def __enter__(self):
        self._logger.log(self._level, "[%s] start", self._name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self._logger.log(self._level, "[%s] done", self._name)
        else:
            self._logger.error("[%s] error: %s", self._name, exc_val)
        return False