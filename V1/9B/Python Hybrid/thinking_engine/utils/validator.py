# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Lightweight runtime validation helpers (no pydantic dependency)."""

from __future__ import annotations

from typing import Any, Iterable


def require_in_range(value: float, lo: float, hi: float, name: str = "value") -> float:
    if not (lo <= value <= hi):
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {value}")
    return value


def require_one_of(value: Any, choices: Iterable[Any], name: str = "value") -> Any:
    choices = list(choices)
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")
    return value


def require_non_empty(text: str, name: str = "text") -> str:
    if text is None or not str(text).strip():
        raise ValueError(f"{name} must be non-empty")
    return text
