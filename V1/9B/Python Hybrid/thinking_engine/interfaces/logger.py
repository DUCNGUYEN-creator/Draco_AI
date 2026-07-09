# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""thinking_engine.interfaces.logger — contract for structured engine logging."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EngineLogger(Protocol):
    def debug(self, msg: str, **fields: Any) -> None:
        ...

    def info(self, msg: str, **fields: Any) -> None:
        ...

    def warning(self, msg: str, **fields: Any) -> None:
        ...

    def error(self, msg: str, **fields: Any) -> None:
        ...
