# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""thinking_engine.interfaces.tool — contract every callable tool must satisfy."""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str

    def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool. Must return {"output": ..., "ok": bool}."""
        ...
