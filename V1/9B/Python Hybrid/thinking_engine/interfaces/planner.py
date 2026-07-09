# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""thinking_engine.interfaces.planner — contract for goal/plan decomposers."""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class Planner(Protocol):
    def decompose(
        self,
        goal: str,
        intent: Dict[str, Any],
        max_steps: int = 6,
        bridge: Any = None,
        tokenizer: Any = None,
    ) -> List[str]:
        ...
