# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ExecutionPlan
================
A finalized, ordered, time-stamped record of what was planned vs. what
actually executed — handed to the Tool Execution stage and later to
Reflection (so ``answer_rewriter.py`` can say "step 3 of the plan
wasn't completed" instead of silently dropping a sub-goal).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .task_graph import TaskGraph


@dataclass
class ExecutionPlan:
    goal: str
    graph: TaskGraph
    started_at: float = field(default_factory=time.time)
    step_results: Dict[str, Any] = field(default_factory=dict)

    def record_result(self, task_id: str, result: Any) -> None:
        self.step_results[task_id] = result
        self.graph.mark_done(task_id)

    def summary(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "complete": self.graph.is_complete(),
            "steps_completed": len(self.step_results),
            "elapsed_seconds": round(time.time() - self.started_at, 3),
        }
