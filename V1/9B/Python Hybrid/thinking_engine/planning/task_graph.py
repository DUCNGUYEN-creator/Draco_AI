# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
TaskGraph
===========
DAG of plan steps with explicit dependency edges (step B depends on
step A completing first). GoalDecomposer/PlanDecomposer emit a flat
ordered list; TaskGraph is the structured form Tools/Execution stages
consume when steps can run out of strict textual order (e.g. two
independent sub-goals that don't depend on each other).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set

from ..exceptions import PlanningError


@dataclass
class TaskNode:
    id: str
    description: str
    depends_on: List[str] = field(default_factory=list)
    done: bool = False


class TaskGraph:
    def __init__(self) -> None:
        self._nodes: Dict[str, TaskNode] = {}

    def add_task(self, task_id: str, description: str, depends_on: List[str] | None = None) -> TaskNode:
        node = TaskNode(id=task_id, description=description, depends_on=depends_on or [])
        self._nodes[task_id] = node
        return node

    @classmethod
    def from_ordered_steps(cls, steps: List[str]) -> "TaskGraph":
        """Build a simple linear-dependency graph from PlanDecomposer's
        flat ordered list — step i depends on step i-1."""
        graph = cls()
        prev_id = None
        for i, step in enumerate(steps):
            tid = f"step_{i+1}"
            graph.add_task(tid, step, depends_on=[prev_id] if prev_id else [])
            prev_id = tid
        return graph

    def ready_tasks(self) -> List[TaskNode]:
        """Tasks whose dependencies are all done and that aren't done yet."""
        return [
            n for n in self._nodes.values()
            if not n.done and all(self._nodes[d].done for d in n.depends_on if d in self._nodes)
        ]

    def mark_done(self, task_id: str) -> None:
        if task_id not in self._nodes:
            raise PlanningError(f"Unknown task id: {task_id}")
        self._nodes[task_id].done = True

    def is_complete(self) -> bool:
        return all(n.done for n in self._nodes.values())

    def topological_order(self) -> List[str]:
        visited: Set[str] = set()
        order: List[str] = []

        def visit(tid: str) -> None:
            if tid in visited:
                return
            visited.add(tid)
            for dep in self._nodes[tid].depends_on:
                if dep in self._nodes:
                    visit(dep)
            order.append(tid)

        for tid in self._nodes:
            visit(tid)
        return order
