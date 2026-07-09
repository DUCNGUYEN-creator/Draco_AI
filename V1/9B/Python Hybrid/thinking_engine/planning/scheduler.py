# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
PlanScheduler
===============
Picks the next batch of ready tasks from a TaskGraph respecting a
parallelism cap — the Planning-layer counterpart of
routing/budget_router.py (which allocates *compute* budget; this
allocates *task-graph traversal order*).
"""

from __future__ import annotations

from typing import List

from .task_graph import TaskGraph, TaskNode


class PlanScheduler:
    def __init__(self, max_parallel: int = 2) -> None:
        self.max_parallel = max_parallel

    def next_batch(self, graph: TaskGraph) -> List[TaskNode]:
        return graph.ready_tasks()[: self.max_parallel]
