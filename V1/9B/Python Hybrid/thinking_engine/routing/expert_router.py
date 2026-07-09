# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ExpertRouter
==============
Top-level routing facade composing IntentDetector's expert-boost output
with SelfEvolvingRouter and ExpertLoadBalancer — the same three-step
chain that was inlined inside engine_v1.py's ``process()``:
    expert_boost = detector.to_expert_boost(intent)
    expert_boost = evolving_router.apply(intent, expert_boost)
    expert_boost = load_balancer.balanced_boost(expert_boost)
Formalized here so Routing is callable as one stage instead of three
loose calls scattered through the engine.
"""

from __future__ import annotations

import threading
from typing import Any, Dict

from ..perception.understanding.intent_detector import IntentDetector
from .evolving_router import SelfEvolvingRouter
from .load_balancer import ExpertLoadBalancer


class ExpertRouter:
    def __init__(
        self,
        detector: IntentDetector,
        evolving_router: SelfEvolvingRouter,
        load_balancer: ExpertLoadBalancer,
    ) -> None:
        self.detector = detector
        self.evolving_router = evolving_router
        self.load_balancer = load_balancer
        self._router_lock = threading.Lock()

    def route(self, intent: Dict[str, Any]) -> Dict[int, float]:
        expert_boost = self.detector.to_expert_boost(intent)
        with self._router_lock:
            expert_boost = self.evolving_router.apply(intent["intent"], expert_boost)
        expert_boost = self.load_balancer.balanced_boost(expert_boost)
        return expert_boost

    def record_feedback(
        self, intent_type: str, expert_id: int, success: bool, rating: float
    ) -> None:
        with self._router_lock:
            self.evolving_router.update(intent_type, expert_id, success)
        self.load_balancer.update_score(expert_id, rating)
