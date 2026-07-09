# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
FeedbackCollector
==================
Receives user-facing feedback (thumbs up/down, correction text) and
routes it to the right downstream handler: RouterUpdater for expert
routing signals, Assessor.record_outcome for calibration, and
ExperienceBuffer for long-term pattern learning.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional


class FeedbackCollector:
    def __init__(self) -> None:
        self._handlers: List[Callable] = []
        self._lock = threading.Lock()
        self._history: List[Dict[str, Any]] = []

    def register_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        with self._lock:
            self._handlers.append(handler)

    def submit(
        self,
        question: str,
        answer: str,
        rating: float,            # 0.0 = very bad, 1.0 = perfect
        correction: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        feedback = {
            "question": question,
            "answer": answer,
            "rating": rating,
            "correction": correction,
            "metadata": metadata or {},
        }
        with self._lock:
            self._history.append(feedback)
            handlers = list(self._handlers)
        for handler in handlers:
            try:
                handler(feedback)
            except Exception:
                pass  # feedback handlers must never break the main flow

    def recent(self, n: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            return self._history[-n:]
