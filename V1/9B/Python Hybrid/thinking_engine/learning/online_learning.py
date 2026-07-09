# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
OnlineLearner
==============
Coordinates calibrator updates and router updates from a single
feedback event — the place that knows "this feedback about (question,
answer) maps to these verifier scores" without either FeedbackCollector
or RouterUpdater needing to know about each other.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class OnlineLearner:
    def __init__(
        self,
        assessor=None,          # reflection.hallucination.Assessor
        router_updater=None,    # learning.RouterUpdater
    ) -> None:
        self._assessor = assessor
        self._router_updater = router_updater

    def on_feedback(self, feedback: Dict[str, Any]) -> None:
        rating = feedback.get("rating", 0.5)
        was_correct = rating >= 0.5

        if self._assessor is not None:
            self._assessor.record_outcome(
                claim=feedback.get("question", ""),
                raw_verifier_scores={},   # populated by caller if available
                was_correct=was_correct,
            )

        if self._router_updater is not None:
            self._router_updater.update_from_feedback(feedback, was_correct)
