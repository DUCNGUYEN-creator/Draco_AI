# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
EngineOutput
=============
The final, immutable output object returned by ThinkingEngine.process()
— a structured wrapper around both the engine's internal state snapshot
(thought_plan, intent, expert_boost, miro_tau, ...) and the display-
ready formatted text. Replaces the raw dict that engine_v1.py returned
so callers can type-check the output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EngineOutput:
    """Mirrors the dict shape engine_v1.py returned from process() so
    existing callers (transformer bridge, CLI, tests) keep working
    unchanged when they unpack this via .__dict__ or .get()."""

    # Display output
    formatted_text: str = ""
    calibrated_confidence: float = 0.5

    # Pipeline diagnostics (retained for transformer bridge integration)
    intent: Dict[str, Any] = field(default_factory=dict)
    expert_boost: Dict[int, float] = field(default_factory=dict)
    miro_tau: float = 5.0
    thought_plan: Dict[str, Any] = field(default_factory=dict)
    messages: List[dict] = field(default_factory=list)
    creativity: float = 0.6
    rewritten_query: str = ""
    process_mode: str = "fast"
    difficulty_score: float = 0.0
    clarification_needed: bool = False
    clarification_question: str = ""
    cot_verification: Dict[str, Any] = field(default_factory=dict)
    tool_injection_active: bool = False
    topic_shift: bool = False
    ethical_warning: str = ""
    hallucination_report: Optional[Dict[str, Any]] = None

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "expert_boost": self.expert_boost,
            "miro_tau": self.miro_tau,
            "thought_plan": self.thought_plan,
            "messages": self.messages,
            "creativity": self.creativity,
            "rewritten_query": self.rewritten_query,
            "process_mode": self.process_mode,
            "difficulty_score": self.difficulty_score,
            "clarification_needed": self.clarification_needed,
            "clarification_question": self.clarification_question,
            "cot_verification": self.cot_verification,
            "tool_injection_active": self.tool_injection_active,
            "calibrated_confidence": self.calibrated_confidence,
            "topic_shift": self.topic_shift,
            "ethical_warning": self.ethical_warning,
            "hallucination_report": self.hallucination_report,
        }
