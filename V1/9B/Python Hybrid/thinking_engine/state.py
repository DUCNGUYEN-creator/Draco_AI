# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
thinking_engine.state
=======================
``ThinkingState`` is the single mutable object that flows through every
stage of the pipeline:

    Perception → Memory Retrieval → Knowledge Retrieval → Planning →
    Reasoning → Tool Execution → Reasoning Loop → Verifier Layer →
    Hallucination Assessment → Reflection → Answer Rewriter → Output

Each stage reads what it needs and writes its own namespaced field back.
This keeps stage implementations decoupled (a stage never reaches into
another stage's private scratch space) while still allowing the final
Engine to assemble one coherent response dict.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .constants import PIPELINE_STAGES


@dataclass
class StageTrace:
    stage: str
    started_at: float
    finished_at: Optional[float] = None
    error: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        if self.finished_at is None:
            return -1.0
        return (self.finished_at - self.started_at) * 1000.0


@dataclass
class ThinkingState:
    """Mutable carrier object passed through ``Pipeline.run``."""

    # ── Raw input ────────────────────────────────────────────────────
    question: str
    history: List[dict] = field(default_factory=list)
    memory_candidates: Optional[List[dict]] = None
    memory_summary: str = ""
    ltm_facts: List[dict] = field(default_factory=list)
    user_id: Optional[str] = None
    think_mode: bool = False
    force_system2: bool = False
    max_experts: Optional[int] = None

    # ── Perception outputs ───────────────────────────────────────────
    rewritten_query: str = ""
    intent: Dict[str, Any] = field(default_factory=dict)
    expert_boost: Dict[int, float] = field(default_factory=dict)
    miro_tau: float = 5.0
    topic_shift: bool = False

    # ── Memory retrieval outputs ─────────────────────────────────────
    reranked_memory: str = ""
    full_memory: str = ""

    # ── Knowledge retrieval outputs ──────────────────────────────────
    entities: List[str] = field(default_factory=list)
    reasoning_path: List[str] = field(default_factory=list)
    analogy_note: str = ""
    fact_issues: List[str] = field(default_factory=list)

    # ── Planning outputs ──────────────────────────────────────────────
    goal_decomposition: List[str] = field(default_factory=list)
    subgoals: List[str] = field(default_factory=list)
    instruction_chain: List[str] = field(default_factory=list)

    # ── Reasoning outputs ──────────────────────────────────────────────
    best_branch: str = ""
    all_branches: List[str] = field(default_factory=list)
    thoughts: List[str] = field(default_factory=list)
    debate_synthesis: str = ""
    debate_opinions: Dict[int, str] = field(default_factory=dict)
    used_experts: List[int] = field(default_factory=list)
    sc_path: str = ""
    abduction_hypotheses: List[str] = field(default_factory=list)
    counterfactual: str = ""
    hypothesis_result: Optional[dict] = None
    difficulty_score: float = 0.0
    process_mode: str = "fast"
    base_confidence: float = 0.5
    metaphor_note: str = ""
    spatial_note: str = ""

    # ── Tool execution outputs ───────────────────────────────────────
    tool_injection: str = ""
    tool_injection_active: bool = False
    zero_shot_tool: Optional[dict] = None
    parsed_tool_calls: List[dict] = field(default_factory=list)
    tool_results: List[dict] = field(default_factory=list)

    # ── Reasoning loop / verifier outputs ────────────────────────────
    cot_verification: Dict[str, Any] = field(default_factory=dict)

    # ── Hallucination assessment outputs ─────────────────────────────
    hallucination_report: Optional[Dict[str, Any]] = None

    # ── Reflection outputs ────────────────────────────────────────────
    reflection_report: Optional[Dict[str, Any]] = None
    clarification_needed: bool = False
    clarification_question: str = ""
    calibrated_confidence: float = 0.5
    ethical_warning: str = ""

    # ── Answer rewriter / output ──────────────────────────────────────
    messages: List[dict] = field(default_factory=list)
    final_answer: Optional[str] = None

    # ── Tracing ───────────────────────────────────────────────────────
    traces: List[StageTrace] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    # ── Helpers ───────────────────────────────────────────────────────
    def start_stage(self, stage: str) -> StageTrace:
        if stage not in PIPELINE_STAGES:
            raise ValueError(f"Unknown pipeline stage: {stage!r}")
        trace = StageTrace(stage=stage, started_at=time.time())
        self.traces.append(trace)
        return trace

    def finish_stage(self, trace: StageTrace, error: Optional[str] = None) -> None:
        trace.finished_at = time.time()
        trace.error = error

    def total_duration_ms(self) -> float:
        if not self.traces:
            return 0.0
        return sum(t.duration_ms for t in self.traces if t.duration_ms >= 0)

    def to_engine_output(self) -> Dict[str, Any]:
        """Assemble the final response dict — mirrors engine_v1.py's
        ``ThinkingEngineV1.process()`` return shape so existing callers
        (transformer bridge, CLI, tests) keep working unchanged."""
        thought_plan = {
            "best_branch": self.best_branch,
            "all_branches": self.all_branches,
            "thoughts": self.thoughts,
            "reasoning_path": self.reasoning_path,
            "confidence": self.base_confidence,
            "calibrated_confidence": self.calibrated_confidence,
            "debate_synthesis": self.debate_synthesis,
            "debate_opinions": self.debate_opinions,
            "sc_path": self.sc_path,
            "process_mode": self.process_mode,
            "subgoals": self.subgoals,
            "goal_decomposition": self.goal_decomposition,
            "instruction_chain": self.instruction_chain,
            "cot_verification": self.cot_verification,
            "tool_injection": self.tool_injection,
            "counterfactual": self.counterfactual,
            "analogy": self.analogy_note,
            "difficulty_score": self.difficulty_score,
            "metaphor_note": self.metaphor_note,
            "spatial_note": self.spatial_note,
            "abduction": self.abduction_hypotheses,
            "hypothesis": self.hypothesis_result,
            "ethical_warning": self.ethical_warning,
            "zero_shot_tool": self.zero_shot_tool,
            "topic_shift": self.topic_shift,
            "hallucination_report": self.hallucination_report,
            "reflection_report": self.reflection_report,
        }
        return {
            "intent": self.intent,
            "expert_boost": self.expert_boost,
            "miro_tau": self.miro_tau,
            "thought_plan": thought_plan,
            "messages": self.messages,
            "creativity": self.intent.get("creativity", 0.6),
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
