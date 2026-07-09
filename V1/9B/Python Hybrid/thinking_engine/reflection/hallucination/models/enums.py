# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Enumerations shared across the Hallucination subsystem."""

from __future__ import annotations

from enum import Enum


class RiskLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def from_score(cls, score: float, thresholds: dict) -> "RiskLevel":
        if score >= thresholds.get("critical", 0.85):
            return cls.CRITICAL
        if score >= thresholds.get("high", 0.60):
            return cls.HIGH
        if score >= thresholds.get("medium", 0.35):
            return cls.MEDIUM
        if score >= thresholds.get("low", 0.15):
            return cls.LOW
        return cls.NONE


class EvidenceType(str, Enum):
    KNOWLEDGE_GRAPH = "knowledge_graph"
    RAG_DOCUMENT = "rag_document"
    MEMORY = "memory"
    TOOL_RESULT = "tool_result"
    REASONING_TRACE = "reasoning_trace"
    USER_PROVIDED = "user_provided"
    NONE = "none"


class VerifierKind(str, Enum):
    RETRIEVAL = "retrieval"
    REASONING = "reasoning"
    CONSISTENCY = "consistency"
    CONTRADICTION = "contradiction"
    NUMERICAL = "numerical"
    SYMBOLIC = "symbolic"
    CITATION = "citation"
    PLANNER = "planner"
    TOOL = "tool"
    ENSEMBLE = "ensemble"
