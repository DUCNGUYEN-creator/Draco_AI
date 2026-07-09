# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
thinking_engine.exceptions
============================
Centralized exception hierarchy. Every subsystem raises a subclass of
``ThinkingEngineError`` so callers can catch broadly (``except
ThinkingEngineError``) or narrowly (``except VerifierError``).
"""

from __future__ import annotations


class ThinkingEngineError(Exception):
    """Base class for all Thinking Engine errors."""


# ── Infrastructure layer ────────────────────────────────────────────
class InfrastructureError(ThinkingEngineError):
    """Errors raised by Memory, Knowledge, Planning, or Tools subsystems."""


class MemoryError_(InfrastructureError):
    """Raised on memory read/write/eviction failures."""


class KnowledgeGraphError(InfrastructureError):
    """Raised on knowledge-graph traversal or mutation failures."""


class ToolExecutionError(InfrastructureError):
    """Raised when a tool call fails or returns invalid output."""


class PlanningError(InfrastructureError):
    """Raised when planning/decomposition cannot produce a usable plan."""


# ── Cognition layer ─────────────────────────────────────────────────
class CognitionError(ThinkingEngineError):
    """Errors raised by the Reasoning subsystem."""


class SearchError(CognitionError):
    """Raised on search-algorithm failures (BFS/DFS/A*/MCTS/Beam/...)."""


class DebateError(CognitionError):
    """Raised when multi-agent debate cannot reach a synthesis."""


class ReasoningBudgetExceeded(CognitionError):
    """Raised when a reasoning loop exceeds its allotted step/time budget."""


# ── Verification layer ──────────────────────────────────────────────
class VerificationError(ThinkingEngineError):
    """Base class for Reflection / Hallucination subsystem errors."""


class VerifierError(VerificationError):
    """Raised when an individual verifier fails to evaluate evidence."""


class CalibrationError(VerificationError):
    """Raised on calibration-model fit/predict failures."""


class CorrelationError(VerificationError):
    """Raised on evidence-correlation/deduplication failures."""


class FusionError(VerificationError):
    """Raised when fusing multiple verifier signals into one risk score fails."""


class ReflectionError(VerificationError):
    """Raised when self-reflection / critique / rewrite fails."""


class RegistryError(VerificationError):
    """Raised on registry lookup/registration conflicts (unknown key, duplicate)."""


class StrategyError(VerificationError):
    """Raised when an assessment strategy is misconfigured or unknown."""


class TelemetryError(VerificationError):
    """Raised on telemetry emission failures (non-fatal by design — callers
    should generally log and continue rather than propagate)."""


# ── Safety / Config ──────────────────────────────────────────────────
class SafetyError(ThinkingEngineError):
    """Raised by the safety/ subsystem (prompt-guard, injection, policy)."""


class ConfigError(ThinkingEngineError):
    """Raised on invalid engine configuration."""
