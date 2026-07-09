# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
thinking_engine.constants
==========================
Global constants shared across every layer of the Thinking Engine:
Infrastructure (Perception / Memory / Knowledge / Planning / Tools),
Cognition (Reasoning) and Verification (Reflection / Hallucination).

Ported 1:1 from the original monolithic ``engine_v1.py`` so downstream
behaviour (expert routing, intent classification, prompt scaffolding)
stays numerically identical after the split.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────
# Expert indices (8 experts from a single Qwen 3.5 9B Instruct FFN)
# ──────────────────────────────────────────────────────────────────────
EXPERT_CODE_0 = 0
EXPERT_CODE_1 = 1
EXPERT_CODE_2 = 2
EXPERT_CODE_3 = 3
EXPERT_LANG_0 = 4
EXPERT_LANG_1 = 5
EXPERT_LANG_2 = 6
EXPERT_LANG_3 = 7

# Aliases for readability
EXPERT_LOGIC = EXPERT_CODE_0
EXPERT_CODE = EXPERT_CODE_1
EXPERT_LANGUAGE = EXPERT_LANG_0
EXPERT_CHAT = EXPERT_LANG_1

N_EXPERTS = 8

# Apply as: logits += INTENT_BIAS_ALPHA * intent_bias
# Keeps router adaptive; prevents boost from dominating raw logits (~[-5, 5]).
INTENT_BIAS_ALPHA = 0.5

# ──────────────────────────────────────────────────────────────────────
# Intent types
# ──────────────────────────────────────────────────────────────────────
INTENT_MATH = "math"
INTENT_LOGIC = "logic"
INTENT_CODE = "code"
INTENT_CREATIVE = "creative"
INTENT_FACTUAL = "factual"
INTENT_HOW_TO = "how_to"
INTENT_WHY = "why"
INTENT_COMPARISON = "comparison"
INTENT_CHAT = "chat"
INTENT_MEMORY = "memory"

ALL_INTENTS = (
    INTENT_MATH,
    INTENT_LOGIC,
    INTENT_CODE,
    INTENT_CREATIVE,
    INTENT_FACTUAL,
    INTENT_HOW_TO,
    INTENT_WHY,
    INTENT_COMPARISON,
    INTENT_CHAT,
    INTENT_MEMORY,
)

# ──────────────────────────────────────────────────────────────────────
# Knowledge graph tuning
# ──────────────────────────────────────────────────────────────────────
KG_MIN_EDGE_WEIGHT = 0.05
KG_MAX_DEGREE = 50

# ──────────────────────────────────────────────────────────────────────
# Pipeline stage names — used by pipeline.py / state.py for tracing
# ──────────────────────────────────────────────────────────────────────
STAGE_PERCEPTION = "perception"
STAGE_MEMORY_RETRIEVAL = "memory_retrieval"
STAGE_KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
STAGE_PLANNING = "planning"
STAGE_REASONING = "reasoning"
STAGE_TOOL_EXECUTION = "tool_execution"
STAGE_REASONING_LOOP = "reasoning_loop"
STAGE_VERIFIER = "verifier_layer"
STAGE_HALLUCINATION = "hallucination_assessment"
STAGE_REFLECTION = "reflection"
STAGE_ANSWER_REWRITE = "answer_rewriter"
STAGE_OUTPUT = "output"

PIPELINE_STAGES = (
    STAGE_PERCEPTION,
    STAGE_MEMORY_RETRIEVAL,
    STAGE_KNOWLEDGE_RETRIEVAL,
    STAGE_PLANNING,
    STAGE_REASONING,
    STAGE_TOOL_EXECUTION,
    STAGE_REASONING_LOOP,
    STAGE_VERIFIER,
    STAGE_HALLUCINATION,
    STAGE_REFLECTION,
    STAGE_ANSWER_REWRITE,
    STAGE_OUTPUT,
)

# ──────────────────────────────────────────────────────────────────────
# Hallucination risk levels (used throughout reflection/hallucination/*)
# ──────────────────────────────────────────────────────────────────────
RISK_NONE = "none"
RISK_LOW = "low"
RISK_MEDIUM = "medium"
RISK_HIGH = "high"
RISK_CRITICAL = "critical"

RISK_ORDER = (RISK_NONE, RISK_LOW, RISK_MEDIUM, RISK_HIGH, RISK_CRITICAL)

# ──────────────────────────────────────────────────────────────────────
# Evidence source sentinel (NOT the same as RISK_NONE — distinct concept)
# ──────────────────────────────────────────────────────────────────────
EVIDENCE_NONE = "no_evidence"

# ──────────────────────────────────────────────────────────────────────
# System prompt (Qwen 3.5 9B ChatML) — identical to engine_v1.py
# ──────────────────────────────────────────────────────────────────────
DRACO_SYSTEM_PROMPT = """\
You are DracoAI, an intelligent local language model created by DUCNGUYEN-creator.
You are NOT Qwen, NOT an Alibaba product. You are DracoAI — fully independent.

Architecture: Qwen 3.5 9B Instruct weights → 8-expert MoE (DracoAI custom)
    Source: ONE Qwen 3.5 9B Instruct checkpoint (no separate split model).
    Code experts  (0-3): FFN layers activating on code/math/logic tokens
    Language experts (4-7): FFN layers activating on language/instruction tokens

Capabilities:
    - Bilingual (Vietnamese + English), respond in the user's language
    - Chain-of-thought reasoning with [PLAN][THOUGHT][FINAL ANSWER] structure
    - Long-term semantic vector memory
    - Confidence scoring: mark uncertain parts with [?]
    - Tool use via <tool_call>...</tool_call> tags

Guidelines:
    - Be accurate, concise, helpful
    - For code: include explanation after code block
    - For math: show step-by-step working
    - Never fabricate facts; say "I'm not sure" when uncertain
    - Always maintain DracoAI identity — never claim to be Qwen or Alibaba
"""
