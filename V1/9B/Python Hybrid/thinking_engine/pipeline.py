# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
thinking_engine.pipeline
==========================
The Global Pipeline — runs a ``ThinkingState`` through the canonical
12-stage sequence:

    Perception → Memory Retrieval → Knowledge Retrieval → Planning →
    Reasoning → Tool Execution → Reasoning Loop → Verifier Layer →
    Hallucination Assessment → Reflection → Answer Rewriter → Output

This is a direct, structured continuation of engine_v1.py's
``ThinkingEngineV1.process()`` method — every stage below corresponds
to a section of that original method, now callable and testable in
isolation. ``engine.py``'s ``ThinkingEngine`` class owns one
``Pipeline`` instance and delegates ``process()`` to ``Pipeline.run()``.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from .config import EngineConfig
from .constants import (
    INTENT_CHAT,
    INTENT_CODE,
    INTENT_LOGIC,
    INTENT_MATH,
    STAGE_ANSWER_REWRITE,
    STAGE_HALLUCINATION,
    STAGE_KNOWLEDGE_RETRIEVAL,
    STAGE_MEMORY_RETRIEVAL,
    STAGE_OUTPUT,
    STAGE_PERCEPTION,
    STAGE_PLANNING,
    STAGE_REASONING,
    STAGE_REASONING_LOOP,
    STAGE_REFLECTION,
    STAGE_TOOL_EXECUTION,
    STAGE_VERIFIER,
)
from .state import ThinkingState


class Pipeline:
    """Owns references to one instance of every subsystem component and
    runs a ThinkingState through all 12 stages in order. Constructed and
    wired up by ``engine.ThinkingEngine.__init__`` — Pipeline itself
    contains no component-construction logic, only orchestration, so
    engine.py stays the single place dependency wiring happens.
    """

    def __init__(self, components: "EngineComponents", config: EngineConfig) -> None:
        self.c = components
        self.config = config

    # ── Stage 1: Perception ──────────────────────────────────────────
    def _stage_perception(self, state: ThinkingState) -> None:
        trace = state.start_stage(STAGE_PERCEPTION)
        try:
            question = self.c.prompt_sanitizer.sanitize(state.question)
            question = self.c.contextual_rewriter.rewrite(question, state.history)
            state.rewritten_query = question

            intent = self.c.intent_detector.detect(question)
            if state.user_id:
                intent = self.c.user_profile_manager.apply_to_intent(state.user_id, intent)
            state.intent = intent

            state.expert_boost = self.c.expert_router.route(intent)
            state.miro_tau = self.c.intent_detector.to_miro_tau(intent)

            metaphor = self.c.metaphor_detector.detect(question)
            if metaphor:
                state.metaphor_note = metaphor

            state.difficulty_score = self.c.difficulty_scorer.score(question, intent)
            state.process_mode = self.c.dual_process_decider.decide_mode(intent, question)
            if state.force_system2 or state.difficulty_score >= 0.65:
                state.process_mode = "slow"

            state.base_confidence = 0.85 if state.process_mode == "fast" else 0.65
        finally:
            state.finish_stage(trace)

    # ── Stage 2: Memory Retrieval ─────────────────────────────────────
    def _stage_memory_retrieval(self, state: ThinkingState) -> None:
        trace = state.start_stage(STAGE_MEMORY_RETRIEVAL)
        try:
            state.full_memory = self.c.memory_retrieval.retrieve(
                query=state.rewritten_query,
                intent=state.intent,
                memory_candidates=state.memory_candidates,
                memory_summary=state.memory_summary,
                top_k=self.config.memory.rerank_top_k,
            )
        finally:
            state.finish_stage(trace)

    # ── Stage 3: Knowledge Retrieval ──────────────────────────────────
    def _stage_knowledge_retrieval(self, state: ThinkingState) -> None:
        trace = state.start_stage(STAGE_KNOWLEDGE_RETRIEVAL)
        try:
            state.entities = state.intent.get("entities", [])

            rag_result = self.c.rag_pipeline.run(state.rewritten_query, state.intent)
            if rag_result.get("applicable") and rag_result.get("context_text"):
                state.full_memory = (state.full_memory + " [RAG] " + rag_result["context_text"]).strip(" |")

            if len(state.entities) >= 2:
                analogy_target = self.c.analogical_mapper.find_analogy(
                    self.c.kg, state.entities[0], state.entities[1],
                    state.entities[1] if len(state.entities) < 3 else state.entities[2],
                )
                state.analogy_note = self.c.analogical_mapper.describe_analogy(
                    state.entities[0], state.entities[1],
                    state.entities[1] if len(state.entities) < 3 else state.entities[2],
                    analogy_target,
                )

            if self.c.counterfactual_reasoner.is_applicable(state.rewritten_query, state.intent):
                state.counterfactual = self.c.counterfactual_reasoner.generate(
                    state.rewritten_query, state.intent, self.c.bridge, self.c.tokenizer
                )

            if self.c.abduction_engine.is_applicable(state.rewritten_query, state.intent):
                state.abduction_hypotheses = self.c.abduction_engine.generate_hypotheses(
                    state.rewritten_query, self.c.kg, state.intent,
                    bridge=self.c.bridge, tokenizer=self.c.tokenizer,
                )
        finally:
            state.finish_stage(trace)

    # ── Stage 4: Planning ─────────────────────────────────────────────
    def _stage_planning(self, state: ThinkingState) -> None:
        trace = state.start_stage(STAGE_PLANNING)
        try:
            if self.c.tool_registry.should_use_tools(state.intent, state.rewritten_query):
                state.tool_injection = self.c.tool_context_builder.build_tool_injection(
                    state.intent, state.rewritten_query
                )
                state.tool_injection_active = bool(state.tool_injection)
        finally:
            state.finish_stage(trace)

    # ── Stage 5+6+7: Reasoning / Tool Execution / Reasoning Loop ──────
    def _stage_reasoning(self, state: ThinkingState) -> None:
        trace = state.start_stage(STAGE_REASONING)
        try:
            budget = self.c.budget_router.allocate(state.process_mode, state.difficulty_score)
            max_experts = state.max_experts or self.config.max_experts

            result = self.c.reasoning_controller.run(
                question=state.rewritten_query,
                intent=state.intent,
                think_mode=state.think_mode,
                process_mode=state.process_mode,
                base_conf=state.base_confidence,
                max_experts=max_experts,
                bridge=self.c.bridge,
                tokenizer=self.c.tokenizer,
                llm_generate_fn=self.c.llm_generate_fn,
                max_council_rounds=budget.get("debate_rounds", self.config.reasoning.max_council_rounds),
                self_consistency_paths=budget.get(
                    "self_consistency_paths", self.config.reasoning.self_consistency_paths
                ),
            )
            state.best_branch = result["best_branch"]
            state.all_branches = result["all_branches"]
            state.debate_synthesis = result["debate_synthesis"]
            state.debate_opinions = result["debate_opinions"]
            state.used_experts = result["used_experts"]
            state.goal_decomposition = result["goal_decomposition"]
            state.subgoals = result["subgoals"]
            state.sc_path = result["sc_path"]

            state.reasoning_path = self.c.reasoning_path_computer.compute(self.c.kg, state.entities)

            state.thoughts = [state.best_branch] if state.best_branch else []
            if state.subgoals:
                state.thoughts.extend(state.subgoals[:2])
        finally:
            state.finish_stage(trace)

    def _stage_tool_execution(self, state: ThinkingState) -> None:
        trace = state.start_stage(STAGE_TOOL_EXECUTION)
        try:
            # In this offline pipeline (no live LLM output yet at this
            # point), tool calls are only executed if the caller pre-parsed
            # them from a previous turn's output (zero_shot_tool path) —
            # the main <tool_call> extraction happens post-generation in
            # ResponseBuilder / the caller's own generate-then-execute loop.
            if state.zero_shot_tool:
                state.tool_results = self.c.tool_executor.execute([state.zero_shot_tool])
        finally:
            state.finish_stage(trace)

    def _stage_reasoning_loop(self, state: ThinkingState) -> None:
        trace = state.start_stage(STAGE_REASONING_LOOP)
        try:
            if state.thoughts:
                state.thoughts, verification = self.c.recursive_reflection_loop.run(state.thoughts)
                state.cot_verification = verification
        finally:
            state.finish_stage(trace)

    # ── Stage 8: Verifier Layer ────────────────────────────────────────
    def _stage_verifier(self, state: ThinkingState) -> None:
        trace = state.start_stage(STAGE_VERIFIER)
        try:
            if not state.cot_verification and state.thoughts:
                state.cot_verification = self.c.chain_verifier.verify_thoughts(state.thoughts)
        finally:
            state.finish_stage(trace)

    # ── Stage 9: Hallucination Assessment ──────────────────────────────
    def _stage_hallucination(self, state: ThinkingState) -> None:
        trace = state.start_stage(STAGE_HALLUCINATION)
        try:
            if state.final_answer:
                # reasoning_paths feeds ConsistencyVerifier's "does the
                # final answer agree with multiple INDEPENDENT reasoning
                # paths" check. Only include items that are genuinely
                # comparable in content/length to a final answer —
                # debate opinions (full per-expert responses) and the
                # self-consistency vote path. Deliberately EXCLUDES
                # all_branches: those are short strategic thinking-
                # approach labels generated before reasoning even ran
                # (e.g. "Break down into steps"), not candidate answers,
                # so comparing final_answer against them via lexical
                # overlap produces a structurally low, meaningless score
                # that would falsely look like "low self-consistency".
                reasoning_paths: List[str] = list(state.debate_opinions.values())
                if state.sc_path:
                    reasoning_paths.append(state.sc_path)

                context = {
                    "reasoning_path": state.reasoning_path,
                    "reasoning_paths": reasoning_paths,
                    "memory_summary": state.full_memory,
                    "tool_results": state.tool_results,
                    "subgoals": state.subgoals,
                    "goal_decomposition": state.goal_decomposition,
                    "debate_opinions": state.debate_opinions,
                    "best_branch": state.best_branch,
                }
                report = self.c.hallucination_assessor.assess(
                    answer=state.final_answer,
                    context=context,
                    strategy_name=self.config.hallucination.strategy,
                )
                state.hallucination_report = report.as_dict()
        finally:
            state.finish_stage(trace)

    # ── Stage 10: Reflection ────────────────────────────────────────────
    def _stage_reflection(self, state: ThinkingState) -> None:
        trace = state.start_stage(STAGE_REFLECTION)
        try:
            if state.final_answer:
                refined_answer, reports = self.c.critic.recursive_critique(
                    state.rewritten_query, state.final_answer, state.ltm_facts
                )
                post_check = self.c.critic.post_generation_check(
                    refined_answer, state.rewritten_query,
                    self.c.kg, self.c.temporal_kg,
                    state.base_confidence, state.ltm_facts,
                )
                state.fact_issues = post_check["fact_issues"]
                state.ethical_warning = post_check["ethical_note"]
                state.calibrated_confidence = post_check["calibrated_conf"]
                state.final_answer = post_check["tagged_answer"]

                consistency = self.c.consistency_checker.check(
                    state.final_answer,
                    {"best_branch": state.best_branch, "debate_synthesis": state.debate_synthesis},
                )
                if not consistency["consistent"]:
                    state.fact_issues.extend(consistency["issues"])

            if self.c.active_learning_loop.needs_clarification(state.calibrated_confidence, state.intent):
                state.clarification_needed = True
                state.clarification_question = self.c.active_learning_loop.generate_clarification(
                    state.rewritten_query, state.intent
                )
        finally:
            state.finish_stage(trace)

    # ── Stage 11: Answer Rewriter ────────────────────────────────────────
    def _stage_answer_rewrite(self, state: ThinkingState) -> None:
        trace = state.start_stage(STAGE_ANSWER_REWRITE)
        try:
            is_ethical = not bool(state.ethical_warning)
            if self.c.answer_rewriter.needs_rewrite(state.hallucination_report, is_ethical):
                rewrite_note = self.c.answer_rewriter.build_rewrite_instruction(
                    state.hallucination_report, state.ethical_warning
                )
                state.messages.append({"role": "system", "content": rewrite_note})
        finally:
            state.finish_stage(trace)

    # ── Stage 12: Output ──────────────────────────────────────────────────
    def _stage_output(self, state: ThinkingState) -> None:
        trace = state.start_stage(STAGE_OUTPUT)
        try:
            thought_plan_for_compiler = state.to_engine_output()["thought_plan"]
            state.messages = self.c.prompt_compiler.compile(
                question=state.rewritten_query,
                intent=state.intent,
                thought_plan=thought_plan_for_compiler,
                memory_summary=state.full_memory,
                history=state.history,
            )
            state.messages = self.c.context_window_manager.manage(state.messages)
        finally:
            state.finish_stage(trace)

    # ── Public entry point ──────────────────────────────────────────────
    def run(self, state: ThinkingState) -> ThinkingState:
        """Runs stages 1-8 + 11-12 unconditionally (prompt assembly path).
        Stages 9-10 (Hallucination Assessment, Reflection) only produce
        meaningful output once ``state.final_answer`` has been set by the
        caller after receiving the LLM's generation for the messages this
        pipeline assembles — mirroring engine_v1.py's two-phase flow
        (compile prompt -> generate -> recursive_critique/post_generation_check).
        """
        self._stage_perception(state)
        self._stage_memory_retrieval(state)
        self._stage_knowledge_retrieval(state)
        self._stage_planning(state)
        self._stage_reasoning(state)
        self._stage_tool_execution(state)
        self._stage_reasoning_loop(state)
        self._stage_verifier(state)
        self._stage_hallucination(state)   # no-op until state.final_answer is set
        self._stage_reflection(state)      # no-op until state.final_answer is set
        self._stage_answer_rewrite(state)
        self._stage_output(state)
        return state

    def run_post_generation(self, state: ThinkingState, final_answer: str) -> ThinkingState:
        """Second-phase entry point: call this after the caller has
        generated a response using state.messages, to run Hallucination
        Assessment + Reflection + Answer Rewriter against the actual
        generated text."""
        state.final_answer = final_answer
        self._stage_hallucination(state)
        self._stage_reflection(state)
        self._stage_answer_rewrite(state)
        return state
