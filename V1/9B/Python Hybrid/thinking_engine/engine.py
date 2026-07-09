# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
thinking_engine.engine
========================
``ThinkingEngine`` is the top-level facade — the direct structural
successor to the monolithic ``engine_v1.py``'s ``ThinkingEngineV1``.
It owns exactly one instance of every subsystem component (wired
together in ``EngineComponents``), owns one ``Pipeline``, and exposes
a ``process()`` method whose input/output contract matches the
original engine so callers (the transformer bridge, CLI tools, tests)
require no changes beyond the import path.

Usage
-----
    from thinking_engine import ThinkingEngine

    engine = ThinkingEngine(numpy_model=my_model)   # or leave None for stub bridge
    out = engine.process("Tính 2 + 2 * 3", history=[])
    print(out.formatted_text)

    # Two-phase flow (mirrors engine_v1.py's compile -> generate -> critique):
    out = engine.process("...", history=[])          # produces out.messages to send to the LLM
    # ... caller generates a response using out.messages ...
    final = engine.finish("...", history=[], generated_text="the model's raw output")
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .config import EngineConfig
from .constants import DRACO_SYSTEM_PROMPT
from .execution.response_builder import ResponseBuilder
from .execution.formatter import ResponseFormatter
from .execution.output import EngineOutput
from .interfaces.llm import TransformerBridgeAdapter, load_default_bridge
from .knowledge.bayesian_updater import BayesianBeliefUpdater
from .knowledge.fact_checker import FactConsistencyChecker
from .knowledge.knowledge_graph import KnowledgeGraph
from .knowledge.rag import RAGPipeline
from .knowledge.temporal_graph import TemporalKnowledgeGraph
from .learning.experience import ExperienceBuffer
from .learning.feedback import FeedbackCollector
from .learning.statistics import LearningStats
from .memory.context_window import ContextWindowManager
from .memory.memory_retrieval import MemoryRetrieval
from .memory.user_profile import UserProfileManager
from .perception.language.metaphor_detector import MetaphorDetector
from .perception.prompt.compiler import PromptCompiler
from .perception.prompt.contextual_rewriter import ContextualPromptRewriter
from .perception.prompt.sanitizer import PromptSanitizer
from .perception.understanding.difficulty import DifficultyScorer
from .perception.understanding.dual_process import DualProcessDecider
from .perception.understanding.intent_detector import IntentDetector
from .reasoning.cognitive.abduction import AbductionEngine
from .reasoning.cognitive.analogy import AnalogicalMapper
from .reasoning.cognitive.counterfactual import CounterfactualReasoner
from .reasoning.debate.council import MultiAgentDebate
from .reasoning.execution.controller import ReasoningController
from .reasoning.search.mcts import MCTSLight
from .reasoning.thinking.chain_verifier import ChainOfThoughtVerifier
from .reasoning.thinking.recursive_reflection import RecursiveReflectionLoop
from .reasoning.thinking.reasoning_path import ReasoningPathComputer
from .reasoning.thinking.self_consistency import SelfConsistency
from .reasoning.thinking.tree_of_thought import TreeOfThoughts
from .reflection.answer_rewriter import AnswerRewriter
from .reflection.confidence_calibrator import ConfidenceCalibrator
from .reflection.consistency import ConsistencyChecker
from .reflection.critic import Critic
from .reflection.hallucination import Assessor as HallucinationAssessor
from .reflection.self_reflection import SelfReflection
from .routing.evolving_router import SelfEvolvingRouter
from .routing.expert_router import ExpertRouter
from .routing.load_balancer import ExpertLoadBalancer
from .planning.goal_decomposer import GoalDecomposer
from .planning.plan_decomposer import PlanDecomposer
from .routing.budget_router import BudgetRouter
from .safety.active_learning import ActiveLearningLoop
from .safety.ethical_filter import EthicalFilter
from .state import ThinkingState
from .tools.context_builder import ToolContextBuilder
from .tools.executor import ToolExecutor
from .tools.registry import ToolRegistry
from .utils.thread import LockedBridge, NamedLocks


@dataclass
class EngineComponents:
    """Holds exactly one instance of every subsystem component. Built
    once in ``ThinkingEngine.__init__`` and threaded through every
    ``Pipeline`` stage — this is the engine's full dependency graph made
    explicit and inspectable, replacing engine_v1.py's implicit
    ``self.xxx`` attribute soup on the monolithic class.
    """

    bridge: Any
    tokenizer: Any
    llm_generate_fn: Optional[Callable]

    # Perception
    prompt_sanitizer: PromptSanitizer
    contextual_rewriter: ContextualPromptRewriter
    intent_detector: IntentDetector
    metaphor_detector: MetaphorDetector
    difficulty_scorer: DifficultyScorer
    dual_process_decider: DualProcessDecider
    prompt_compiler: PromptCompiler

    # Memory
    memory_retrieval: MemoryRetrieval
    context_window_manager: ContextWindowManager
    user_profile_manager: UserProfileManager

    # Knowledge
    kg: KnowledgeGraph
    temporal_kg: TemporalKnowledgeGraph
    rag_pipeline: RAGPipeline
    fact_checker: FactConsistencyChecker
    analogical_mapper: AnalogicalMapper
    bayesian_updater: BayesianBeliefUpdater

    # Routing
    expert_router: ExpertRouter
    budget_router: BudgetRouter

    # Reasoning
    reasoning_controller: ReasoningController
    reasoning_path_computer: ReasoningPathComputer
    chain_verifier: ChainOfThoughtVerifier
    recursive_reflection_loop: RecursiveReflectionLoop
    abduction_engine: AbductionEngine
    counterfactual_reasoner: CounterfactualReasoner

    # Planning
    goal_decomposer: GoalDecomposer
    plan_decomposer: PlanDecomposer

    # Tools
    tool_registry: ToolRegistry
    tool_context_builder: ToolContextBuilder
    tool_executor: ToolExecutor

    # Reflection (+ Hallucination)
    critic: Critic
    consistency_checker: ConsistencyChecker
    answer_rewriter: AnswerRewriter
    hallucination_assessor: HallucinationAssessor

    # Safety
    ethical_filter: EthicalFilter
    active_learning_loop: ActiveLearningLoop

    # Learning
    feedback_collector: FeedbackCollector
    experience_buffer: ExperienceBuffer
    learning_stats: LearningStats

    # Execution
    response_builder: ResponseBuilder
    response_formatter: ResponseFormatter

    # Concurrency
    kg_lock: threading.Lock


def _mcts_kwargs(mcts_config: Any) -> Dict[str, Any]:
    return {"n_sim": mcts_config.n_sim, "max_rollout_depth": mcts_config.max_rollout_depth}


class ThinkingEngine:
    """Top-level facade. One instance should be created per model/session
    and reused across requests — most components are stateless or hold
    only cheap accumulators (routing stats, calibration history), so
    construction cost is paid once, not per-request."""

    def __init__(
        self,
        numpy_model: Any = None,
        tokenizer: Any = None,
        config: Optional[EngineConfig] = None,
        llm_generate_fn: Optional[Callable] = None,
    ) -> None:
        self.config = config or EngineConfig.default()
        self.config.validate()

        raw_bridge = load_default_bridge(numpy_model=numpy_model)
        kg_lock = threading.Lock()
        bridge_lock = threading.Lock()

        # ── Bridge wiring, in order ──────────────────────────────────
        # 1. raw_bridge   : either the REAL modeling.transformer.TransformerBridge
        #                   (backend="numpy"/"llama.cpp", no is_connected(),
        #                   no expert_boost_to_array/build_intent_bias — see
        #                   interfaces/llm.py's module docstring for the full
        #                   real-vs-assumed API diff) or StubLLMBridge.
        # 2. adapter       : TransformerBridgeAdapter gives every caller in
        #                   reasoning/planning/memory a stable is_connected()
        #                   + expert_boost_to_array()/build_intent_bias(),
        #                   regardless of which raw_bridge is underneath.
        #                   WITHOUT this step, the ~9 call sites across
        #                   reasoning/debate/council.py, planning/*, and
        #                   memory/summarization.py that call
        #                   `bridge.is_connected()` would raise AttributeError
        #                   the moment a real TransformerBridge is plugged in
        #                   (it only ever "worked" against StubLLMBridge,
        #                   which happened to define is_connected() itself).
        # 3. locked_bridge : serializes concurrent generate() calls from
        #                   ReasoningController's ThreadPoolExecutor futures
        #                   so they never race on the same KVCache.
        #                   LockedBridge.__getattr__ transparently forwards
        #                   is_connected()/backend/expert_boost_to_array/
        #                   build_intent_bias/generate_from_engine_output
        #                   through to the adapter underneath.
        adapter = TransformerBridgeAdapter(raw_bridge, vocab_size=numpy_model.vocab_size) \
            if numpy_model is not None and hasattr(numpy_model, "vocab_size") \
            else TransformerBridgeAdapter(raw_bridge)
        locked_bridge = LockedBridge(adapter, bridge_lock) if raw_bridge is not None else None

        kg = KnowledgeGraph()
        kg.init_default()
        temporal_kg = TemporalKnowledgeGraph()

        intent_detector = IntentDetector()
        evolving_router = SelfEvolvingRouter()
        load_balancer = ExpertLoadBalancer()
        expert_router = ExpertRouter(intent_detector, evolving_router, load_balancer)

        tot = TreeOfThoughts(MCTSLight(**_mcts_kwargs(self.config.reasoning.tot)))
        debate = MultiAgentDebate()
        self_consistency = SelfConsistency()
        goal_decomposer = GoalDecomposer()
        plan_decomposer = PlanDecomposer(MCTSLight(**_mcts_kwargs(self.config.reasoning.plan_decomposer)))

        reasoning_controller = ReasoningController(
            tot=tot,
            debate=debate,
            self_consistency=self_consistency,
            goal_decomposer=goal_decomposer,
            plan_decomposer=plan_decomposer,
            kg=kg,
            temporal_kg=temporal_kg,
            kg_lock=kg_lock,
            max_workers=self.config.reasoning.max_workers,
        )

        tool_registry = ToolRegistry()

        self.components = EngineComponents(
            bridge=locked_bridge,
            tokenizer=tokenizer,
            llm_generate_fn=llm_generate_fn,
            prompt_sanitizer=PromptSanitizer(),
            contextual_rewriter=ContextualPromptRewriter(),
            intent_detector=intent_detector,
            metaphor_detector=MetaphorDetector(),
            difficulty_scorer=DifficultyScorer(),
            dual_process_decider=DualProcessDecider(),
            prompt_compiler=PromptCompiler(),
            memory_retrieval=MemoryRetrieval(),
            context_window_manager=ContextWindowManager(
                max_tokens=self.config.memory.context_token_budget
            ),
            user_profile_manager=UserProfileManager(),
            kg=kg,
            temporal_kg=temporal_kg,
            rag_pipeline=RAGPipeline(),
            fact_checker=FactConsistencyChecker(),
            analogical_mapper=AnalogicalMapper(),
            bayesian_updater=BayesianBeliefUpdater(),
            expert_router=expert_router,
            budget_router=BudgetRouter(),
            reasoning_controller=reasoning_controller,
            reasoning_path_computer=ReasoningPathComputer(),
            chain_verifier=ChainOfThoughtVerifier(),
            recursive_reflection_loop=RecursiveReflectionLoop(),
            abduction_engine=AbductionEngine(MCTSLight(**_mcts_kwargs(self.config.reasoning.abduction))),
            counterfactual_reasoner=CounterfactualReasoner(),
            goal_decomposer=goal_decomposer,
            plan_decomposer=plan_decomposer,
            tool_registry=tool_registry,
            tool_context_builder=ToolContextBuilder(tool_registry),
            tool_executor=ToolExecutor(tool_registry),
            critic=Critic(),
            consistency_checker=ConsistencyChecker(),
            answer_rewriter=AnswerRewriter(),
            hallucination_assessor=HallucinationAssessor(config=self.config.hallucination),
            ethical_filter=EthicalFilter(),
            active_learning_loop=ActiveLearningLoop(),
            feedback_collector=FeedbackCollector(),
            experience_buffer=ExperienceBuffer(),
            learning_stats=LearningStats(),
            response_builder=ResponseBuilder(),
            response_formatter=ResponseFormatter(),
            kg_lock=kg_lock,
        )

        from .pipeline import Pipeline

        self.pipeline = Pipeline(self.components, self.config)
        self._last_state: Optional[ThinkingState] = None

    def process(
        self,
        question: str,
        history: Optional[List[dict]] = None,
        memory_candidates: Optional[List[dict]] = None,
        memory_summary: str = "",
        ltm_facts: Optional[List[dict]] = None,
        user_id: Optional[str] = None,
        think_mode: bool = False,
        force_system2: bool = False,
        max_experts: Optional[int] = None,
    ) -> EngineOutput:
        """Phase 1: assemble the prompt (Perception through Answer
        Rewriter/Output). Returns an EngineOutput whose .messages field
        is ready to send to the LLM bridge. Mirrors engine_v1.py's
        ThinkingEngineV1.process().

        The internal ThinkingState produced here is cached on
        ``self._last_state`` so a subsequent ``finish()`` call — if made
        without an explicit ``prior_state`` — reuses the SAME reasoning
        context (best_branch, reasoning_path, subgoals, debate_opinions)
        instead of silently falling back to an empty state. Passing an
        explicit ``prior_state`` remains the correct choice for
        concurrent/multi-session use where relying on this single-slot
        cache would be unsafe.
        """
        state = ThinkingState(
            question=question,
            history=history or [],
            memory_candidates=memory_candidates,
            memory_summary=memory_summary,
            ltm_facts=ltm_facts or [],
            user_id=user_id,
            think_mode=think_mode,
            force_system2=force_system2,
            max_experts=max_experts,
        )
        state = self.pipeline.run(state)
        self._last_state = state
        return self._to_engine_output(state)

    def finish(
        self,
        question: str,
        generated_text: str,
        history: Optional[List[dict]] = None,
        prior_state: Optional[ThinkingState] = None,
        **process_kwargs: Any,
    ) -> EngineOutput:
        """Phase 2: given the LLM's raw generated text for the messages
        produced by process(), run Hallucination Assessment + Reflection
        + Answer Rewriter against the actual output. Mirrors
        engine_v1.py's post-generation recursive_critique /
        post_generation_check flow.

        Resolution order for which ThinkingState to run this phase
        against:
            1. ``prior_state`` if explicitly passed (safest for
               concurrent/multi-session use).
            2. ``self._last_state`` — the state produced by the most
               recent ``process()`` call on this engine instance.
            3. A fresh, empty ``ThinkingState`` — ONLY as a last resort
               when neither of the above is available. This path
               deliberately has no access to best_branch/reasoning_path/
               subgoals, so Hallucination Assessment's ConsistencyVerifier
               and ReasoningVerifier will correctly abstain (low
               confidence) rather than misreport risk against context
               that was never actually produced for this question.
        """
        state = prior_state if prior_state is not None else self._last_state
        if state is None:
            state = ThinkingState(question=question, history=history or [])
            state = self.pipeline.run(state)
        state = self.pipeline.run_post_generation(state, generated_text)
        return self._to_engine_output(state)

    def _to_engine_output(self, state: ThinkingState) -> EngineOutput:
        d = state.to_engine_output()
        formatted_text = ""
        if state.final_answer:
            formatted_text = self.components.response_formatter.format(
                {"text": state.final_answer},
                hallucination_report=state.hallucination_report,
                ethical_warning="",  # already folded into final_answer via post_generation_check
                calibrated_confidence=state.calibrated_confidence,
                rewrite_instruction="",
            )
        return EngineOutput(
            formatted_text=formatted_text,
            calibrated_confidence=d["calibrated_confidence"],
            intent=d["intent"],
            expert_boost=d["expert_boost"],
            miro_tau=d["miro_tau"],
            thought_plan=d["thought_plan"],
            messages=d["messages"],
            creativity=d["creativity"],
            rewritten_query=d["rewritten_query"],
            process_mode=d["process_mode"],
            difficulty_score=d["difficulty_score"],
            clarification_needed=d["clarification_needed"],
            clarification_question=d["clarification_question"],
            cot_verification=d["cot_verification"],
            tool_injection_active=d["tool_injection_active"],
            topic_shift=d["topic_shift"],
            ethical_warning=d["ethical_warning"],
            hallucination_report=d["hallucination_report"],
        )

    def submit_feedback(
        self,
        question: str,
        answer: str,
        rating: float,
        correction: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.components.feedback_collector.submit(question, answer, rating, correction, metadata)
        self.components.experience_buffer.add(
            {"question": question, "answer": answer, "rating": rating, "metadata": metadata or {}}
        )
