# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ReasoningController
======================
Orchestrates the concurrent "heavy task" block that used to be inlined
inside engine_v1.py's ``ThinkingEngineV1.process()``:

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_kg      = executor.submit(self._safe_extract_triples, ...)
        future_tot     = executor.submit(self._run_tot_with_llm, ...)
        future_debate  = executor.submit(self._run_council_with_llm, ...)  # or debate
        future_goal    = executor.submit(self.goal_decomposer.decompose, ...)
        future_subgoals= executor.submit(self.decomposer.decompose, ...)
        ... light sequential tasks run on the main thread while waiting ...
        future_kg.result()   # MUST complete before any KG read
        ...

Ported with identical ordering and the identical KG-write-before-read
race fix (RACE-FIX: future_kg.result() is awaited before any KG read,
e.g. ReasoningPathComputer.compute()).
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from ...constants import INTENT_CODE, INTENT_LOGIC, INTENT_MATH
from ..debate.council import MultiAgentDebate
from ..thinking.self_consistency import SelfConsistency
from ..thinking.tree_of_thought import TreeOfThoughts


class ReasoningController:
    """Threads the heavy reasoning tasks through a shared ThreadPoolExecutor,
    using the same locking discipline as engine_v1.py's _kg_lock / _bridge_lock
    so KVCache and KG writes stay race-free across concurrent futures.
    """

    def __init__(
        self,
        tot: TreeOfThoughts,
        debate: MultiAgentDebate,
        self_consistency: SelfConsistency,
        goal_decomposer: Any,
        plan_decomposer: Any,
        kg: Any,
        temporal_kg: Any,
        kg_lock: Optional[threading.Lock] = None,
        max_workers: int = 4,
    ) -> None:
        self.tot = tot
        self.debate = debate
        self.sc = self_consistency
        self.goal_decomposer = goal_decomposer
        self.plan_decomposer = plan_decomposer
        self.kg = kg
        self.temporal_kg = temporal_kg
        self._kg_lock = kg_lock or threading.Lock()
        self.max_workers = max_workers

    # ── Thread-safe KG write helper (ported from _safe_extract_triples) ──
    def safe_extract_triples(self, text: str, conf: float) -> None:
        with self._kg_lock:
            self.kg.extract_and_add_triples(text, conf)
            self.temporal_kg.extract_and_add_triples(text, conf)

    def run(
        self,
        question: str,
        intent: Dict[str, Any],
        think_mode: bool,
        process_mode: str,
        base_conf: float,
        max_experts: int,
        bridge: Any,
        tokenizer: Any,
        llm_generate_fn=None,
        goal_keywords: Optional[List[str]] = None,
        max_council_rounds: int = 3,
        self_consistency_paths: int = 3,
    ) -> Dict[str, Any]:
        """Returns a dict with all the futures' resolved results, using the
        exact same field names the monolithic engine produced so
        downstream PromptCompiler / ThinkingState mapping stays unchanged.
        """
        goal_keywords = goal_keywords or [
            "kế hoạch", "plan", "lộ trình", "roadmap", "30 ngày", "schedule",
        ]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_kg = executor.submit(self.safe_extract_triples, question, base_conf)

            future_tot = executor.submit(
                self.tot.run_with_llm, question, intent, bridge, tokenizer, llm_generate_fn
            )

            future_debate = None
            if think_mode:
                future_debate = executor.submit(
                    self.debate.run_full_council,
                    question, intent, max_council_rounds, max_experts,
                    bridge, tokenizer,
                )
            elif process_mode == "slow":
                future_debate = executor.submit(
                    self.debate.generate_debate, question, intent, bridge, tokenizer
                )

            future_goal = None
            if any(kw in question.lower() for kw in goal_keywords):
                future_goal = executor.submit(
                    self.goal_decomposer.decompose, question, intent, 6, bridge, tokenizer
                )

            future_subgoals = None
            if process_mode == "slow" or think_mode:
                future_subgoals = executor.submit(
                    self.plan_decomposer.decompose, question, intent, 4, bridge, tokenizer
                )

            # ── KG write MUST complete before any KG read ──────────────
            future_kg.result()

            best_branch, all_branches = future_tot.result()

            debate_synthesis = ""
            debate_opinions: Dict[int, str] = {}
            used_experts: List[int] = []
            if future_debate is not None:
                if think_mode:
                    council_result = future_debate.result()
                    debate_synthesis = council_result["final_answer"]
                    debate_opinions = council_result["opinions"]
                    used_experts = list(debate_opinions.keys())
                else:
                    debate_synthesis, debate_opinions = future_debate.result()
                    used_experts = list(debate_opinions.keys())

            goal_decomposition: List[str] = future_goal.result() if future_goal is not None else []
            subgoals: List[str] = future_subgoals.result() if future_subgoals is not None else []

            sc_path = ""
            if think_mode and intent.get("intent") in (INTENT_MATH, INTENT_LOGIC, INTENT_CODE):
                sc_paths = self.sc.generate_paths(question, intent, n_paths=self_consistency_paths)
                sc_path = self.sc.vote(sc_paths, question)

        return {
            "best_branch": best_branch,
            "all_branches": all_branches,
            "debate_synthesis": debate_synthesis,
            "debate_opinions": debate_opinions,
            "used_experts": used_experts,
            "goal_decomposition": goal_decomposition,
            "subgoals": subgoals,
            "sc_path": sc_path,
        }
