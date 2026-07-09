# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Critic
========
Orchestrates the recursive self-critique loop AND the full
post-generation check pipeline (fact consistency, temporal consistency,
ethical filter, uncertainty tagging). Ported 1:1 from engine_v1.py's
``ThinkingEngineV1.recursive_critique`` / ``post_generation_check``.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, TYPE_CHECKING

from .self_reflection import SelfReflection
from .confidence_calibrator import ConfidenceCalibrator

if TYPE_CHECKING:  # pragma: no cover
    from ..knowledge.fact_checker import FactConsistencyChecker
    from ..knowledge.temporal_graph import TemporalKnowledgeGraph
    from ..knowledge.knowledge_graph import KnowledgeGraph
    from ..safety.ethical_filter import EthicalFilter
    from ..reasoning.thinking.uncertainty import UncertaintyQuantifier


class Critic:
    def __init__(
        self,
        reflect: Optional[SelfReflection] = None,
        calibrator: Optional[ConfidenceCalibrator] = None,
        fact_checker: Optional["FactConsistencyChecker"] = None,
        ethical_filter: Optional["EthicalFilter"] = None,
        uncertainty_quantifier: Optional["UncertaintyQuantifier"] = None,
    ) -> None:
        from ..knowledge.fact_checker import FactConsistencyChecker
        from ..safety.ethical_filter import EthicalFilter
        from ..reasoning.thinking.uncertainty import UncertaintyQuantifier

        self.reflect = reflect or SelfReflection()
        self.calibrator = calibrator or ConfidenceCalibrator()
        self.fact_checker = fact_checker or FactConsistencyChecker()
        self.ethical_filter = ethical_filter or EthicalFilter()
        self.uq = uncertainty_quantifier or UncertaintyQuantifier()

    def recursive_critique(
        self,
        question: str,
        answer: str,
        ltm_facts: Optional[List[dict]] = None,
        max_iter: int = 3,
    ) -> Tuple[str, List[dict]]:
        """Iteratively critiques and refines the answer up to max_iter
        times. Returns (final_answer, list_of_critique_reports)."""
        facts = ltm_facts or []
        reports: List[dict] = []
        current = answer
        for _ in range(max_iter):
            report = self.reflect.critique(current, question, facts)
            reports.append(report)
            if not report["should_refine"]:
                break
            refine_note = self.reflect.build_refine_prompt(current, report)
            # PRODUCTION HOOK: pass refine_note to LLM for actual refinement.
            current = f"{current}\n[AUTO-REFINED: {refine_note[:120]}]"
        return current, reports

    def post_generation_check(
        self,
        answer: str,
        question: str,
        kg: Any,
        temporal_kg: Any,
        base_confidence: float,
        ltm_facts: Optional[List[dict]] = None,
    ) -> dict:
        """Full post-generation pipeline:
        1. Fact consistency check vs KG
        2. Temporal consistency check
        3. Ethical filter
        4. Uncertainty tagging
        """
        fact_issues = self.fact_checker.check(answer, kg)
        temp_issues = temporal_kg.check_temporal_consistency(answer)
        all_issues = fact_issues + temp_issues

        is_ethical = self.ethical_filter.is_safe(answer)
        eth_note = "" if is_ethical else self.ethical_filter.build_rewrite_instruction()

        tagged = self.uq.tag(answer, base_confidence=base_confidence)
        calibrated = self.calibrator.calibrate(base_confidence)

        return {
            "answer": answer,
            "tagged_answer": tagged,
            "fact_issues": all_issues,
            "is_ethical": is_ethical,
            "ethical_note": eth_note,
            "confidence": base_confidence,
            "calibrated_conf": calibrated,
        }
