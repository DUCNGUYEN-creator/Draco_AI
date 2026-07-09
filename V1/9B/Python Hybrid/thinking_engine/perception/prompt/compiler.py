# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
PromptCompiler
================
Compiles the final ChatML message list using the
[PLAN][THOUGHT][FINAL ANSWER] scaffold. This is the very last step of
the Answer-Rewriter / Output stages before the messages are handed to
the LLM bridge. Ported 1:1 from engine_v1.py's ``PromptCompiler``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...constants import (
    DRACO_SYSTEM_PROMPT,
    INTENT_CHAT,
    INTENT_CODE,
    INTENT_CREATIVE,
    INTENT_LOGIC,
    INTENT_MATH,
)

_EXPERT_NOTES: Dict[str, str] = {
    INTENT_MATH: "\n[ACTIVE] Logic/Math experts (0,2). Show step-by-step working.",
    INTENT_LOGIC: "\n[ACTIVE] Logic/Math + Debug experts (0,2). Use formal reasoning.",
    INTENT_CODE: "\n[ACTIVE] Code + Language experts (1,4). Write code with explanation.",
    INTENT_CREATIVE: "\n[ACTIVE] Creative + Language experts (6,4). Be expressive.",
    INTENT_CHAT: "\n[ACTIVE] Chat expert (5). Be friendly and natural.",
}


class PromptCompiler:
    def compile(
        self,
        question: str,
        intent: Dict[str, Any],
        thought_plan: Dict[str, Any],
        memory_summary: str = "",
        history: Optional[List[dict]] = None,
    ) -> List[dict]:
        msgs: List[dict] = []
        sys_content = DRACO_SYSTEM_PROMPT
        if memory_summary:
            sys_content += f"\n\n[MEMORY]\n{memory_summary[:400]}"

        expert_note = _EXPERT_NOTES.get(intent.get("intent", INTENT_CHAT), "")
        if expert_note:
            sys_content += expert_note

        if intent.get("sentiment") == "negative":
            sys_content += (
                "\n[EMOTION] User may be upset or frustrated. "
                "Be extra empathetic, concise, and supportive. "
                "Avoid jargon. Prioritize emotional acknowledgment."
            )

        if thought_plan.get("tool_injection"):
            sys_content += f"\n\n{thought_plan['tool_injection']}"
        if thought_plan.get("counterfactual"):
            sys_content += f"\n\n{thought_plan['counterfactual']}"
        if thought_plan.get("metaphor_note"):
            sys_content += f"\n\n{thought_plan['metaphor_note']}"
        if thought_plan.get("spatial_note"):
            sys_content += f"\n\n{thought_plan['spatial_note']}"
        if thought_plan.get("ethical_warning"):
            sys_content += f"\n\n{thought_plan['ethical_warning']}"

        msgs.append({"role": "system", "content": sys_content})

        if thought_plan.get("thoughts"):
            lines = [
                "[PLAN]",
                (
                    f"Type: {intent.get('intent', '?')} | Lang: {intent.get('lang', '?')} | "
                    f"Entities: {', '.join(intent.get('entities', [])[:3]) or '---'}"
                ),
            ]
            if thought_plan.get("subgoals"):
                lines.append("\n[SUBGOALS]")
                lines.extend(thought_plan["subgoals"])

            if thought_plan.get("goal_decomposition"):
                lines.append("\n[GOAL DECOMPOSITION]")
                lines.extend(thought_plan["goal_decomposition"])

            if thought_plan.get("instruction_chain"):
                lines.append("\n[INSTRUCTION CHAIN]")
                lines.extend(thought_plan["instruction_chain"])

            if thought_plan.get("debate_synthesis"):
                lines.append(f"\n[DEBATE]\n{thought_plan['debate_synthesis']}")
            if thought_plan.get("sc_path"):
                lines.append(f"\n[SELF-CONSISTENCY]\n{thought_plan['sc_path']}")
            if thought_plan.get("best_branch"):
                lines.append(f"\n[THOUGHT 1]\n{thought_plan['best_branch']}")
            for i, t in enumerate(thought_plan.get("thoughts", [])[:2], 2):
                lines.append(f"\n[THOUGHT {i}]\n{t}")
            if thought_plan.get("reasoning_path"):
                lines.append(
                    f"\n[KNOWLEDGE PATH]\n{' → '.join(thought_plan['reasoning_path'])}"
                )

            if thought_plan.get("analogy"):
                lines.append(f"\n{thought_plan['analogy']}")

            if thought_plan.get("fact_issues"):
                lines.append(
                    f"\n[FACT CHECK] Issues: {'; '.join(thought_plan['fact_issues'][:3])}"
                )

            if thought_plan.get("hypothesis"):
                h = thought_plan["hypothesis"]
                lines.append(
                    f"\n[HYPOTHESIS] {h.get('hypothesis', '')} → "
                    f"verdict={h.get('verdict', '?')} "
                    f"(support={h.get('support_strength', 0):.2f})"
                )

            if thought_plan.get("cot_verification"):
                v = thought_plan["cot_verification"]
                if not v.get("is_sound"):
                    lines.append(
                        f"\n[COT VERIFY — issues: {'; '.join(v.get('issues', [])[:2])}]"
                    )

            # Hallucination risk surfaced directly in the prompt scaffold —
            # lets the model see its own previous-turn risk report when the
            # caller threads it through thought_plan (e.g. in a refine loop).
            if thought_plan.get("hallucination_report"):
                hr = thought_plan["hallucination_report"]
                lines.append(
                    f"\n[HALLUCINATION RISK] level={hr.get('risk_level', '?')} "
                    f"score={hr.get('risk_score', 0):.2f}"
                )

            if thought_plan.get("abduction"):
                lines.append("\n[ABDUCTIVE HYPOTHESES]")
                lines.extend(thought_plan["abduction"][:3])

            lines.append("\n[FINAL ANSWER]")
            msgs.append({"role": "system", "content": "\n".join(lines)})

        if history:
            msgs.extend(history[-10:])
        msgs.append({"role": "user", "content": question})
        return msgs
