# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
MultiAgentDebate
==================
Lets code and language expert groups debate for complex questions — all
experts from a single Qwen 3.5 9B Instruct source. Ported 1:1 from
engine_v1.py's ``MultiAgentDebate`` (V2: RAM-efficient ``run_full_council``
with no debate_log, configurable max_experts, expert 0 always included).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from ...constants import EXPERT_CODE_0, EXPERT_CODE_1, EXPERT_CODE_2, EXPERT_LANG_0, EXPERT_LANG_1
from ...constants import INTENT_CODE, INTENT_LOGIC, INTENT_MATH
from .arbitration import Arbitrator
from .expert import EXPERT_NAMES, ROLE_TEMPLATES
from .voting import ConsensusChecker


class MultiAgentDebate:
    EXPERT_NAMES = EXPERT_NAMES
    _ROLE_TEMPLATES = ROLE_TEMPLATES

    def __init__(self) -> None:
        self._arbitrator = Arbitrator()
        self._consensus = ConsensusChecker()

    # ── Two-expert quick debate (non-council, fast slow-mode) ─────────
    def generate_debate(
        self,
        question: str,
        intent: dict,
        bridge: Any = None,
        tokenizer: Any = None,
    ) -> Tuple[str, Dict[int, str]]:
        itype = intent["intent"]
        opinions: Dict[int, str] = {}

        if itype in (INTENT_MATH, INTENT_LOGIC):
            debater_a, debater_b = EXPERT_CODE_0, EXPERT_CODE_2
        elif itype == INTENT_CODE:
            debater_a, debater_b = EXPERT_CODE_1, EXPERT_LANG_0
        else:
            debater_a, debater_b = EXPERT_LANG_0, EXPERT_LANG_1

        opinions[debater_a] = self._get_initial_thought(
            debater_a, question, intent, bridge=bridge, tokenizer=tokenizer
        )
        opinions[debater_b] = self._get_initial_thought(
            debater_b, question, intent, bridge=bridge, tokenizer=tokenizer
        )

        synthesis_stub = (
            f"[DEBATE SYNTHESIS] Combining perspectives from "
            f"{self.EXPERT_NAMES[debater_a]} and {self.EXPERT_NAMES[debater_b]}: "
            f"Balance technical correctness with clear communication."
        )
        if (
            bridge is not None
            and hasattr(bridge, "is_connected")
            and bridge.is_connected()
            and tokenizer is not None
        ):
            try:
                synth_prompt = (
                    f"Expert A ({self.EXPERT_NAMES[debater_a]}): {opinions[debater_a][:200]}\n"
                    f"Expert B ({self.EXPERT_NAMES[debater_b]}): {opinions[debater_b][:200]}\n\n"
                    f"Synthesize both perspectives into one concise answer for: {question}"
                )
                text = (
                    "<|im_start|>system\nYou are a debate moderator. Synthesize expert opinions.\n"
                    f"<|im_end|>\n<|im_start|>user\n{synth_prompt}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
                ids = tokenizer.encode(text, add_bos=True)
                out = bridge.generate(ids, max_new_tokens=200)
                if out:
                    decoded = tokenizer.decode(out).strip()
                    if decoded:
                        synthesis_stub = f"[DEBATE SYNTHESIS] {decoded}"
            except Exception:
                pass

        return synthesis_stub, opinions

    # ── Full 8-expert Council Debate ──────────────────────────────────
    def _get_initial_thought(
        self,
        exp_id: int,
        question: str,
        intent: dict,
        bridge: Any = None,
        tokenizer: Any = None,
    ) -> str:
        role_hint = self._ROLE_TEMPLATES.get(exp_id, "Provide a balanced response.")
        itype = intent.get("intent", "chat")
        q_short = question[:60]

        if (
            bridge is not None
            and hasattr(bridge, "is_connected")
            and bridge.is_connected()
            and tokenizer is not None
        ):
            try:
                system = role_hint
                prompt = f"Question: {question}\n\nYour expert response:"
                text = (
                    f"<|im_start|>system\n{system}<|im_end|>\n"
                    f"<|im_start|>user\n{prompt}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
                ids = tokenizer.encode(text, add_bos=True)
                out = bridge.generate(ids, max_new_tokens=128)
                if out:
                    decoded = tokenizer.decode(out).strip()
                    if decoded:
                        return f"[{self.EXPERT_NAMES[exp_id]}] {decoded}"
            except Exception:
                pass  # fall through to stub

        return f"[{self.EXPERT_NAMES[exp_id]}] For '{q_short}' (intent={itype}): {role_hint}"

    def _format_others(self, others: Dict[int, str]) -> str:
        lines = []
        for eid, thought in others.items():
            lines.append(f"  • {self.EXPERT_NAMES.get(eid, f'Expert{eid}')}: {thought[:120]}")
        return "\n".join(lines)

    def _expert_review(
        self,
        exp_id: int,
        question: str,
        intent: dict,
        my_old_thought: str,
        others_thoughts: Dict[int, str],
        bridge: Any = None,
        tokenizer: Any = None,
    ) -> str:
        if (
            bridge is not None
            and hasattr(bridge, "is_connected")
            and bridge.is_connected()
            and tokenizer is not None
        ):
            try:
                others_text = "\n".join(
                    f"  • {self.EXPERT_NAMES.get(eid, f'Expert{eid}')}: {t[:150]}"
                    for eid, t in others_thoughts.items()
                )
                system = self._ROLE_TEMPLATES.get(exp_id, "")
                prompt = (
                    f"Your previous stance: {my_old_thought[:200]}\n\n"
                    f"Other experts' opinions:\n{others_text}\n\n"
                    f"Question: {question}\n\n"
                    f"Updated thought (incorporate useful insights, reject weak ones):"
                )
                text = (
                    f"<|im_start|>system\n{system}<|im_end|>\n"
                    f"<|im_start|>user\n{prompt}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
                ids = tokenizer.encode(text, add_bos=True)
                out = bridge.generate(ids, max_new_tokens=128)
                if out:
                    decoded = tokenizer.decode(out).strip()
                    if decoded:
                        return f"[{self.EXPERT_NAMES[exp_id]}][R2] {decoded}"
            except Exception:
                pass  # fall through to stub

        my_keywords = set(my_old_thought.lower().split())
        agreements = 0
        for peer_thought in others_thoughts.values():
            peer_kw = set(peer_thought.lower().split())
            if len(my_keywords & peer_kw) > 5:
                agreements += 1

        if agreements >= 3:
            return (
                f"[{self.EXPERT_NAMES[exp_id]}][R2] I maintain my approach. "
                f"{agreements} peers share similar reasoning. "
                f"Key point: {self._ROLE_TEMPLATES.get(exp_id, '')}"
            )
        top_peer_id = max(others_thoughts, key=lambda k: len(others_thoughts[k]))
        top_peer_name = self.EXPERT_NAMES.get(top_peer_id, f"Expert{top_peer_id}")
        return (
            f"[{self.EXPERT_NAMES[exp_id]}][R2] Reconsidering after reading peers. "
            f"Incorporating insight from {top_peer_name}. "
            f"Updated stance: {self._ROLE_TEMPLATES.get(exp_id, '')} "
            f"+ cross-validate with {top_peer_name}'s perspective."
        )

    def _check_consensus(self, thoughts, threshold: float = 0.75) -> bool:
        return self._consensus.check(thoughts, threshold)

    def _arbitrate(
        self,
        final_thoughts: Dict[int, str],
        rounds_done: int,
        question: str,
        intent: dict,
    ) -> str:
        return self._arbitrator.arbitrate(final_thoughts, rounds_done, question, intent)

    def run_full_council(
        self,
        question: str,
        intent: dict,
        max_rounds: int = 3,
        max_experts: Optional[int] = None,
        bridge: Any = None,
        tokenizer: Any = None,
    ) -> Dict[str, Any]:
        """Full expert council debate — RAM-efficient (V2).

        max_experts: max number of experts to use (inclusive of expert 0).
        Defaults to 4. Expert 0 (arbiter) is ALWAYS included. Clamped to [1, 8].
        """
        if max_experts is None:
            max_experts = 4
        max_experts = max(1, min(max_experts, 8))

        expert_ids = [0] + [e for e in range(1, 8) if e < max_experts]
        n_experts = len(expert_ids)

        thoughts: Dict[int, str] = {
            eid: self._get_initial_thought(eid, question, intent, bridge=bridge, tokenizer=tokenizer)
            for eid in expert_ids
        }

        rounds_done = 0
        for round_idx in range(1, max_rounds + 1):
            old = thoughts.copy()  # shallow copy — only references, cheap
            for exp_id in expert_ids:
                others = {k: v for k, v in old.items() if k != exp_id}
                thoughts[exp_id] = self._expert_review(
                    exp_id,
                    question,
                    intent,
                    my_old_thought=old[exp_id],
                    others_thoughts=others,
                    bridge=bridge,
                    tokenizer=tokenizer,
                )
            rounds_done = round_idx
            if self._check_consensus(thoughts.values()):
                break

        final_answer = self._arbitrate(thoughts, rounds_done, question, intent)
        return {
            "final_answer": final_answer,
            "opinions": thoughts,
            "rounds_done": rounds_done,
            "n_experts_used": n_experts,
        }
