# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
TreeOfThoughts
================
Generates candidate reasoning "branches" per intent type and picks the
best one via MCTSLight. Ported 1:1 from engine_v1.py's
``TreeOfThoughts``. The real-LLM variant (``_run_tot_with_llm`` in the
monolith) now lives in ``run_with_llm`` below so this stays usable in
both the stub and LLM-connected configurations.
"""

from __future__ import annotations

from typing import Any, List, Tuple

from ...constants import INTENT_CODE, INTENT_CREATIVE, INTENT_LOGIC, INTENT_MATH
from ..search.mcts import MCTSLight


class TreeOfThoughts:
    def __init__(self, mcts: MCTSLight) -> None:
        self.mcts = mcts

    def generate_branches(self, question: str, intent: dict) -> List[str]:
        itype = intent["intent"]
        if itype in (INTENT_MATH, INTENT_LOGIC):
            return [
                f"Xác định yếu tố cần tính/chứng minh trong '{question[:40]}'",
                "Áp dụng công thức/quy tắc phù hợp trực tiếp",
                "Phân rã thành từng bước nhỏ, giải từng phần",
            ]
        if itype == INTENT_CODE:
            return [
                "Thiết kế interface/API trước, implement sau",
                "Viết hàm nhỏ trước, ghép lại theo bottom-up",
                "Test-driven: xác định test cases trước rồi implement",
            ]
        if itype == INTENT_CREATIVE:
            return [
                "Tập trung vào nhân vật và cảm xúc",
                "Xây dựng plot rõ ràng với conflict và resolution",
                "Góc nhìn mới lạ, bất ngờ, độc đáo",
            ]
        return [
            "Trả lời ngắn gọn và trực tiếp",
            "Giải thích với ví dụ cụ thể",
            "Nhìn từ nhiều góc độ khác nhau",
        ]

    def run(self, question: str, intent: dict) -> Tuple[str, List[str]]:
        """Stub/MCTS-only path — no LLM involved."""
        branches = self.generate_branches(question, intent)
        best = self.mcts.search(question, branches)
        return best, branches

    def run_with_llm(
        self,
        question: str,
        intent: dict,
        bridge: Any = None,
        tokenizer: Any = None,
        llm_generate_fn=None,
    ) -> Tuple[str, List[str]]:
        """Real-LLM variant ported from engine_v1.py's
        ``ThinkingEngineV1._run_tot_with_llm``. ``llm_generate_fn`` is an
        optional ``(prompt, max_new_tokens, system) -> str`` callable
        (e.g. the engine's ``_llm_generate`` helper); when not supplied,
        falls back to the MCTS stub.
        """
        branches = self.generate_branches(question, intent)
        connected = bridge is not None and hasattr(bridge, "is_connected") and bridge.is_connected()
        if not connected or tokenizer is None or llm_generate_fn is None:
            return self.run(question, intent)

        best_thought = ""
        best_score = -1.0
        all_thoughts: List[str] = []
        q_words = set(question.lower().split())

        for branch in branches:
            system = "You are a reasoning expert. Follow the given thinking approach precisely."
            prompt = f"Thinking approach: {branch}\n\nQuestion: {question}\n\nThought:"
            thought = llm_generate_fn(prompt, max_new_tokens=128, system=system)
            if not thought:
                thought = branch
            all_thoughts.append(thought)
            t_words = set(thought.lower().split())
            score = len(q_words & t_words) * 0.1 + min(len(thought) / 400.0, 0.5)
            if score > best_score:
                best_score = score
                best_thought = thought

        return best_thought or branches[0], all_thoughts
