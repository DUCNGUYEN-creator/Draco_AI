# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ActiveLearningLoop
=====================
When confidence is low, generates a clarifying question to ask the
user before committing to an answer. Ported 1:1 from engine_v1.py's
``ActiveLearningLoop``.
"""

from __future__ import annotations

from ..constants import (
    INTENT_CHAT,
    INTENT_CODE,
    INTENT_COMPARISON,
    INTENT_FACTUAL,
    INTENT_HOW_TO,
    INTENT_MATH,
)

_CLARIFICATION_TEMPLATES = {
    INTENT_MATH: "Bạn muốn tính {topic} theo phương pháp nào — chính xác hay xấp xỉ?",
    INTENT_CODE: "Ngôn ngữ lập trình và phiên bản bạn đang dùng cho {topic} là gì?",
    INTENT_FACTUAL: "Bạn hỏi về {topic} trong ngữ cảnh nào — kỹ thuật hay tổng quát?",
    INTENT_HOW_TO: "Mức độ chi tiết bạn cần cho '{topic}' là: cơ bản hay nâng cao?",
    INTENT_COMPARISON: "Bạn muốn so sánh {topic} theo tiêu chí nào — hiệu suất, chi phí, hay dễ dùng?",
}
_DEFAULT_TEMPLATE = "Bạn có thể nói rõ hơn về '{topic}' không? Tôi muốn trả lời chính xác hơn."


class ActiveLearningLoop:
    def needs_clarification(self, confidence: float, intent: dict) -> bool:
        return confidence < 0.5

    def generate_clarification(self, question: str, intent: dict) -> str:
        itype = intent.get("intent", INTENT_CHAT)
        entities = intent.get("entities", [])
        topic = entities[0] if entities else question[:30]
        template = _CLARIFICATION_TEMPLATES.get(itype, _DEFAULT_TEMPLATE)
        return template.format(topic=topic)
