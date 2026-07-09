# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Evidence / EvidenceBundle — the unit of "support" a verifier inspects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .enums import EvidenceType


@dataclass
class Evidence:
    text: str
    source_type: EvidenceType = EvidenceType.NONE
    source_id: Optional[str] = None
    trust_score: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceBundle:
    """All evidence gathered for ONE claim — multiple Evidence items from
    possibly different sources (KG, RAG, memory, tool results)."""

    claim: str
    items: List[Evidence] = field(default_factory=list)

    def add(self, evidence: Evidence) -> None:
        self.items.append(evidence)

    def is_empty(self) -> bool:
        return len(self.items) == 0

    def best_trust(self) -> float:
        return max((e.trust_score for e in self.items), default=0.0)

    def texts(self) -> List[str]:
        return [e.text for e in self.items]
