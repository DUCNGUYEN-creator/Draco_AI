# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
VerifierRegistry
===================
Maps a verifier NAME to a zero-arg constructor. Pre-populated with this
engine's 9 built-in verifiers; ``register()`` lets new ones be added
(e.g. a future embedding-based RetrievalVerifierV2) without touching
factory.py or assessor.py — they just need to be registered under a
new name and referenced from strategy/*.py's verifier-name lists.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from ....exceptions import RegistryError
from ..verifiers.citation import CitationVerifier
from ..verifiers.consistency import ConsistencyVerifier
from ..verifiers.contradiction import ContradictionVerifier
from ..verifiers.numerical import NumericalVerifier
from ..verifiers.planner import PlannerVerifier
from ..verifiers.reasoning import ReasoningVerifier
from ..verifiers.retrieval import RetrievalVerifier
from ..verifiers.symbolic import SymbolicVerifier
from ..verifiers.tool import ToolVerifier

_BUILTINS: Dict[str, Callable] = {
    "retrieval": RetrievalVerifier,
    "reasoning": ReasoningVerifier,
    "consistency": ConsistencyVerifier,
    "contradiction": ContradictionVerifier,
    "numerical": NumericalVerifier,
    "symbolic": SymbolicVerifier,
    "citation": CitationVerifier,
    "planner": PlannerVerifier,
    "tool": ToolVerifier,
}


class VerifierRegistry:
    def __init__(self) -> None:
        self._constructors: Dict[str, Callable] = dict(_BUILTINS)

    def register(self, name: str, constructor: Callable, overwrite: bool = False) -> None:
        if name in self._constructors and not overwrite:
            raise RegistryError(f"Verifier '{name}' already registered. Pass overwrite=True to replace it.")
        self._constructors[name] = constructor

    def create(self, name: str):
        if name not in self._constructors:
            raise RegistryError(f"Unknown verifier: {name!r}. Registered: {list(self._constructors)}")
        return self._constructors[name]()

    def create_many(self, names: List[str]) -> List:
        return [self.create(name) for name in names]

    def available(self) -> List[str]:
        return list(self._constructors)
