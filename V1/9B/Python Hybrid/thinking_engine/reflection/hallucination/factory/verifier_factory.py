# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
VerifierFactory
==================
Builds (and caches) verifier instances by name, and assembles a
VerifierEnsemble from a list of names — the single call strategy/*.py
makes to go from "I want these 6 verifiers" to a ready-to-run ensemble.
"""

from __future__ import annotations

from typing import Dict, List

from ..registry.verifier_registry import VerifierRegistry
from ..verifiers.ensemble import VerifierEnsemble


class VerifierFactory:
    def __init__(self, registry: VerifierRegistry | None = None) -> None:
        self.registry = registry or VerifierRegistry()
        self._instance_cache: Dict[str, object] = {}

    def get(self, name: str):
        if name not in self._instance_cache:
            self._instance_cache[name] = self.registry.create(name)
        return self._instance_cache[name]

    def build_ensemble(self, names: List[str]) -> VerifierEnsemble:
        return VerifierEnsemble([self.get(name) for name in names])
