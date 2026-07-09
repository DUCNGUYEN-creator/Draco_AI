# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
FusionFactory
================
Builds (and caches) fusion-strategy instances, resolving the
"ensemble" pseudo-name to fusion.ensemble.FusionEnsemble.
"""

from __future__ import annotations

from typing import Dict

from ..fusion.base import BaseFusionStrategy
from ..fusion.ensemble import FusionEnsemble
from ..registry.fusion_registry import FusionRegistry


class FusionFactory:
    def __init__(self, registry: FusionRegistry | None = None) -> None:
        self.registry = registry or FusionRegistry()
        self._instance_cache: Dict[str, BaseFusionStrategy] = {}

    def get(self, name: str) -> BaseFusionStrategy:
        if name == "ensemble":
            return self._instance_cache.setdefault("ensemble", FusionEnsemble())
        if name not in self._instance_cache:
            self._instance_cache[name] = self.registry.create(name)
        return self._instance_cache[name]
