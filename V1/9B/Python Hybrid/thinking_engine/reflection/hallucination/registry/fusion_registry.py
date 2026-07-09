# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
FusionRegistry
=================
Same plugin pattern as VerifierRegistry, for fusion strategies. The
existing fusion.selector.FusionSelector remains the simple, direct
lookup path for config-driven selection; this registry exists for
cases where a NEW fusion method needs to be registered dynamically
(e.g. loaded from a user plugin) without modifying fusion/selector.py.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from ....exceptions import RegistryError
from ..fusion.bayesian import BayesianFusion
from ..fusion.dempster_shafer import DempsterShaferFusion
from ..fusion.logistic import LogisticFusion
from ..fusion.noisy_or import NoisyOrFusion
from ..fusion.weighted import WeightedAverageFusion

_BUILTINS: Dict[str, Callable] = {
    "weighted": WeightedAverageFusion,
    "noisy_or": NoisyOrFusion,
    "bayesian": BayesianFusion,
    "dempster_shafer": DempsterShaferFusion,
    "logistic": LogisticFusion,
}


class FusionRegistry:
    def __init__(self) -> None:
        self._constructors: Dict[str, Callable] = dict(_BUILTINS)

    def register(self, name: str, constructor: Callable, overwrite: bool = False) -> None:
        if name in self._constructors and not overwrite:
            raise RegistryError(f"Fusion method '{name}' already registered. Pass overwrite=True to replace it.")
        self._constructors[name] = constructor

    def create(self, name: str):
        if name not in self._constructors:
            raise RegistryError(f"Unknown fusion method: {name!r}. Registered: {list(self._constructors)}")
        return self._constructors[name]()

    def available(self) -> List[str]:
        return list(self._constructors)
