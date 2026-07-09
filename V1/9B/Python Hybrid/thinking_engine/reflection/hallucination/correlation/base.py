# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""BaseCorrelator — shared contract for evidence-level and verifier-level
correlation detectors."""

from __future__ import annotations

import abc
from typing import Any, List

from ..models.correlation import CorrelationGroup


class BaseCorrelator(abc.ABC):
    @abc.abstractmethod
    def correlate(self, items: List[Any]) -> List[CorrelationGroup]:
        """Group ``items`` (evidence texts, or verifier-result dicts) into
        CorrelationGroups. Items not correlated with anything else
        appear as their own singleton group."""
