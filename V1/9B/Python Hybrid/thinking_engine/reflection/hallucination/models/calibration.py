# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""CalibrationPoint / CalibrationModel — the data + fitted-parameter
shapes shared by every calibration/*.py implementation (Platt, Isotonic,
Beta, Temperature, Histogram)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class CalibrationPoint:
    raw_score: float
    label: int  # 1 = was actually correct/supported, 0 = was hallucination


@dataclass
class CalibrationModel:
    method: str
    params: Dict[str, Any] = field(default_factory=dict)
    n_samples: int = 0
    fitted: bool = False
    history: List[Tuple[float, int]] = field(default_factory=list)
