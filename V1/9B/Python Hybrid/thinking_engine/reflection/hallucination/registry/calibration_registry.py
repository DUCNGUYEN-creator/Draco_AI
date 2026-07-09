# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
CalibrationRegistry
======================
Same plugin pattern, for calibration methods.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from ....exceptions import RegistryError
from ..calibration.beta import BetaCalibrator
from ..calibration.histogram import HistogramCalibrator
from ..calibration.isotonic import IsotonicCalibrator
from ..calibration.platt import PlattCalibrator
from ..calibration.temperature import TemperatureCalibrator

_BUILTINS: Dict[str, Callable] = {
    "platt": PlattCalibrator,
    "isotonic": IsotonicCalibrator,
    "beta": BetaCalibrator,
    "temperature": TemperatureCalibrator,
    "histogram": HistogramCalibrator,
}


class CalibrationRegistry:
    def __init__(self) -> None:
        self._constructors: Dict[str, Callable] = dict(_BUILTINS)

    def register(self, name: str, constructor: Callable, overwrite: bool = False) -> None:
        if name in self._constructors and not overwrite:
            raise RegistryError(f"Calibration method '{name}' already registered. Pass overwrite=True to replace it.")
        self._constructors[name] = constructor

    def create(self, name: str):
        if name not in self._constructors:
            raise RegistryError(f"Unknown calibration method: {name!r}. Registered: {list(self._constructors)}")
        return self._constructors[name]()

    def available(self) -> List[str]:
        return list(self._constructors)
