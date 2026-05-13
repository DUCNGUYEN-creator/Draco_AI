# DracoAI V1 — modeling/ops/activation.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Activation functions — pure NumPy, no model state."""
from __future__ import annotations
import math
import numpy as np

__all__ = ["silu", "gelu"]


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU/Swish: x * sigmoid(x). Gate clipped to prevent overflow."""
    return x / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU approximation (tanh form)."""
    return 0.5 * x * (1.0 + np.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))