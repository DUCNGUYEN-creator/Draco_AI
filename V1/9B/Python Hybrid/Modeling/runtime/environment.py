# DracoAI V1 — modeling/runtime/environment.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Hybrid mode orchestrator — thinking (CPU) ↔ inference (GPU) dispatch.

ExecutionEnvironment is a lightweight context object created once per
generate() session. It holds the resolved backend name and provides
execute() hooks for dispatching compute functions.

Rules:
  - "Thinking" (sampling, penalty, routing) always runs on CPU.
  - "Inference" (forward pass) runs on the optimal_backend.
  - This module has NO knowledge of model weights, layers, or cache.
"""
from __future__ import annotations
import logging
from typing import Callable, Any

from ..constants import KERNEL_NUMPY, KERNEL_TRITON, KERNEL_NUMBA

__all__ = ["ExecutionEnvironment", "get_environment"]

logger = logging.getLogger(__name__)


class ExecutionEnvironment:
    """
    Execution context for a generation session.

    Attributes
    ----------
    inference_backend : KERNEL_NUMPY | KERNEL_TRITON | KERNEL_NUMBA
    thinking_backend  : always KERNEL_NUMPY (CPU)
    """

    def __init__(self, inference_backend: str = KERNEL_NUMPY):
        self.inference_backend = inference_backend
        self.thinking_backend  = KERNEL_NUMPY
        logger.debug("[Env] inference=%s thinking=%s",
                     inference_backend, self.thinking_backend)

    def execute_inference(self, fn: Callable, *args, **kwargs) -> Any:
        """
        Run fn on the inference backend.
        Currently a pass-through — kernel selection is done inside layers/
        via the kernel registry. This hook exists for future device migration.
        """
        return fn(*args, **kwargs)

    def execute_thinking(self, fn: Callable, *args, **kwargs) -> Any:
        """Run fn on CPU (always NumPy). Sampling and penalty functions."""
        return fn(*args, **kwargs)

    def __repr__(self) -> str:
        return (f"ExecutionEnvironment(inference={self.inference_backend}, "
                f"thinking={self.thinking_backend})")


def get_environment() -> ExecutionEnvironment:
    """Build an ExecutionEnvironment from current hardware capability."""
    try:
        from ..device import get_optimal_backend
        backend = get_optimal_backend()
    except Exception:
        backend = KERNEL_NUMPY
    return ExecutionEnvironment(inference_backend=backend)