# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
hallucination.exceptions
==========================
Re-exports the hallucination-specific exception hierarchy from the
engine-level exceptions.py under this package's own namespace, PLUS
defines any hallucination-internal exceptions not needed by the rest of
the engine (e.g. PipelineStageError — only meaningful within this
sub-package).
"""

from ...exceptions import (  # noqa: F401  (re-exports)
    CalibrationError,
    CorrelationError,
    FusionError,
    RegistryError,
    StrategyError,
    TelemetryError,
    VerificationError,
    VerifierError,
)


class PipelineStageError(VerificationError):
    """Raised when a specific pipeline stage (e.g. EvidencePipeline,
    VerificationPipeline) fails with an unrecoverable error that the
    pipeline's own per-claim try/except cannot handle gracefully."""


__all__ = [
    "CalibrationError",
    "CorrelationError",
    "FusionError",
    "PipelineStageError",
    "RegistryError",
    "StrategyError",
    "TelemetryError",
    "VerificationError",
    "VerifierError",
]
