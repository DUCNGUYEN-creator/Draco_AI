# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""GenerationRequest — wraps an InferenceRequest with engine-level
metadata (user_id, turn_id, think_mode) for logging and telemetry."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .inference_request import InferenceRequest


@dataclass
class GenerationRequest:
    inference: InferenceRequest
    user_id: Optional[str] = None
    turn_id: str = field(default_factory=lambda: f"t{int(time.time()*1000)}")
    think_mode: bool = False
    created_at: float = field(default_factory=time.time)
