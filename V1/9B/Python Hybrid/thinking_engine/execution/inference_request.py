# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""InferenceRequest — structured container for everything the LLM bridge
needs to run a forward pass, assembled by pipeline.py from ThinkingState."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class InferenceRequest:
    messages: List[dict]
    max_new_tokens: int = 512
    temp: float = 0.7
    top_p: float = 0.9
    min_p: float = 0.05
    use_mirostat: bool = True
    use_speculative: bool = True
    adaptive_temp: bool = False
    intent_boost: Optional[List[float]] = None
    intent_bias: Optional[List[float]] = None
    stream_cb: Optional[Callable] = None
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)
