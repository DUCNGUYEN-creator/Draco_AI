# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
thinking_engine.config
========================
Centralized, dataclass-based configuration. A single ``EngineConfig``
instance is threaded through ``ThinkingEngine`` and handed to every
subsystem that needs tunable parameters — avoids "magic number" drift
between modules that the monolithic ``engine_v1.py`` was prone to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .exceptions import ConfigError


@dataclass
class MCTSConfig:
    n_sim: int = 8
    max_rollout_depth: int = 10
    exploration_c: float = 1.41421356  # sqrt(2), standard UCB1 constant


@dataclass
class ReasoningConfig:
    tot: MCTSConfig = field(default_factory=lambda: MCTSConfig(8, 10))
    plan_decomposer: MCTSConfig = field(default_factory=lambda: MCTSConfig(5, 8))
    abduction: MCTSConfig = field(default_factory=lambda: MCTSConfig(8, 10))
    goal_decomposer_rollout_depth: int = 20
    max_council_rounds: int = 3
    self_consistency_paths: int = 3
    max_workers: int = 4  # ThreadPoolExecutor size for the concurrent stage


@dataclass
class HallucinationConfig:
    """Tuning knobs for the Verification layer's deepest subsystem."""

    strategy: str = "balanced"  # "fast" | "balanced" | "paranoid" | "custom"
    cache_size: int = 512
    cache_ttl_seconds: float = 600.0
    calibration_method: str = "platt"  # platt | isotonic | beta | temperature | histogram
    fusion_method: str = "noisy_or"  # weighted | noisy_or | bayesian | dempster_shafer | logistic
    correlation_similarity_threshold: float = 0.82
    min_samples_for_calibration: int = 5
    risk_thresholds: "dict[str, float]" = field(
        default_factory=lambda: {
            "low": 0.15,
            "medium": 0.35,
            "high": 0.60,
            "critical": 0.85,
        }
    )


@dataclass
class MemoryConfig:
    working_memory_max_messages: int = 30
    context_token_budget: int = 2800
    forgetting_decay_rate: float = 0.1
    forgetting_threshold: float = 0.05
    rerank_top_k: int = 3
    rerank_threshold: float = 0.1


@dataclass
class SafetyConfig:
    ethical_threshold: float = 0.2
    enable_prompt_sanitizer: bool = True


@dataclass
class EngineConfig:
    max_experts: int = 4
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    hallucination: HallucinationConfig = field(default_factory=HallucinationConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    enable_parallel_reasoning: bool = True

    def validate(self) -> None:
        if not (1 <= self.max_experts <= 8):
            raise ConfigError(f"max_experts must be in [1, 8], got {self.max_experts}")
        if self.hallucination.strategy not in ("fast", "balanced", "paranoid", "custom"):
            raise ConfigError(
                f"Unknown hallucination strategy: {self.hallucination.strategy!r}"
            )

    @staticmethod
    def default() -> "EngineConfig":
        cfg = EngineConfig()
        cfg.validate()
        return cfg
