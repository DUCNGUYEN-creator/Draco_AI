# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
hallucination.config
======================
Re-exports thinking_engine.config.HallucinationConfig under the
hallucination package's own namespace, so intra-package code can do:

    from .config import HallucinationConfig

without reaching up three package levels to the root config. Zero
additional logic here — a pure re-export keeps the full config DRY in
one place (config.py at the engine root) while satisfying the import
locality preference inside this deep sub-package.
"""

from ...config import HallucinationConfig  # noqa: F401  (re-export)

__all__ = ["HallucinationConfig"]
