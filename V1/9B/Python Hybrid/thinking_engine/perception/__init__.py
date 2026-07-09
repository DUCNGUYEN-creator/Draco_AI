# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""thinking_engine.perception — Infrastructure layer, stage 1 of the pipeline.

Turns a raw user message into a structured intent + a sanitized,
rewritten query. Never performs reasoning or verification — only
understanding.
"""

from .prompt.sanitizer import PromptSanitizer
from .prompt.contextual_rewriter import ContextualPromptRewriter
from .prompt.instruction_chain import InstructionChainParser
from .prompt.compiler import PromptCompiler
from .language.language_detector import LanguageDetector
from .language.entity_extractor import EntityExtractor
from .language.metaphor_detector import MetaphorDetector
from .language.sentiment import SentimentAnalyzer
from .understanding.intent_detector import IntentDetector
from .understanding.difficulty import DifficultyScorer
from .understanding.dual_process import DualProcessDecider
from .understanding.task_classifier import TaskClassifier
from .understanding.ambiguity_detector import AmbiguityDetector

__all__ = [
    "PromptSanitizer",
    "ContextualPromptRewriter",
    "InstructionChainParser",
    "PromptCompiler",
    "LanguageDetector",
    "EntityExtractor",
    "MetaphorDetector",
    "SentimentAnalyzer",
    "IntentDetector",
    "DifficultyScorer",
    "DualProcessDecider",
    "TaskClassifier",
    "AmbiguityDetector",
]
