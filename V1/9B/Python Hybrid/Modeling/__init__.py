# DracoAI V1 — modeling/__init__.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
FIXES (this revision):
  ✅ FIX-INIT-ALL-MISSING : ExecutionEnvironment, get_environment, and
     GenerationSession were imported at module level but absent from __all__.
     pyflakes correctly reported them as "imported but unused" because nothing
     in this file referenced them after the import.  Adding them to __all__
     documents that they are intentional public re-exports and silences the
     warning without removing useful public API surface.
"""
from .config import (
    ModelConfig, SINK_TOKENS, SPEC_THRESH, DEFAULT_TEMP, DEFAULT_TOP_P,
    MOE_NOISE_SCALE, ROPE_THETA, COMPUTE_DTYPE,
)
from .transformer import DracoTransformerV1, TransformerBlock, TransformerBridge

from .kv_cache.kv_cache     import KVCache
from .kv_cache.prefix_cache import PrefixCache
from .kv_cache.snapshot     import SnapshotStack
from .kv_cache.engram_cache import EngramCache, EngramBlock

from .quant.int4         import QuantizedLinear
from .quant.gguf_loader  import GGUFExporter

from .sampling.sampler   import Sampler
from .sampling.mirostat  import mirostat_v2
from .sampling.penalties import (
    apply_repetition_penalty, apply_frequency_penalty, apply_presence_penalty)

from .runtime.tensor_pool   import TensorPool
from .runtime.profiler      import InferenceProfiler
from .runtime.health        import HealthMonitor
from .runtime.precision     import DynamicPrecisionManager
from .runtime.wal           import WriteAheadLog
from .runtime.environment   import ExecutionEnvironment, get_environment
from .runtime.session       import GenerationSession

from .runtime.speculative import MTPHead, SpeculativeDecoder, SpeculativeTreeDecoder
from .runtime.scheduler   import RequestHandle, ContinuousBatchingScheduler

__version__ = "1.0.0"

__all__ = [
    # Config & constants
    "ModelConfig", "SINK_TOKENS", "SPEC_THRESH", "DEFAULT_TEMP", "DEFAULT_TOP_P",
    "MOE_NOISE_SCALE", "ROPE_THETA", "COMPUTE_DTYPE",
    # Core model
    "DracoTransformerV1", "TransformerBlock", "TransformerBridge",
    # KV cache
    "KVCache", "PrefixCache", "SnapshotStack",
    "EngramCache", "EngramBlock",
    # Quantisation
    "QuantizedLinear", "GGUFExporter",
    # Sampling
    "Sampler", "mirostat_v2",
    "apply_repetition_penalty", "apply_frequency_penalty", "apply_presence_penalty",
    # Runtime
    "TensorPool", "HealthMonitor", "DynamicPrecisionManager",
    "WriteAheadLog", "InferenceProfiler",
    "ExecutionEnvironment", "get_environment",
    "GenerationSession",
    # Speculative & scheduler
    "MTPHead", "SpeculativeDecoder", "SpeculativeTreeDecoder",
    "RequestHandle", "ContinuousBatchingScheduler",
]