# DracoAI V1 — modeling/__init__.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
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
    "ModelConfig", "SINK_TOKENS", "SPEC_THRESH", "DEFAULT_TEMP", "DEFAULT_TOP_P",
    "MOE_NOISE_SCALE", "ROPE_THETA", "COMPUTE_DTYPE",
    "DracoTransformerV1", "TransformerBlock", "TransformerBridge",
    "KVCache", "PrefixCache", "SnapshotStack",
    "EngramCache", "EngramBlock",
    "QuantizedLinear", "GGUFExporter",
    "Sampler", "mirostat_v2",
    "apply_repetition_penalty", "apply_frequency_penalty", "apply_presence_penalty",
    "TensorPool", "HealthMonitor", "DynamicPrecisionManager",
    "WriteAheadLog", "InferenceProfiler",
    "MTPHead", "SpeculativeDecoder", "SpeculativeTreeDecoder",
    "RequestHandle", "ContinuousBatchingScheduler",
]