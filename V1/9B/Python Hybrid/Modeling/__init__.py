# DracoAI V1 — modeling/__init__.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
DracoAI V1 — public API surface.

FIXES / NEW in this revision:
  ✅ FIX-INIT-ALL-MISSING          : ExecutionEnvironment, get_environment,
     and GenerationSession are exported in __all__.
  ✅ FEAT-TERNARY                   : TernaryLinear, ternarize_weight exported.
  ✅ FEAT-MLA                       : MLAProjection exported.
  ✅ FEAT-HYBRID-ATTENTION          : HybridAttentionConfig,
     build_default_global_layers exported.
  ✅ FEAT-MEDUSA                    : MedusaHeads, MedusaDecoder exported.
  ✅ FEAT-SELF-CORRECTION           : SelfCorrectionManager exported.
  ✅ FEAT-SPARSITY                  : SparsityPredictor, apply_sparsity_mask.
  ✅ FEAT-KV-QUANT                  : kv_quantize, kv_dequantize, kv_memory_bytes.
  ✅ FEAT-HEALTH-SIGNAL             : SelfCorrectionSignal exported.
"""
from .config import (
    ModelConfig, SINK_TOKENS, SPEC_THRESH, DEFAULT_TEMP, DEFAULT_TOP_P,
    MOE_NOISE_SCALE, ROPE_THETA, COMPUTE_DTYPE,
)
from .transformer import DracoTransformerV1, TransformerBlock, TransformerBridge

# KV cache
from .kv_cache.kv_cache     import KVCache
from .kv_cache.prefix_cache import PrefixCache
from .kv_cache.snapshot     import SnapshotStack
from .kv_cache.engram_cache import EngramCache, EngramBlock
from .kv_cache.kv_quant     import (kv_quantize, kv_dequantize,
                                    kv_quantize_batch, kv_dequantize_batch,
                                    kv_memory_bytes)

# Quantization
from .quant.int4            import QuantizedLinear
from .quant.ternary_linear  import TernaryLinear, ternarize_weight
from .quant.gguf_loader     import GGUFExporter

# Layers
from .layers.attention_mla    import MLAProjection
from .layers.hybrid_attention import HybridAttentionConfig, build_default_global_layers

# Sampling
from .sampling.sampler    import Sampler
from .sampling.mirostat   import mirostat_v2
from .sampling.penalties  import (
    apply_repetition_penalty, apply_frequency_penalty, apply_presence_penalty)

# Ops
from .ops.sparsity import SparsityPredictor, apply_sparsity_mask

# Runtime
from .runtime.tensor_pool     import TensorPool
from .runtime.profiler        import InferenceProfiler
from .runtime.health          import HealthMonitor, SelfCorrectionSignal
from .runtime.precision       import DynamicPrecisionManager
from .runtime.wal             import WriteAheadLog
from .runtime.environment     import ExecutionEnvironment, get_environment
from .runtime.session         import GenerationSession
from .runtime.speculative     import MTPHead, SpeculativeDecoder, SpeculativeTreeDecoder
from .runtime.medusa          import MedusaHeads, MedusaDecoder
from .runtime.scheduler       import RequestHandle, ContinuousBatchingScheduler
from .runtime.self_correction import SelfCorrectionManager

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
    "kv_quantize", "kv_dequantize",
    "kv_quantize_batch", "kv_dequantize_batch", "kv_memory_bytes",

    # Quantisation
    "QuantizedLinear", "TernaryLinear", "ternarize_weight", "GGUFExporter",

    # Layers
    "MLAProjection",
    "HybridAttentionConfig", "build_default_global_layers",

    # Sampling
    "Sampler", "mirostat_v2",
    "apply_repetition_penalty", "apply_frequency_penalty", "apply_presence_penalty",

    # Ops
    "SparsityPredictor", "apply_sparsity_mask",

    # Runtime
    "TensorPool", "InferenceProfiler",
    "HealthMonitor", "SelfCorrectionSignal",
    "DynamicPrecisionManager",
    "WriteAheadLog",
    "ExecutionEnvironment", "get_environment",
    "GenerationSession",
    "MTPHead", "SpeculativeDecoder", "SpeculativeTreeDecoder",
    "MedusaHeads", "MedusaDecoder",
    "RequestHandle", "ContinuousBatchingScheduler",
    "SelfCorrectionManager",
]