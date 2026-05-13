# DracoAI V1 — modeling/quant/gguf_loader.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""GGUF FP16 exporter for llama.cpp. Requires: pip install gguf"""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING
import numpy as np
from .int4 import QuantizedLinear

if TYPE_CHECKING:
    from ..transformer import DracoTransformerV1

__all__ = ["GGUFExporter"]

logger = logging.getLogger(__name__)


class GGUFExporter:
    _ATTN_MAP = {"W_q": "attn_q",  "W_k": "attn_k",
                 "W_v": "attn_v",  "W_o": "attn_output"}
    _FFN_MAP  = {"W_g": "ffn_gate", "W_u": "ffn_up", "W_d": "ffn_down"}

    def __init__(self, model: "DracoTransformerV1"):
        self.model = model

    def write_gguf(self, output_path: str):
        try:
            from gguf import GGUFWriter
        except ImportError:
            raise ImportError(
                "pip install gguf  "
                "(https://github.com/ggerganov/llama.cpp/tree/master/gguf-py)")

        m   = self.model
        cfg = m.config if isinstance(m.config, dict) else m.config.to_dict()
        w   = GGUFWriter(output_path, "llama")

        w.add_name("DracoAI-V1")
        w.add_description("DracoAI V1 MoE Transformer")
        w.add_uint32("llama.context_length",          cfg.get("window",     1024))
        w.add_uint32("llama.embedding_length",        cfg.get("d_model",     128))
        w.add_uint32("llama.block_count",             cfg.get("n_layers",      4))
        w.add_uint32("llama.attention.head_count",    cfg.get("n_heads",       4))
        w.add_uint32("llama.attention.head_count_kv", cfg.get("n_kv_heads",    2))
        w.add_float32("llama.attention.layer_norm_rms_epsilon", 1e-6)
        w.add_uint32("llama.vocab_size",              cfg.get("vocab_size", 151936))
        w.add_uint32("llama.rope.dimension_count",    cfg.get("head_dim",     32))
        w.add_uint32("llama.expert_count",            cfg.get("n_experts",     8))
        w.add_uint32("llama.expert_used_count",       2)

        w.add_tensor("token_embd.weight",  m.embedding.astype(np.float16))
        w.add_tensor("output_norm.weight", m.norm_f.astype(np.float16))
        w.add_tensor("output.weight",      m.lm_head.astype(np.float16))

        for i, blk in enumerate(m.blocks):
            pfx = f"blk.{i}"

            def _add(name, arr):
                v = arr.dequantize() if isinstance(arr, QuantizedLinear) else arr
                w.add_tensor(f"{pfx}.{name}.weight", v.astype(np.float16))

            _add("attn_norm", blk.norm1)
            _add("ffn_norm",  blk.norm2)
            for attr, tname in self._ATTN_MAP.items():
                W = getattr(blk.attn, attr)
                v = W.dequantize() if isinstance(W, QuantizedLinear) else W
                w.add_tensor(f"{pfx}.{tname}.weight", v.T.astype(np.float16))
            w.add_tensor(f"{pfx}.ffn_gate_inp.weight",
                         blk.moe.W_router.astype(np.float16))
            for e, exp in enumerate(blk.moe.experts):
                for attr, tname in self._FFN_MAP.items():
                    W = getattr(exp, attr)
                    v = W.dequantize() if isinstance(W, QuantizedLinear) else W
                    w.add_tensor(f"{pfx}.{tname}_exps.{e}.weight", v.T.astype(np.float16))
            for attr, tname in self._FFN_MAP.items():
                W = getattr(blk.moe.shared, attr)
                v = W.dequantize() if isinstance(W, QuantizedLinear) else W
                w.add_tensor(f"{pfx}.{tname}_shexp.weight", v.T.astype(np.float16))

        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        logger.info(
            "[DracoAI] GGUF written to %s  "
            "(quantise: llama-quantize %s out.gguf Q4_K_M)", output_path, output_path)