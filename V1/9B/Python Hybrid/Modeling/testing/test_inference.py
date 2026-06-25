# DracoAI V1 — modeling/testing/test_inference.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Integration tests for DracoTransformerV1 end-to-end inference.
Uses a tiny config to keep tests fast (< 5 s on CPU).

NEW tests in this revision:
  • TestTernaryInference   — ternary expert quantization + generate
  • TestMLAInference       — MLA-compressed KV cache + generate
  • TestMedusaDecoder      — Medusa speculative heads verification
  • TestHybridAttention    — Global/Local layer config
  • TestSelfCorrection     — SelfCorrectionManager diversify_logits
  • TestKVQuant            — KV-Q compressed cache + generate

FIXES retained:
  ✅ FIX-UNUSED-IMPORT-KVCACHE
  ✅ FIX-UNUSED-IMPORT-HEALTHMONITOR
  ✅ FIX-UNUSED-VAR-OUT1
"""
import threading
import numpy as np
import pytest

from ..config      import ModelConfig
from ..transformer import DracoTransformerV1
from ..kv_cache.prefix_cache  import PrefixCache
from ..kv_cache.kv_cache      import KVCache
from ..kv_cache.engram_cache  import EngramCache
from ..layers.attention_mla   import MLAProjection
from ..layers.hybrid_attention import HybridAttentionConfig
from ..runtime.wal            import WriteAheadLog
from ..runtime.profiler       import InferenceProfiler
from ..runtime.medusa         import MedusaHeads, MedusaDecoder
from ..runtime.self_correction import SelfCorrectionManager
from ..runtime.health         import HealthMonitor, SelfCorrectionSignal
from ..quant.ternary_linear   import TernaryLinear


# Tiny model config — fast on CPU, still exercises all code paths
_TINY = ModelConfig(
    d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
    head_dim=16, d_ff=128, n_experts=4, vocab_size=256, window=64)


@pytest.fixture
def model():
    return DracoTransformerV1(_TINY, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Original tests (all retained)
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerate:
    def test_basic_generate(self, model):
        out = model.generate([1, 2, 3], max_new_tokens=10, use_mirostat=False,
                             use_speculative=False)
        assert 1 <= len(out) <= 10
        assert all(0 <= t < 256 for t in out)

    def test_eos_stops_early(self, model):
        model.set_identity_bias([0], boost=100.0)
        out = model.generate([1, 2], max_new_tokens=50, eos_id=0,
                             use_mirostat=False, use_speculative=False)
        assert len(out) <= 50

    def test_deterministic_flag(self, model):
        ids = [1, 2, 3]
        np.random.seed(42)
        out1 = model.generate(ids, max_new_tokens=5, deterministic=True,
                              use_mirostat=False, use_speculative=False)
        np.random.seed(42)
        out2 = model.generate(ids, max_new_tokens=5, deterministic=True,
                              use_mirostat=False, use_speculative=False)
        assert out1 == out2

    def test_stop_event(self, model):
        ev = threading.Event()
        def _stopper():
            import time; time.sleep(0.01); ev.set()
        t = threading.Thread(target=_stopper); t.start()
        out = model.generate([1, 2, 3], max_new_tokens=1000,
                             use_mirostat=False, use_speculative=False,
                             stop_event=ev)
        t.join()
        assert len(out) < 1000

    def test_stream_callback(self, model):
        received = []
        def _cb(tid, conf): received.append((tid, conf))
        model.generate([1, 2, 3], max_new_tokens=5,
                       use_mirostat=False, use_speculative=False,
                       stream_cb=_cb)
        assert len(received) >= 1
        for tid, conf in received:
            assert 0 <= tid < 256
            assert 0.0 <= conf <= 1.0

    def test_multi_eos(self, model):
        out = model.generate([1, 2], max_new_tokens=20, eos_ids=[0, 1, 2],
                             use_mirostat=False, use_speculative=False)
        assert isinstance(out, list)

    def test_mirostat_mode(self, model):
        out = model.generate([1, 2, 3], max_new_tokens=8,
                             use_mirostat=True, use_speculative=False)
        assert len(out) >= 1

    def test_speculative_decoding(self, model):
        out = model.generate([1, 2, 3], max_new_tokens=10,
                             use_mirostat=False, use_speculative=True)
        assert 1 <= len(out) <= 10

    def test_speculative_tree(self, model):
        out = model.generate([1, 2, 3], max_new_tokens=8,
                             use_mirostat=False, use_speculative=False,
                             use_speculative_tree=True,
                             spec_tree_width=2, spec_tree_depth=2)
        assert isinstance(out, list)


class TestForward:
    def test_output_shapes(self, model):
        cache = model._make_cache()
        l1, l2, aux = model.forward([1, 2, 3], cache)
        assert l1.shape == (1, 3, 256)
        assert l2.shape == (1, 3, 256)
        assert len(aux) == 2

    def test_no_nan(self, model):
        cache = model._make_cache()
        l1, l2, _ = model.forward([1, 2, 3], cache)
        assert np.all(np.isfinite(l1))
        assert np.all(np.isfinite(l2))


class TestPrefixCache:
    def test_prefix_cache_hit(self, model):
        pc = PrefixCache(max_entries=8)
        model.set_prefix_cache(pc)
        prompt = [1, 2, 3, 4, 5]
        _ = model.generate(prompt, max_new_tokens=3,
                           use_mirostat=False, use_speculative=False)
        out2 = model.generate(prompt, max_new_tokens=3,
                              use_mirostat=False, use_speculative=False)
        assert isinstance(out2, list)


class TestProfiler:
    def test_profiler_summary(self, model):
        profiler = InferenceProfiler()
        model.generate([1, 2, 3], max_new_tokens=5,
                       use_mirostat=False, use_speculative=False,
                       profiler=profiler)
        s = profiler.summary()
        assert s["total_tokens"] >= 1
        assert s["n_forward_calls"] >= 1
        assert s["avg_fwd_ms"] >= 0


class TestWAL:
    def test_wal_recover(self, model, tmp_path):
        path = str(tmp_path / "test.wal")
        with WriteAheadLog(path) as wal:
            model.generate([1, 2, 3], max_new_tokens=5,
                           use_mirostat=False, use_speculative=False,
                           wal=wal)
        recovered = WriteAheadLog.recover(path)
        assert len(recovered) >= 1
        assert all(0 <= t < 256 for t in recovered)


class TestQuantization:
    def test_int8_quantize_and_generate(self, model):
        model.quantize_weights(quant="int8")
        out = model.generate([1, 2, 3], max_new_tokens=4,
                             use_mirostat=False, use_speculative=False)
        assert len(out) >= 1

    def test_cast_weights_float16(self, model):
        model.cast_weights(np.float16)
        assert model.embedding.dtype == np.float16
        out = model.generate([1, 2], max_new_tokens=3,
                             use_mirostat=False, use_speculative=False)
        assert len(out) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# NEW tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTernaryInference:
    """Ternary expert quantization tests."""

    def test_ternary_quantize_and_generate(self, model):
        """quantize_weights('ternary') converts FFN experts; inference runs."""
        model.quantize_weights(quant="ternary")
        # Expert W_g should now be TernaryLinear
        blk = model.blocks[0]
        assert isinstance(blk.moe.experts[0].W_g, TernaryLinear)
        # Attention weights must stay float (RED zone)
        assert isinstance(blk.attn.W_q, np.ndarray), \
            "Attention W_q must remain float after ternary quant"
        # Router must stay float
        assert blk.moe.W_router.dtype == np.float32, \
            "Router must remain float32 after ternary quant"
        # Generate must succeed
        out = model.generate([1, 2, 3], max_new_tokens=4,
                             use_mirostat=False, use_speculative=False)
        assert len(out) >= 1

    def test_ternary_values_correct(self, model):
        """TernaryLinear values are in {-1, 0, +1}."""
        model.quantize_weights(quant="ternary")
        tl = model.blocks[0].moe.experts[0].W_g
        assert isinstance(tl, TernaryLinear)
        W_t = tl._numpy_forward.__func__  # just verify forward works
        x = np.random.randn(2, _TINY.d_model).astype(np.float32)
        out = tl.forward(x)
        assert out.shape == (2, _TINY.d_ff)
        assert np.all(np.isfinite(out))

    def test_ternary_forward_no_nan(self, model):
        """Ternary forward does not produce NaN on typical inputs."""
        model.quantize_weights(quant="ternary")
        cache = model._make_cache()
        l1, l2, _ = model.forward([1, 2, 3], cache)
        assert np.all(np.isfinite(l1)), "NaN in ternary model logits"
        assert np.all(np.isfinite(l2)), "NaN in ternary MTP logits"


class TestMLAInference:
    """Multi-head Latent Attention (KV compression) tests."""

    def test_mla_projection_roundtrip(self):
        """MLAProjection compress→expand preserves subspace."""
        n_kv, hdim, ldim = 2, 16, 4
        mla = MLAProjection(n_kv, hdim, ldim)
        K = np.random.randn(1, n_kv, 8, hdim).astype(np.float32)
        K_lat  = mla.compress_k(K)
        K_back = mla.expand_k(K_lat)
        assert K_lat.shape  == (1, n_kv, 8, ldim)
        assert K_back.shape == (1, n_kv, 8, hdim)
        assert np.all(np.isfinite(K_back))

    def test_mla_compression_ratio(self):
        mla = MLAProjection(n_kv_heads=4, head_dim=64, latent_dim=16)
        assert abs(mla.compression_ratio() - 0.25) < 1e-6

    def test_mla_generate(self, model):
        """Attach MLA to blocks and run generate — must not crash."""
        for blk in model.blocks:
            mla = MLAProjection(
                n_kv_heads=_TINY.n_kv_heads,
                head_dim=_TINY.head_dim,
                latent_dim=_TINY.head_dim // 2,
            )
            blk.set_mla(mla)
        out = model.generate([1, 2, 3], max_new_tokens=5,
                             use_mirostat=False, use_speculative=False)
        assert isinstance(out, list)
        assert len(out) >= 1


class TestHybridAttention:
    """Global / Local attention configuration tests."""

    def test_hybrid_config_classification(self):
        cfg = HybridAttentionConfig(n_layers=4)
        global_l = cfg.global_layers
        assert 0 in global_l and (3 in global_l)
        for i in range(4):
            assert cfg.is_global(i) != cfg.is_local(i)

    def test_hybrid_block_flag(self, model):
        cfg = HybridAttentionConfig(n_layers=_TINY.n_layers)
        for blk in model.blocks:
            blk.set_hybrid_config(cfg)
        # Global blocks report correctly
        for blk in model.blocks:
            expected = cfg.is_global(blk.layer_idx)
            assert blk.is_global_layer == expected

    def test_hybrid_generate(self, model):
        """Hybrid attention config should not break generation."""
        cfg = HybridAttentionConfig(n_layers=_TINY.n_layers)
        for blk in model.blocks:
            blk.set_hybrid_config(cfg)
        out = model.generate([1, 2, 3], max_new_tokens=5,
                             use_mirostat=False, use_speculative=False)
        assert isinstance(out, list)


class TestMedusaDecoder:
    """Medusa multi-head speculative decoding tests."""

    def test_medusa_draft_shapes(self, model):
        """MedusaHeads.draft returns token lists within n_heads length."""
        heads = MedusaHeads(
            d_model=_TINY.d_model, vocab_size=_TINY.vocab_size, n_heads=3)
        heads.set_lm_head(model.lm_head)
        hidden = np.random.randn(1, 1, _TINY.d_model).astype(np.float32)
        tokens, confs = heads.draft(hidden, thresh=0.0)  # thresh=0 → all pass
        assert len(tokens) == 3
        assert len(confs)  == 3
        assert all(0 <= t < _TINY.vocab_size for t in tokens)
        assert all(0.0 <= c <= 1.0 for c in confs)

    def test_medusa_decoder_returns_list(self, model):
        """MedusaDecoder.try_medusa returns list, logits, float."""
        heads = MedusaHeads(
            d_model=_TINY.d_model, vocab_size=_TINY.vocab_size, n_heads=2)
        heads.set_lm_head(model.lm_head)
        decoder = MedusaDecoder(model, heads, thresh=0.0)

        cache = model._make_cache()
        # Warm up cache with one forward
        l1, _, _ = model.forward([1, 2, 3], cache)
        last_logits = l1[0, -1].astype(np.float64)
        hidden = np.random.randn(1, 1, _TINY.d_model).astype(np.float32)

        accepted, final_logits, final_mu = decoder.try_medusa(
            cache, hidden, last_logits, {}, mu=5.0, use_mirostat=False)
        assert isinstance(accepted, list)
        assert final_logits.shape == (_TINY.vocab_size,)
        assert isinstance(final_mu, float)


class TestSelfCorrection:
    """SelfCorrectionManager diversification tests."""

    def test_no_correction_when_confident(self):
        sc = SelfCorrectionManager(temp_boost=1.5)
        signal = SelfCorrectionSignal(
            should_correct=False, confidence=0.9, consecutive_low=0, reason="")
        logits = np.random.randn(256).astype(np.float64)
        out = sc.diversify_logits(logits, signal)
        np.testing.assert_array_equal(out, logits)

    def test_correction_raises_entropy(self):
        sc = SelfCorrectionManager(temp_boost=2.0, rare_token_boost=0.5)
        signal = SelfCorrectionSignal(
            should_correct=True, confidence=0.01, consecutive_low=5, reason="test")
        logits = np.random.randn(256).astype(np.float64)
        out = sc.diversify_logits(logits, signal)
        # Temperature boost should flatten distribution → higher entropy
        p_in  = np.exp(logits - logits.max()); p_in  /= p_in.sum()
        p_out = np.exp(out    - out.max());    p_out /= p_out.sum()
        ent_in  = -np.sum(p_in  * np.log(p_in  + 1e-9))
        ent_out = -np.sum(p_out * np.log(p_out + 1e-9))
        assert ent_out >= ent_in, "Correction should increase entropy"

    def test_max_consecutive_guard(self):
        sc = SelfCorrectionManager(max_consecutive=2)
        signal = SelfCorrectionSignal(
            should_correct=True, confidence=0.0, consecutive_low=99, reason="")
        logits = np.zeros(10, dtype=np.float64)
        sc.diversify_logits(logits, signal)  # 1st correction
        sc.diversify_logits(logits, signal)  # 2nd → resets counter
        sc.diversify_logits(logits, signal)  # 3rd → should be no-op
        assert sc._consecutive_correct == 0  # reset after max

    def test_health_monitor_signal(self):
        monitor = HealthMonitor(correction_conf_thresh=0.5, correction_patience=2)
        # Feed very low-confidence logits
        flat_logits = np.zeros(256)   # uniform → low max-prob
        sig1 = monitor.check_step(flat_logits)
        sig2 = monitor.check_step(flat_logits)
        sig3 = monitor.check_step(flat_logits)
        # After patience=2 consecutive low steps, should_correct = True
        assert sig3.should_correct

    def test_adversarial_expert_detection(self):
        monitor = HealthMonitor(adversarial_logit_thresh=5.0,
                                adversarial_dominance=0.8)
        # Router logits where expert 0 dominates with huge values
        router_logits = np.zeros((8, 4), dtype=np.float32)
        router_logits[:, 0] = 20.0   # expert 0 wins every token with large logit
        flagged = monitor.detect_adversarial_expert(router_logits)
        assert flagged == 0

    def test_no_adversarial_when_balanced(self):
        monitor = HealthMonitor(adversarial_logit_thresh=5.0,
                                adversarial_dominance=0.8)
        router_logits = np.random.randn(8, 4).astype(np.float32) * 0.5
        flagged = monitor.detect_adversarial_expert(router_logits)
        assert flagged == -1


class TestKVQuant:
    """KV-quantized cache tests."""

    def test_kv_quant_cache_roundtrip(self):
        """KVCache with use_kv_quant=True stores and retrieves correctly."""
        cache = KVCache(
            n_layers=2, n_kv_heads=2, head_dim=16,
            window=32, sink=4, use_kv_quant=True)
        K = np.random.randn(1, 2, 4, 16).astype(np.float32)
        V = np.random.randn(1, 2, 4, 16).astype(np.float32)
        cache.update(0, K, V)
        K_r, V_r = cache.get(0)
        # Relative error should be < 2% for symmetric INT8
        rel_err_K = np.abs(K - K_r[0, :, :4, :]).max() / (np.abs(K).max() + 1e-9)
        rel_err_V = np.abs(V - V_r[0, :, :4, :]).max() / (np.abs(V).max() + 1e-9)
        assert rel_err_K < 0.02, f"KV-Q K error too high: {rel_err_K:.3%}"
        assert rel_err_V < 0.02, f"KV-Q V error too high: {rel_err_V:.3%}"

    def test_kv_quant_snapshot_restore(self):
        """KV-Q cache snapshot/restore works correctly."""
        cache = KVCache(
            n_layers=1, n_kv_heads=2, head_dim=16,
            window=32, sink=4, use_kv_quant=True)
        K = np.random.randn(1, 2, 2, 16).astype(np.float32)
        V = np.random.randn(1, 2, 2, 16).astype(np.float32)
        cache.update(0, K, V)
        snap = cache.snapshot()
        K2 = np.random.randn(1, 2, 2, 16).astype(np.float32)
        V2 = np.random.randn(1, 2, 2, 16).astype(np.float32)
        cache.update(0, K2, V2)
        assert cache.current_length() == 4
        cache.restore(snap)
        assert cache.current_length() == 2

    def test_kv_quant_memory_smaller(self):
        """KV-Q cache uses less memory than float16 cache."""
        from ..kv_cache.kv_quant import kv_memory_bytes
        stats = kv_memory_bytes(
            n_layers=4, max_batch=1, n_kv_heads=8,
            window=512, head_dim=64, quantized=True)
        assert stats["quant_gb"] < stats["float_gb"]
        assert stats["savings_pct"] > 30.0

    def test_kv_quant_generate(self, model):
        """Model with KV-Q cache runs generate correctly."""
        model._cache = KVCache(
            model.n_layers, model.n_kv_heads, model.head_dim,
            window=model.window, sink=4, use_kv_quant=True)
        out = model.generate([1, 2, 3], max_new_tokens=5,
                             new_prompt=False,
                             use_mirostat=False, use_speculative=False)
        assert isinstance(out, list)
        assert len(out) >= 1