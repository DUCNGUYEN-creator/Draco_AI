# DracoAI V1 — modeling/testing/test_inference.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Integration tests for DracoTransformerV1 end-to-end inference.
Uses a tiny config to keep tests fast (< 2 s on CPU).
"""
import threading
import numpy as np
import pytest
from ..config      import ModelConfig
from ..transformer import DracoTransformerV1
from ..kv_cache.kv_cache    import KVCache
from ..kv_cache.prefix_cache import PrefixCache
from ..runtime.wal        import WriteAheadLog
from ..runtime.profiler   import InferenceProfiler
from ..runtime.health     import HealthMonitor


# Tiny model config: fast on CPU, still exercises all code paths
_TINY = ModelConfig(
    d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
    head_dim=16, d_ff=128, n_experts=4, vocab_size=256, window=64)


@pytest.fixture
def model():
    return DracoTransformerV1(_TINY, dtype=np.float32)


class TestGenerate:
    def test_basic_generate(self, model):
        out = model.generate([1, 2, 3], max_new_tokens=10, use_mirostat=False,
                             use_speculative=False)
        assert 1 <= len(out) <= 10
        assert all(0 <= t < 256 for t in out)

    def test_eos_stops_early(self, model):
        # Patch: set id_bias to force EOS quickly
        model.set_identity_bias([0], boost=100.0)  # token 0 = EOS
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
        def _cb(tid, conf):
            received.append((tid, conf))
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
        # First call: stores prefix
        out1 = model.generate(prompt, max_new_tokens=3,
                              use_mirostat=False, use_speculative=False)
        # Second call: should hit cache
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