# DracoAI V1 — modeling/testing/test_cache.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Unit tests for KVCache, PrefixCache, SnapshotStack, EngramCache, KV-Q.

FIXES retained:
  ✅ FIX-PREFIX-UNPACK : PrefixCache.get() returns a 4-tuple.

NEW tests in this revision:
  • TestKVCacheQuantized      — INT8 KV-Q store/retrieve/snapshot/checkpoint
  • TestEngramAdvanceAPI      — advance_committed_end() lock-safe pointer
  • TestKVQuantUtils          — kv_quantize / kv_dequantize utilities
"""
import numpy as np
import pytest
from ..kv_cache.kv_cache     import KVCache
from ..kv_cache.prefix_cache import PrefixCache
from ..kv_cache.snapshot     import SnapshotStack
from ..kv_cache.engram_cache import EngramCache, EngramBlock
from ..kv_cache.kv_quant     import (kv_quantize, kv_dequantize,
                                      kv_quantize_batch, kv_dequantize_batch,
                                      kv_memory_bytes)


# ─────────────────────────────────────────────────────────────────────────────
# Original tests (all retained)
# ─────────────────────────────────────────────────────────────────────────────

class TestKVCache:
    @pytest.fixture
    def cache(self):
        return KVCache(n_layers=2, n_kv_heads=2, head_dim=16, window=32, sink=4)

    def _make_kv(self, n_kv_heads=2, seq=1, head_dim=16):
        K = np.random.randn(1, n_kv_heads, seq, head_dim).astype(np.float32)
        V = np.random.randn(1, n_kv_heads, seq, head_dim).astype(np.float32)
        return K, V

    def test_update_and_get(self, cache):
        K, V = self._make_kv()
        cache.update(0, K, V)
        K_r, V_r = cache.get(0)
        assert K_r.shape == (1, 2, 1, 16)
        np.testing.assert_allclose(K_r[0, :, 0, :], K[0, :, 0, :], atol=1e-3)

    def test_pos_increments(self, cache):
        for _ in range(5):
            K, V = self._make_kv()
            cache.update(0, K, V)
        assert cache.current_length() == 5

    def test_get_pos_public(self, cache):
        K, V = self._make_kv()
        cache.update(0, K, V)
        assert cache.get_pos(0) == 1

    def test_snapshot_restore(self, cache):
        K, V = self._make_kv()
        cache.update(0, K, V)
        snap = cache.snapshot()
        K2, V2 = self._make_kv()
        cache.update(0, K2, V2)
        assert cache.current_length() == 2
        cache.restore(snap)
        assert cache.current_length() == 1

    def test_reset_single_slot(self, cache):
        K, V = self._make_kv()
        cache.update(0, K, V, batch_idx=0)
        cache.reset(batch_idx=0)
        assert cache.current_length(batch_idx=0) == 0

    def test_sliding_window(self):
        cache = KVCache(n_layers=1, n_kv_heads=1, head_dim=4, window=8, sink=2)
        for _ in range(12):
            K = np.ones((1, 1, 1, 4), dtype=np.float32)
            V = np.ones((1, 1, 1, 4), dtype=np.float32)
            cache.update(0, K, V)
        assert cache.current_length() == 8

    def test_vectorised_batch_update(self, cache):
        K = np.random.randn(1, 2, 4, 16).astype(np.float32)
        V = np.random.randn(1, 2, 4, 16).astype(np.float32)
        cache.update(0, K, V)
        assert cache.current_length() == 4

    def test_checkpoint_roundtrip(self, tmp_path, cache):
        K, V = self._make_kv()
        cache.update(0, K, V)
        ckpt = str(tmp_path / "cache_ckpt")
        cache.save_checkpoint(ckpt)
        loaded = KVCache.load_checkpoint(ckpt)
        assert loaded.current_length() == cache.current_length()


class TestPrefixCache:
    def _make_snap(self):
        return {
            "_type":      "full",
            "_K":         np.zeros((2, 1, 2, 8, 4)),
            "_V":         np.zeros((2, 1, 2, 8, 4)),
            "_batch_idx": 0,
            "_cache_pos": 5,
            "_escalated": False,
        }

    def test_put_get_roundtrip(self):
        pc  = PrefixCache(max_entries=4)
        ids = [1, 2, 3, 4, 5]
        logits = np.zeros(100)
        pc.put(ids, self._make_snap(), logits)

        result = pc.get(ids)
        assert result is not None
        snap, plen, ll, engram_snap = result
        assert plen == 5
        assert ll is not None
        assert engram_snap is None

    def test_put_get_with_engram_snap(self):
        pc  = PrefixCache(max_entries=4)
        ids = [10, 20, 30]
        fake_engram = {"_n_blocks": 2, "_last_committed_end": 256}
        pc.put(ids, self._make_snap(), engram_snap=fake_engram)

        result = pc.get(ids)
        assert result is not None
        _, plen, _, retrieved_engram = result
        assert plen == 3
        assert retrieved_engram == fake_engram

    def test_lru_eviction(self):
        pc = PrefixCache(max_entries=2)
        for i in range(3):
            snap = {
                "_type":      "full",
                "_K":         np.zeros((1, 1, 1, 4, 4)),
                "_V":         np.zeros((1, 1, 1, 4, 4)),
                "_batch_idx": 0,
                "_cache_pos": 1,
                "_escalated": False,
            }
            pc.put([i], snap)
        assert len(pc) == 2

    def test_miss_returns_none(self):
        pc = PrefixCache()
        assert pc.get([99, 100]) is None

    def test_invalidate(self):
        pc  = PrefixCache()
        ids = [1, 2, 3]
        pc.put(ids, self._make_snap())
        assert pc.get(ids) is not None
        pc.invalidate(ids)
        assert pc.get(ids) is None

    def test_legacy_4tuple_handled(self):
        import time
        pc = PrefixCache(max_entries=4)
        ids = [7, 8, 9]
        h   = pc._hash(ids)
        snap = self._make_snap()
        snap["_cache_pos"] = 3
        pc._store[h] = (snap, 3, time.perf_counter(), np.zeros(10))

        result = pc.get(ids)
        assert result is not None
        _, plen, ll, engram_snap = result
        assert plen == 3
        assert engram_snap is None


class TestSnapshotStack:
    def test_push_pop(self):
        cache = KVCache(n_layers=1, n_kv_heads=1, head_dim=4, window=16, sink=2)
        K = np.random.randn(1, 1, 1, 4).astype(np.float32)
        V = np.random.randn(1, 1, 1, 4).astype(np.float32)
        cache.update(0, K, V)
        stack = SnapshotStack(cache)
        stack.push()
        assert stack.depth == 1
        cache.update(0, K, V)
        assert cache.current_length() == 2
        stack.pop()
        assert cache.current_length() == 1
        assert stack.depth == 0

    def test_commit_no_restore(self):
        cache = KVCache(n_layers=1, n_kv_heads=1, head_dim=4, window=16, sink=2)
        K = np.random.randn(1, 1, 1, 4).astype(np.float32)
        V = np.random.randn(1, 1, 1, 4).astype(np.float32)
        cache.update(0, K, V)
        stack = SnapshotStack(cache)
        stack.push()
        cache.update(0, K, V)
        stack.commit()
        assert cache.current_length() == 2
        assert stack.depth == 0

    def test_rollback_to(self):
        cache = KVCache(n_layers=1, n_kv_heads=1, head_dim=4, window=16, sink=2)
        K = np.random.randn(1, 1, 1, 4).astype(np.float32)
        V = np.random.randn(1, 1, 1, 4).astype(np.float32)
        stack = SnapshotStack(cache)
        stack.push()
        cache.update(0, K, V)
        stack.push()
        cache.update(0, K, V)
        assert cache.current_length() == 2
        stack.rollback_to(1)
        assert stack.depth == 1
        assert cache.current_length() == 1


# ─────────────────────────────────────────────────────────────────────────────
# NEW tests
# ─────────────────────────────────────────────────────────────────────────────

class TestKVCacheQuantized:
    """INT8 KV-Q cache tests."""

    @pytest.fixture
    def qcache(self):
        return KVCache(
            n_layers=2, n_kv_heads=2, head_dim=16,
            window=32, sink=4, use_kv_quant=True)

    def _kv(self, seq=1):
        K = np.random.randn(1, 2, seq, 16).astype(np.float32)
        V = np.random.randn(1, 2, seq, 16).astype(np.float32)
        return K, V

    def test_update_get_accuracy(self, qcache):
        """KV-Q round-trip error < 2% relative."""
        K, V = self._kv()
        qcache.update(0, K, V)
        K_r, V_r = qcache.get(0)
        assert K_r.shape == (1, 2, 1, 16)
        rel_k = np.abs(K - K_r[0, :, :1, :]).max() / (np.abs(K).max() + 1e-9)
        rel_v = np.abs(V - V_r[0, :, :1, :]).max() / (np.abs(V).max() + 1e-9)
        assert rel_k < 0.02, f"K relative error too high: {rel_k:.3%}"
        assert rel_v < 0.02, f"V relative error too high: {rel_v:.3%}"

    def test_snapshot_restore_quant(self, qcache):
        """Snapshot/restore works correctly with KV-Q."""
        K, V = self._kv()
        qcache.update(0, K, V)
        snap = qcache.snapshot()
        K2, V2 = self._kv()
        qcache.update(0, K2, V2)
        assert qcache.current_length() == 2
        qcache.restore(snap)
        assert qcache.current_length() == 1

    def test_reset_clears_scales(self, qcache):
        """reset() zeroes scale arrays too."""
        K, V = self._kv()
        qcache.update(0, K, V)
        qcache.reset()
        assert qcache.current_length() == 0
        assert (qcache._K_scale == 1.0).all()

    def test_memory_footprint_smaller(self, qcache):
        """KV-Q cache has smaller memory footprint than float16."""
        float_bytes = 2 * 1 * 2 * 32 * 16 * 2  # K+V, float16
        q_bytes = qcache.memory_bytes()
        # float16 equivalent: 2 layers * 2 heads * 32 window * 16 hdim * 2 bytes * 2 KV
        float16_equiv = 2 * 1 * 2 * 32 * 16 * 2 * 2
        assert q_bytes < float16_equiv

    def test_checkpoint_roundtrip_quant(self, tmp_path, qcache):
        K, V = self._kv(seq=3)
        qcache.update(0, K, V)
        path = str(tmp_path / "qcache")
        qcache.save_checkpoint(path)
        loaded = KVCache.load_checkpoint(path)
        assert loaded._use_kv_quant
        assert loaded.current_length() == qcache.current_length()

    def test_delta_snapshot_with_quant(self, qcache):
        """Delta snapshots record float32 old-values and restore correctly."""
        K, V = self._kv()
        qcache.update(0, K, V)
        snap = qcache.snapshot(delta_threshold=10)
        K2, V2 = self._kv()
        qcache.update(0, K2, V2, snap=snap)
        assert qcache.current_length() == 2
        qcache.restore(snap)
        assert qcache.current_length() == 1

    def test_sliding_window_quant(self):
        """Ring buffer wrap works correctly with KV-Q."""
        cache = KVCache(n_layers=1, n_kv_heads=1, head_dim=4,
                        window=8, sink=2, use_kv_quant=True)
        for _ in range(12):
            K = np.ones((1, 1, 1, 4), dtype=np.float32)
            V = np.ones((1, 1, 1, 4), dtype=np.float32)
            cache.update(0, K, V)
        assert cache.current_length() == 8
        K_r, V_r = cache.get(0)
        assert K_r.shape[2] == 8


class TestEngramAdvanceAPI:
    """advance_committed_end() lock-safety tests."""

    def _make_engram(self):
        return EngramCache(
            n_layers=1, n_kv_heads=1, head_dim=8,
            d_model=16, block_size=4)

    def test_advance_moves_pointer(self):
        eng = self._make_engram()
        assert eng._last_committed_end == 0
        eng.advance_committed_end(16)
        assert eng._last_committed_end == 16

    def test_advance_is_monotonic(self):
        """advance_committed_end never moves pointer backward."""
        eng = self._make_engram()
        eng.advance_committed_end(32)
        eng.advance_committed_end(16)   # backward → no-op
        assert eng._last_committed_end == 32

    def test_advance_idempotent(self):
        eng = self._make_engram()
        eng.advance_committed_end(8)
        eng.advance_committed_end(8)    # same value → no-op
        assert eng._last_committed_end == 8

    def test_advance_concurrent(self):
        """Concurrent calls are safe (lock protected)."""
        import threading
        eng = self._make_engram()
        errors = []

        def _worker(end_pos):
            try:
                eng.advance_committed_end(end_pos)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_worker, args=(i * 4,))
                   for i in range(1, 16)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors, f"Thread errors: {errors}"
        # Final value must be max of all submitted values
        assert eng._last_committed_end == 15 * 4

    def test_commit_block_uses_lock(self):
        """commit_block() respects advance_committed_end sentinel."""
        eng = self._make_engram()
        eng.advance_committed_end(128)  # mark as already committed
        # A block ending at 128 or earlier should be rejected
        n_l, n_h, bs, hd = 1, 1, 4, 8
        K = np.random.randn(n_l, n_h, bs, hd).astype(np.float32)
        V = np.random.randn(n_l, n_h, bs, hd).astype(np.float32)
        committed = eng.commit_block(0, 128, K, V)
        assert not committed, "Should be rejected (end_pos <= last_committed_end)"

    def test_snapshot_restore_preserves_end(self):
        """restore() sets _last_committed_end to snapshot value."""
        eng = self._make_engram()
        eng.advance_committed_end(64)
        snap = eng.snapshot()
        eng.advance_committed_end(128)
        assert eng._last_committed_end == 128
        eng.restore(snap)
        assert eng._last_committed_end == 64


class TestKVQuantUtils:
    """Standalone kv_quant.py utility function tests."""

    def test_quantize_dequantize_roundtrip(self):
        x = np.random.randn(4, 8, 32).astype(np.float32)
        q, scale = kv_quantize(x)
        x_back = kv_dequantize(q, scale)
        assert q.dtype == np.int8
        assert scale.dtype == np.float16
        rel_err = np.abs(x - x_back).max() / (np.abs(x).max() + 1e-9)
        assert rel_err < 0.01, f"Relative error too high: {rel_err:.3%}"

    def test_quantize_values_in_range(self):
        x = np.random.randn(2, 16, 64).astype(np.float32)
        q, _ = kv_quantize(x)
        assert q.min() >= -127 and q.max() <= 127

    def test_batch_quantize_shapes(self):
        KV = np.random.randn(2, 1, 4, 32, 16).astype(np.float32)
        q, scale = kv_quantize_batch(KV)
        assert q.shape == KV.shape
        assert q.dtype == np.int8
        assert scale.shape == (*KV.shape[:-1], 1)
        assert scale.dtype == np.float16

    def test_batch_roundtrip_accuracy(self):
        KV = np.random.randn(2, 1, 4, 32, 16).astype(np.float32)
        q, scale = kv_quantize_batch(KV)
        KV_back = kv_dequantize_batch(q, scale)
        rel_err = np.abs(KV - KV_back).max() / (np.abs(KV).max() + 1e-9)
        assert rel_err < 0.01

    def test_memory_estimate_savings(self):
        stats = kv_memory_bytes(
            n_layers=8, max_batch=1, n_kv_heads=8,
            window=1024, head_dim=128)
        assert stats["quant_gb"] < stats["float_gb"]
        assert stats["savings_pct"] > 40.0

    def test_zero_tensor_quantize(self):
        """Zero tensor should quantize to all-zero int8 without division issues."""
        x = np.zeros((2, 4, 16), dtype=np.float32)
        q, scale = kv_quantize(x)
        assert np.all(q == 0)
        x_back = kv_dequantize(q, scale)
        np.testing.assert_array_equal(x_back, x)

    def test_large_values_clamp(self):
        """Large values saturate at ±127 without overflow."""
        x = np.full((1, 1, 8), 1000.0, dtype=np.float32)
        q, scale = kv_quantize(x)
        assert q.max() <= 127 and q.min() >= -127