# DracoAI V1 — modeling/testing/test_cache.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Unit tests for KVCache, PrefixCache, SnapshotStack.

FIXES (this revision):
  ✅ FIX-PREFIX-UNPACK : PrefixCache.get() returns a 4-tuple
     (snap, plen, last_logits, engram_snap).  All test unpack sites updated
     from the old 3-value form ``_, plen, ll = result`` to the correct
     4-value form ``_, plen, ll, _ = result``.
"""
import numpy as np
import pytest
from ..kv_cache.kv_cache     import KVCache
from ..kv_cache.prefix_cache import PrefixCache
from ..kv_cache.snapshot     import SnapshotStack


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
        """get_pos() must be used instead of _cache_pos directly."""
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
        """Vectorised fast path with seq > 1 should give same result as seq=1 loop."""
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

        # ✅ FIX-PREFIX-UNPACK: get() returns 4-tuple, not 3-tuple
        snap, plen, ll, engram_snap = result
        assert plen == 5
        assert ll is not None
        assert engram_snap is None   # not stored, so should be None

    def test_put_get_with_engram_snap(self):
        """Engram snapshot is stored and retrieved correctly."""
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
        """Old 4-tuple entries (snap, plen, ts, last_logits) are read safely."""
        import time
        pc = PrefixCache(max_entries=4)
        ids = [7, 8, 9]
        h   = pc._hash(ids)
        # Manually inject a legacy 4-tuple (no engram_snap field)
        snap = self._make_snap()
        snap["_cache_pos"] = 3
        pc._store[h] = (snap, 3, time.perf_counter(), np.zeros(10))

        result = pc.get(ids)
        assert result is not None
        _, plen, ll, engram_snap = result
        assert plen == 3
        assert engram_snap is None   # padded with None


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