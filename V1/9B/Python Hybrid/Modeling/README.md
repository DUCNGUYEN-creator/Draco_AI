# DracoAI V1 — Modeling (Inference Engine)

**Pure-Python hybrid transformer inference engine.** NumPy is the primary
compute backend; Triton (GPU) and Numba (CPU JIT) are optional accelerators
that fall back silently to NumPy when unavailable. No required dependency
beyond NumPy.

> **Version discipline**: this package keeps the **V1** label throughout.
> No version bumps — every change is a fix or feature that stays within V1's scope.

```
modeling/
│
├── transformer.py          # Entry point: DracoTransformerV1, TransformerBlock, TransformerBridge
├── config.py                # ModelConfig + re-exported constants
├── constants.py              # Immutable compile-time constants — single source of truth
├── dtypes.py                  # COMPUTE_DTYPE policy — single source of truth, queries device.py
├── device.py                   # Hardware detection → backend selection (numpy/triton/numba)
│
├── layers/
│   ├── attention.py         # GQAttention — MLA/hybrid/Engram integration point
│   ├── attention_mla.py      # MLAProjection — KV latent compression
│   ├── hybrid_attention.py    # HybridAttentionConfig — global/local layer scheduling
│   ├── mlp.py               # ExpertFFN (SwiGLU, ternary-aware)
│   ├── moe.py                # MoELayer — routing, dispatch, load-balance, Z-loss
│   ├── norm.py                # RMSNorm wrapper → ops/tensor_ops
│   ├── embedding.py            # Embedding lookup + weight-tying
│   └── block.py               # TransformerBlock — SOLE definition, GQA + MoE pre-norm
│
├── ops/                      # Pure functions, no model state
│   ├── attention_ops.py     # rope_freqs, apply_rope, safe_softmax, causal_mask_bias
│   ├── tensor_ops.py         # rms_norm (float32-safe), mm (unified dispatch)
│   ├── activation.py          # silu, gelu
│   └── sparsity.py             # SparsityPredictor — PowerInfer-style activation skip
│
├── kernels/                  # OPTIONAL hardware kernels — tensor in → tensor out
│   ├── triton/               # GPU (fused_attention, fused_mlp, quant_matmul, ternary_matmul)
│   └── numba/                 # CPU JIT (fused_mlp, quant_matmul)
│
├── kv_cache/
│   ├── kv_cache.py           # SWA-Sink ring buffer, snapshot/restore, multi-batch, KV-Q
│   ├── engram_cache.py        # Three-tier hierarchical compressed memory
│   ├── prefix_cache.py         # LRU prompt-prefix cache
│   ├── snapshot.py              # SnapshotStack — nestable speculative rollback
│   └── kv_quant.py               # Standalone INT8 KV-Q utilities
│
├── quant/
│   ├── int4.py                # QuantizedLinear — INT8/INT4 weight-only
│   ├── ternary_linear.py       # TernaryLinear — BitNet 1.58b, addition-only forward
│   ├── quant_linear.py          # quantize_model_weights() — model-level dispatch
│   ├── gguf_loader.py             # GGUFExporter — FP16 export for llama.cpp
│   └── scales.py                   # Scale/zero-point pure functions
│
├── runtime/                  # Execution lifecycle — independent of layers
│   ├── tensor_pool.py         # TensorPool — reusable buffers, VRAM budget, secure_clear
│   ├── profiler.py             # InferenceProfiler
│   ├── health.py                # HealthMonitor — NaN/saturation/collapse/adversarial/self-correction
│   ├── precision.py               # DynamicPrecisionManager — advisory dtype + VRAM budget
│   ├── wal.py                      # WriteAheadLog — per-token journal
│   ├── scheduler.py                 # ContinuousBatchingScheduler + RequestHandle
│   ├── speculative.py                # MTPHead, SpeculativeDecoder, SpeculativeTreeDecoder
│   ├── medusa.py                       # MedusaHeads, MedusaDecoder — parallel multi-head draft
│   ├── self_correction.py               # SelfCorrectionManager — diversity re-sample
│   ├── environment.py                     # ExecutionEnvironment — thinking(CPU)↔inference(backend)
│   └── session.py                          # GenerationSession — serialisable state + secure_clear
│
├── sampling/
│   ├── sampler.py             # Sampler — topk_topp, mirostat_v2, argmax (static methods)
│   ├── mirostat.py             # Mirostat v2 core (negative-feedback mu update)
│   └── penalties.py              # repetition / frequency / presence penalties
│
├── utils/
│   ├── logging.py              # get_logger, configure_logging, log_section
│   ├── memory.py                 # RSS tracking, format_bytes
│   └── threading.py                # RWLock, atomic_counter, once
│
└── testing/                   # pytest suite — 147 tests
    ├── test_attention.py
    ├── test_quant.py
    ├── test_cache.py
    ├── test_sampling.py
    └── test_inference.py
```

---

## 1. Architecture Overview

### 1.1 One-way data-flow principle

```
runtime → layers → ops → kernels
```

No upward calls. `device.py` is the single source of truth for "what can
this machine do" — every other module (dtypes.py, kernels/, layers/block.py)
queries `device.py` instead of sniffing hardware itself. `constants.py` is
the single source of truth for every named constant; `config.py` only
re-exports, never redefines.

### 1.2 Compute dtype — single source of truth

`COMPUTE_DTYPE` is **not** a frozen module-level alias. It lives inside
`dtypes.py` as a mutable variable `_COMPUTE_DTYPE`, accessed through
`get_compute_dtype()` / `set_compute_dtype()`. `config.py` re-exports these
two functions instead of detecting the dtype itself — guaranteeing that
anyone calling `set_compute_dtype()` anywhere propagates the change
immediately to every importer, regardless of whether they import from
`dtypes` or from `config`.

```python
from modeling.dtypes import get_compute_dtype, set_compute_dtype
set_compute_dtype(np.float32, lock=True)   # lock=True blocks further overrides
```

### 1.3 Kernel dispatch — always has a NumPy fallback

Every kernel in `kernels/` follows the contract: accepts `np.ndarray`,
returns `np.ndarray`, knows nothing about `layers/`, `runtime/`, or
`kv_cache/`. Dispatch order in `ops/tensor_ops.mm()`, `layers/attention.py`,
`layers/mlp.py`:

1. **Triton** (GPU, requires CUDA + the `triton` package)
2. **Numba** (CPU JIT, requires the `numba` package)
3. **NumPy** (always available — final fallback)

Import failures on any kernel are swallowed silently (`except Exception:
pass`) — the NumPy path always runs. `device.py::detect_hardware_capability()`
only runs its probes (CUDA via CuPy/PyTorch, Triton, Numba, AVX2 — via
`/proc/cpuinfo` on Linux, via `sysctl` on macOS) **once**, lazily, cached in
a module-level singleton.

---

## 2. Model Core (`transformer.py`)

### 2.1 `DracoTransformerV1`

Standard transformer forward pass: embedding → N × `TransformerBlock` →
RMSNorm → LM head + an auxiliary MTP head (for speculative decoding). The
single most important detail in `forward()`:

```python
rope_offset = cache.get_pos(batch_idx)   # capture ONCE before any layer's update()
for block in self.blocks:
    x, aux = block.forward(..., rope_offset=rope_offset)
```

Every layer in the same forward pass must use the **same** RoPE offset. If
each layer instead called `cache.get_pos()` on its own after preceding
layers had already called `update()` (advancing `_cache_pos`), layer *i*
would receive an offset skewed by layer index rather than by actual token
position.

### 2.2 `generate()` — the token-generation loop

Supports simultaneously: prefix-cache hits (full/partial/no-logits),
Mirostat v2 or top-k/top-p, single-token speculative decoding (MTP),
speculative tree decoding, repetition penalty (including a dedicated
penalty for the reject branch), adaptive temperature, stream callback, WAL,
periodic checkpointing, health monitoring, precision manager, and per-block
Engram commits.

**Invariants that must be preserved when modifying this loop:**

- Every forward pass that writes KV must advance `cache_pos` exactly
  **once** — a double-forward or a missed write is the most common source
  of silent corruption in this subsystem.
- The speculative **accept** branch must not forward the already-accepted
  token a second time (already fixed — see §6).
- The **reject** branch must restore both the `cache` snapshot and the
  `engram_cache` snapshot, and must restore `mu` back to `mu_pre_verify`
  before resampling.
- `_try_commit_block()` only commits an Engram block when that block sits
  entirely inside the current access window; if it has already been evicted
  from the sliding window, only the pointer is advanced (via
  `advance_committed_end()`, lock-protected) without compression.

### 2.3 `TransformerBridge`

Production bridge between the NumPy backend and `llama.cpp` (via
`llama-cpp-python` + a GGUF file). `export_gguf()` uses `GGUFExporter` (see
`quant/gguf_loader.py`) and then automatically switches to
`BACKEND_LLAMA`. If `llama.generate()`'s streaming call fails (older API),
it raises a clear `RuntimeError` instead of falling back silently — the
caller needs to know to upgrade `llama-cpp-python`, rather than silently
receiving wrong output.

---

## 3. Layers

### 3.1 `GQAttention` (`layers/attention.py`)

Grouped Query Attention with `n_rep = n_heads // n_kv_heads`. Integrates
three optional features, all **backward-compatible when left unset**
(setting nothing = original GQA behavior):

| Parameter | Effect when set |
|---|---|
| `mla` (`MLAProjection`) | Compresses K/V to `latent_dim` before writing to cache, expands back after reading |
| `is_global` | Flag decided by `HybridAttentionConfig` — full-history vs sliding-window (the difference lives in cache configuration, not in a separate code path) |
| `engram` (`EngramCache`) | Blends exact (sliding-window) attention with cross-attention into compressed memory |
| `sparsity_thresh` | Prefill-only: zeroes the contribution of key positions whose max attention weight (across all heads) falls below the threshold |

Order inside `forward()`: RoPE → (MLA compress, if present) →
`cache.update()` → `cache.get()` → (MLA expand, if present) → attention
(Triton fused if available, else NumPy) → sparsity skip (if prefill +
enabled) → Engram blend (if any Engram block exists) → output projection.

### 3.2 `TransformerBlock` (`layers/block.py`)

The **SOLE definition** of `TransformerBlock` — `transformer.py` imports
from here, it does not redefine it. Pre-norm: `x + attn(rms_norm(x))`, then
`x + moe(rms_norm(x))`. `hybrid_config` and `mla` are optional constructor
parameters, both defaulting to `None` (= original behavior).

### 3.3 `MoELayer` (`layers/moe.py`)

Top-k routing with the router always kept in FP32 (RED zone — never
quantized). Two fast paths:

- **Stacked einsum** (float experts) — used when *no* expert is a
  `TernaryLinear`/`QuantizedLinear`.
- **Per-expert dispatch** — fallback when experts are ternary/quantized,
  since einsum doesn't work on non-ndarray weights.

The returned aux dict contains: `importance_loss`, `load_loss`,
`aux_total` (training signal), plus `z_loss` (router Z-loss, for
logging/monitoring only, **does not** affect the routing decision) and
`capacity_util` (Expert-Choice-style capacity tracking, monitoring-only at
inference time).

### 3.4 `HybridAttentionConfig` (`layers/hybrid_attention.py`)

Classifies each layer as **global** (full-history) or **local** (SWA).
Default: `[0, n_layers // 2, n_layers - 1]` — first, middle, and last layer
act as "relay stations". `best_engram_layer()` returns the last global
layer — the best candidate for Engram compression, since it holds the most
semantically-condensed representation.

### 3.5 `MLAProjection` (`layers/attention_mla.py`)

Compresses KV down to `latent_dim < head_dim`. Initialized via QR
decomposition (orthonormal columns) so that `W_kc @ W_ke ≈ I` on the
top-latent_dim subspace — compress→expand is nearly lossless for the
retained subspace, with loss coming only from the discarded dimensions.
**Note**: without training from scratch, the actual reconstruction error is
~30–50%; suitable for memory-constrained inference that accepts a quality
trade-off for a longer context window, not suitable when high accuracy is
required.

---

## 4. KV Cache Subsystem (`kv_cache/`)

### 4.1 `KVCache` — SWA-Sink ring buffer

Sliding-window attention with a sink token region (`SINK_TOKENS`, default
4 — never evicted). Slot index: `abs_pos < sink` → kept at its literal
position; otherwise → ring buffer `sink + (abs_pos - sink) % ring_cap`.

**Hard constructor validation**: `window <= sink` raises `ValueError`
immediately — the ring buffer needs at least 1 slot beyond the sink region.

Supports two parallel storage modes:

- **Float** (default `float16`) — stored directly.
- **KV-Q** (`use_kv_quant=True`) — INT8 values + FP16 per-vector scale.
  The scale is always computed in **float32** before being cast to
  float16, because a `1e-5` epsilon could underflow to 0 in float16 (min
  positive normal ~6e-5), causing division by zero. Measured relative
  error < 2%, saving ~43% memory versus float16.

Snapshots come in two types:

- `"full"` — copies the entire slab, O(window) memory.
- `"delta"` — records only modified (layer, slot) pairs with their values
  **before the write**, auto-escalating to full when the delta list
  exceeds a threshold.

`step()` is a documented no-op kept for API compatibility — `update()` is
the only method that advances `_cache_pos`.

**Eviction hook** (`on_evict` callback, optional): fired right before a
ring slot is overwritten, letting `EngramCache` compress a token about to
be lost before it disappears permanently.

### 4.2 `EngramCache` — three-tier compressed memory

```
Tier 0  Exact KV    → KVCache ring (most recent WINDOW tokens)
Tier 1  Engram      → every block_size tokens compressed into 1 summary vector
Tier 2  ToC         → mean of n_toc_blocks Engram vectors → 1 "chapter" vector
```

Enables O(1)-ish coarse lookup: the query first matches cosine similarity
against ToC chapter vectors, then scores the candidate list in detail via
pure-NumPy BLAS `argpartition` at O(n).

`attend()` returns a **tuple** `(eng_out, eff_alpha)` — the caller must
unpack both. `eff_alpha` is a dynamic blend weight: it decreases as
similarity to the query increases (the more confident the retrieval, the
lower the weight given to exact attention, making room for compressed
memory).

Block storage uses a **plain `list`**, not `collections.deque` — because
random-access reads (`self._blocks[i]`) happen continuously in
`_rebuild_toc_nolock()` and `retrieve_for_layer()`, while eviction
(`pop(0)`) only happens rarely, when `max_blocks` is exceeded.
`deque.pop(0)` is O(1) but random-indexing is O(n) — the opposite of what's
actually needed here.

**Concurrency**: `_last_committed_end` is only ever written inside
`_add_block()` (under lock) or via `advance_committed_end(end_pos)` (also
lock-protected, idempotent, monotonic — never moves the pointer
backward). `commit_block()` has an early-out that runs **without a lock**
to avoid wasting compression work on obviously-stale blocks, but this is
**not** the authoritative duplicate-prevention mechanism — `_add_block()`
is where the re-check happens atomically, under the lock.

### 4.3 `PrefixCache` — LRU prompt cache

`get()` always returns a **4-tuple** `(snap, prefix_len, last_logits,
engram_snap)`, even for legacy entries stored in a 3-tuple or 4-tuple
format (automatically padded with `None`). Hash key: SHA-256 of
`token_ids.tobytes() + rope_theta`.

### 4.4 `SnapshotStack` — nested rollback

`push()` / `pop()` / `commit()` (keep without restoring) /
`rollback_to(level)`. Used for beam search or multi-level speculative
rollback.

---

## 5. Quantization (`quant/`)

Three modes, applied under a **GREEN/RED zone** policy:

| Zone | Component | Policy |
|---|---|---|
| RED | Attention Q/K/V/O, Router, Embedding, LM head, Norm | Always kept FP32/FP16 |
| GREEN | MoE Expert FFN (W_g, W_u, W_d), Shared expert | Safe to ternarize/quantize |

- **`int8`/`int4`** (`QuantizedLinear`) — applied to **both** attention
  and FFN experts.
- **`ternary`** (`TernaryLinear`, BitNet 1.58b) — applied **only** to FFN
  experts (GREEN zone). Attention and the router remain float32.
  Addition-only forward: `Y = (X @ pos_mask.T - X @ neg_mask.T) * scale`,
  with no floating-point multiplication in the mathematical formula
  (though the internal BLAS gemm still multiplies — the true zero-multiply
  benefit is only achieved on the Triton GPU path).

`quantize_model_weights(model, quant, group_size=128)` is the single entry
point, dispatching on the `quant` string; raises `ValueError` for anything
other than `'int8'`, `'int4'`, `'ternary'`.

---

## 6. Speculative Decoding

Three independent mechanisms, each toggleable through `generate()`:

### 6.1 Single-token MTP (`MTPHead` + `SpeculativeDecoder`)

A lightweight auxiliary head predicts the token *following* the one just
generated. If confidence exceeds `SPEC_THRESH` (0.80), that token is
proposed as "spec_pending". At the next step, the verifier samples from
the real distribution; if it matches → **accept** (no need to re-forward —
KV was already written at draft time), if not → **reject** (restore
snapshot, resample using a distribution with
`_apply_rep_penalty_reject` applied — using `actual_cnt = cnt - 1` for the
rejected token, since `freq[]` was artificially +1'd from the proposal).

`SpeculativeDecoder` tracks an accept-rate EMA with **hysteresis**:
automatically disables speculative decoding when EMA < `disable_thresh`
(0.3), automatically re-enables when EMA > `disable_thresh × 1.5` —
preventing constant oscillation around the threshold.

### 6.2 Speculative tree (`SpeculativeTreeDecoder`)

Expands in both width (`tree_width` candidates per level) and depth
(`tree_depth` levels); each candidate is actually forwarded (not draft-only)
so its KV gets written, verified by sampling from the parent distribution,
recursively finding the longest accepted chain. Key invariants:

- **EOS acceptance does not restore the branch snapshot** — because
  `model.forward()` already wrote the EOS token's KV into the cache;
  restoring would erase that entry and cause a position desync for any
  subsequent generation.
- The winning chain is **truncated to the `max_tokens` budget BEFORE
  replay** — otherwise the cache would end up with more entries than
  tokens actually committed to `ids`, creating "phantom attention
  positions".
- Replay always uses `add_noise=False` — because `add_noise=True` makes
  the MoE router add Gumbel noise, making each forward call stochastic;
  replaying with noise would produce logits different from the original
  forward pass used to make the accept/reject decision.
- `best_logits` after truncation is always taken from the **actual last
  replayed token** (not from the chain's end point before it was cut).

### 6.3 Medusa (`MedusaHeads` + `MedusaDecoder`)

Structurally different from the two mechanisms above: Medusa forwards
**once** and then uses `n_heads` parallel linear projections to predict
tokens +1, +2, ..., +n_heads simultaneously (much faster than tree
speculative's sequential forwards, but lower quality since each head
learns independently instead of reusing the MTP hidden state).

`try_medusa()` verifies each draft sequentially: if a draft token matches a
sample from the current verifier distribution → accept, forward that token
(writing KV), update the snapshot; if it doesn't match → reject, restore
the cache to the most recent snapshot, replay the already-accepted portion
with `add_noise=False` to reconstruct a consistent `current_logits`, then
stop.

> ⚠️ **Currently under audit**: the reject branch of `try_medusa()` is
> currently being empirically verified to confirm that already-accepted
> tokens are not forwarded a second time, corrupting `cache_pos`/KV. Before
> modifying/optimizing this subsystem, run
> `pytest testing/test_inference.py::TestMedusaDecoder -v` and cross-check
> the actual number of `model.forward()` calls against the expected token
> count.

---

## 7. Runtime Lifecycle (`runtime/`)

| Module | Role |
|---|---|
| `tensor_pool.py` | Reusable buffer pool, `preallocate()` to avoid malloc on the hot path, `secure_clear()` zeroes buffers before release |
| `health.py` | NaN/Inf, saturation, expert collapse, adversarial expert (Trojan-style dominance detection), self-correction signal (EMA confidence below threshold for N consecutive steps) |
| `precision.py` | `DynamicPrecisionManager` — advisory dtype escalation/de-escalation based on overflow EMA, with a hard VRAM budget cap and independent per-expert dtype recommendations |
| `wal.py` | Binary per-token Write-Ahead Log, flushes every `WAL_FLUSH_INTERVAL` (16) tokens, TOCTOU-safe (checks `_closed` inside the lock) |
| `scheduler.py` | `ContinuousBatchingScheduler` — each request occupies one cache slot, resets the slot when the request finishes |
| `environment.py` | `ExecutionEnvironment` — separates "thinking" (sampling/penalty, always CPU) from "inference" (forward pass, uses the optimal backend) |
| `session.py` | `GenerationSession` — serializable state, `secure_clear()` zeroes `freq`/`pos`/`generated` to defend against Cold Boot Attacks, context-manager pattern |

**About `secure_clear()` in `session.py`**: there is an internal helper
`_zero_string()` that uses ctypes pointer arithmetic into CPython string
internals to attempt zeroing raw string memory. This technique is
**unsafe and not recommended for use** — CPython's string layout is not
guaranteed stable across versions, and this approach is currently not
invoked on any active code path within `secure_clear()` (which only zeroes
list/dict values via ordinary Python assignment, a far safer approach).
Treat this as dead code that needs cleanup or clear isolation, not as a
security mechanism actually being relied upon.

---

## 8. Sampling (`sampling/`)

- **Mirostat v2** (`mirostat.py`) — updates `mu` with the **minus sign**,
  matching the original Basu 2020 formulation:
  `mu ← mu - eta * (surprise - tau)` (negative feedback: higher-than-target
  surprise → mu decreases → the acceptance region narrows).
- **Top-k / top-p / min-p** (`sampler.py`) — applies temperature first,
  then top-k filtering, then top-p (nucleus) cumulative cutoff, then min-p
  relative threshold if set.
- **Penalties** (`penalties.py`) — repetition (decays by positional
  distance), frequency (linear in count), presence (flat, only requires
  ≥1 appearance). All return a **copy**, never mutating the input.

---

## 9. Running Tests

```bash
pytest modeling/testing/ -v
```

147 test cases, organized by subsystem (`test_attention.py`,
`test_cache.py`, `test_quant.py`, `test_sampling.py`,
`test_inference.py`). `test_inference.py` uses a tiny config
(`d_model=64, n_layers=2, n_experts=4, vocab_size=256`) to keep the entire
suite running under 5 seconds on CPU, while still covering every major
code path: ternary, MLA, hybrid attention, Medusa, self-correction, KV-Q.

A clean baseline (all passing) is a mandatory prerequisite before any
bug-fix pass — don't modify code while the baseline has unrelated failing
tests.

---

## 10. Architectural Constraints to Respect When Extending This Package

1. **No upward imports** — `runtime/` must not be imported from
   `transformer.py`; `layers/` must not import from `runtime/` at module
   level (only via `TYPE_CHECKING` or lazy imports inside functions).
2. **`__init__.py` must match the actual package structure** — if a new
   file is added to a subpackage, its corresponding `__init__.py` must be
   updated, and so must the root `modeling/__init__.py` if that API needs
   to be public.
3. **Never create a new frozen `COMPUTE_DTYPE` alias** in any module —
   always call `get_compute_dtype()`.
4. **Every forward pass that writes KV must advance `cache_pos` exactly
   once** — this is the most fragile invariant across the entire
   speculative decoding subsystem.
5. **Changes to `kv_cache/`, `speculative.py`, `medusa.py`,
   `transformer.py` (the MTP path)** require empirical verification before
   modification if that subsystem has a history of prior fixes — high
   regression risk, especially on paths related to `cache_pos`/KV desync.

---

GPL v3 Copyright © 2026 DUCNGUYEN-creator