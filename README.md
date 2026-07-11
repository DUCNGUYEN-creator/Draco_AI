<p align="center">
  <img src="assets/logo.png" alt="Draco Logo" width="200">
</p>

---
# Draco AI V1

**Draco Studio — DUCNGUYEN-creator — GPL v3 — Built in Vietnam**

---

Gmail: zintat19791@gmail.com

Discord: https://discord.gg/dJXMbexdTh (Draco Studio)

---

## Project Status

Draco AI is a **solo, work-in-progress project**: one developer building a transformer inference engine and a cognitive/reasoning layer from scratch, without a team or external funding. It is **not** a pretrained model — there are no released weights. It is a framework and runtime intended for restructuring existing dense models into a Mixture-of-Experts layout and running them efficiently on consumer-grade hardware.

Please read this section before the rest of the README, since it sets expectations for everything below:

- **Nothing here is production-hardened.** Several subsystems described in detail further down are working prototypes rather than finished, battle-tested components. In particular:
  - The **Hallucination Subsystem** (§1.3) currently verifies claims with heuristic scoring rather than deep semantic verification; that upgrade is future work, not something already delivered.
  - **Multi-Agent Debate** (`reasoning/debate/`) is currently debate between multiple experts inside the same underlying MoE model (role-conditioned routing across experts), not independent debate between separate models. "Council Debate (8 experts)" refers to this — 8 experts of one model, not 8 separate models.
  - Some components referenced in design docs and internal audits are not yet wired into the runtime, and a couple of classes mentioned in planning notes (e.g. a spatial-reasoning solver, a multi-turn intent tracker) do not exist in the codebase yet.
  - Thread-safety and encapsulation in a few runtime modules are known weak points currently being audited (see the `runtime/` notes in §2.8).
- **Any performance or efficiency claims in this repo are internal/self-reported**, not independently reproduced third-party evaluations. Treat all such framing here as a design target, not a verified result, until backed by a reproducible benchmark script.
- **The module/line counts below are approximate**, taken from an internal directory sweep at one point during the refactor; they will drift as the codebase changes, and are counted slightly differently in different tables (see the note under the overview table).
- This project does not claim to be the first or only implementation of any particular technique (e.g. ternary quantization, tiered KV memory). Any comparison in this README to other named projects reflects this author's own reading of those projects' public code/docs at the time of writing — one contributor's opinion, not an audited or exhaustive comparison.

**Draco AI is unfinished and could use help.** Issues, pull requests, and critical code review are genuinely welcome — see [Contributing](#contributing) below.

## Table of Contents

- [Project Status](#project-status)
- [Prerequisites & Installation](#prerequisites--installation)
- [Part 1 — `thinking_engine/` (Cognitive Layer)](#part-1--thinking_engine-cognitive-layer)
- [Part 2 — `modeling/` (Inference Engine)](#part-2--modeling-inference-engine)
- [Part 3 — Bridge Between the Two Packages](#part-3--bridge-between-the-two-packages)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

DracoAI V1 consists of two independent packages that can be wired together through a single bridge point:

| Package | Role | Approx. size |
|---|---|---|
| `thinking_engine/` | Cognitive layer — perception, reasoning, reflection, hallucination assessment | ~277 modules / ~13,212 lines / 47 directories |
| `modeling/` | Inference engine — pure-Python/NumPy transformer, KV cache, quantization, speculative decoding | Pure-Python + NumPy backend |

*Note on the counts above: different tables in this README report 277 vs. 278 for `thinking_engine/`. That is most likely a modules-vs-files discrepancy (e.g. whether `__init__.py` files are included), not two independently verified figures — treat both as "~277" until an exact, reproducible count (e.g. `find thinking_engine -name '*.py' | wc -l`) is recorded in a future update.*

The two packages are fully decoupled on dependencies: `thinking_engine/` does not import `numpy` or `modeling.transformer` anywhere except at a single point (`interfaces/llm.py`), so each package can be installed and tested independently of the other.

```text
                    ┌─────────────────────┐        ┌──────────────────────┐
                    │   thinking_engine/   │        │      modeling/        │
                    │  (Cognition layer)   │◄──────►│  (Inference engine)   │
                    │                      │ single  │                      │
                    │  Perception→...→     │ bridge  │  DracoTransformerV1   │
                    │  Reflection/Halluc.  │ point   │  + TransformerBridge  │
                    └─────────────────────┘ (llm.py) └──────────────────────┘
```

## Prerequisites & Installation

- Python 3.10+
- `modeling/` requires **NumPy** only. `triton` (GPU) and `numba` (CPU JIT) are optional accelerators that fall back silently to NumPy when not installed.
- `thinking_engine/` requires **no dependency beyond the Python standard library** (see §3.3).

```bash
git clone https://github.com/DUCNGUYEN-creator/Draco_AI.git
cd Draco_AI
pip install numpy          # required for modeling/
# optional accelerators:
pip install triton numba
```

There is no packaged release yet (no PyPI package, no pretrained weights to download) — this is source-only at this stage.

---

## Part 1 — `thinking_engine/` (Cognitive Layer)

Package split from `engine_v1.py` (~3,775-line monolith) into a modular architecture following the Hybrid Architecture design:

```text
Engine = Infrastructure + Cognition + Verification
```

### 1.1 Three-Layer Architecture

```text
                         ThinkingEngine
                         │
       ┌─────────────────┼──────────────────┐
       │                 │                  │
       ▼                 ▼                  ▼
 Infrastructure      Cognition         Verification

 perception/         reasoning/        reflection/
 memory/             └─ search/         ├─ self_reflection.py
 knowledge/          └─ thinking/       ├─ consistency.py
 routing/            └─ cognitive/      ├─ confidence.py
 planning/           └─ debate/         ├─ confidence_calibrator.py
 tools/              └─ execution/      ├─ answer_rewriter.py
                                        ├─ critic.py
                                        └─ hallucination/    ← Deep subsystem
```

### 1.2 12-Stage Pipeline (Perception → Output)

| # | Stage | Module | Responsibility |
|---|-------|--------|----------|
| 1 | Perception | `perception/` | Sanitize, rewrite, intent, entity, metaphor, difficulty |
| 2 | Memory Retrieval | `memory/` | Rerank, compress, episodic recall |
| 3 | Knowledge Retrieval | `knowledge/` | KG path, RAG, analogy, counterfactual, abduction |
| 4 | Planning | `planning/` | Tool injection, goal/plan decomposition |
| 5 | Reasoning | `reasoning/` | ToT, Council Debate, Self-Consistency, MCTS |
| 6 | Tool Execution | `tools/` | Safe AST eval, sandbox, tool registry |
| 7 | Reasoning Loop | `reasoning/thinking/` | Chain-of-thought verification, recursive reflection |
| 8 | Verifier Layer | `reasoning/thinking/chain_verifier.py` | CoT soundness check |
| 9 | Hallucination Assessment | `reflection/hallucination/` | Deep 6-stage pipeline |
| 10 | Reflection | `reflection/critic.py` | Recursive critique, post-generation check |
| 11 | Answer Rewriter | `reflection/answer_rewriter.py` | Rewrite trigger if risk is high |
| 12 | Output | `perception/prompt/compiler.py` | Compile [PLAN][THOUGHT][FINAL ANSWER] |

### 1.3 Hallucination Subsystem (Deep Verification)


```text
Evidence → Verification → Calibration → Correlation → Fusion → Risk → Report
```

**9 Verifiers:**

| Verifier | Checks | Confidence |
|----------|----------|------------|
| retrieval | Claim matches retrieved evidence | High when trust_score is high |
| contradiction | Claim contradicts evidence (negation-flip, antonym) | Medium |
| consistency | Claim agrees across multiple independent reasoning paths | Increases with n_paths |
| numerical | Arithmetic expression is correct (deterministic, via SafeASTEvaluator) | 0.9, very high |
| symbolic | Logical proposition (tautology/contradiction) is valid | 0.85 when applicable |
| citation | `[hexid]` citation exists in CitationTracker | 0.9 vs registry |
| planner | Claim matches committed subgoals/plan | 0.4, soft signal |
| tool | Claim matches actual tool output (most trusted source) | 0.85 |
| reasoning | Claim traceable from selected reasoning trace | 0.55 |

**Calibration methods:** `platt` · `isotonic` · `beta` · `temperature` · `histogram` + `ensemble`

**5 Fusion methods:**

| Method | Characteristic | Default |
|---|---|---|
| noisy_or | One strong signal dominates all others | ✓ |
| weighted | Weighted average, easy to interpret | N/A |
| bayesian | Sequential update, distinguishes positive/negative signals | N/A |
| dempster_shafer | Handles abstain (mass-on-uncertainty) precisely | N/A |
| logistic | Weighted logit sum, does not saturate extreme probabilities | N/A |

**3 Strategy tiers:**

| Strategy | Verifiers | Use case |
|---|---|---|
| fast | 2 (retrieval + contradiction) | INTENT_CHAT, low-latency |
| balanced | 6 | DEFAULT for every request |
| paranoid | 9 (all, auto-expands as more are registered) | High-stakes |

### 1.4 Directory Structure

```text
thinking_engine/
├── __init__.py           ← ThinkingEngine, EngineConfig
├── engine.py             ← ThinkingEngine (top-level facade)
├── pipeline.py           ← Global Pipeline 12-stage orchestrator
├── state.py              ← ThinkingState (carrier throughout the pipeline)
├── config.py             ← EngineConfig (all tuning knobs)
├── constants.py          ← expert indices, intent types, stage names
├── exceptions.py         ← Exception hierarchy
│
├── interfaces/           ← Protocol types (LLMBridge, Verifier, ...)
│
├── perception/
│   ├── prompt/           ← Sanitizer, ContextualRewriter, Compiler, Parser
│   ├── language/         ← LanguageDetector, EntityExtractor, Metaphor, Sentiment
│   └── understanding/    ← IntentDetector, Difficulty, DualProcess, Task, Ambiguity
│
├── memory/               ← Working/Episodic/Semantic/LongTerm + Retrieval + Forgetting
├── knowledge/            ← KG, TemporalKG, RAG, FactChecker, BayesianUpdater, Analogy
├── routing/              ← ExpertRouter, SelfEvolvingRouter, LoadBalancer, BudgetRouter
│
├── reasoning/
│   ├── search/           ← BFS/DFS/A*/Beam/MCTS/IDA*/Bidirectional
│   ├── thinking/         ← ToT, GoT, SelfConsistency, CoTVerifier, RecursiveReflection, ...
│   ├── cognitive/        ← Abduction, Induction, Deduction, Analogy, Counterfactual, ...
│   ├── debate/           ← MultiAgentDebate (Council), Expert, Arbitration, Voting
│   └── execution/        ← ReasoningController (ThreadPoolExecutor concurrent)
│
├── planning/             ← Planner, GoalDecomposer, PlanDecomposer, TaskGraph, Scheduler
├── tools/                ← Registry, Parser, Executor, SafeASTEvaluator, Validator
│
├── reflection/
│   ├── self_reflection.py
│   ├── consistency.py
│   ├── confidence.py
│   ├── confidence_calibrator.py  ← Always-on lightweight Platt
│   ├── answer_rewriter.py
│   ├── critic.py                 ← Orchestrator: recursive_critique + post_generation_check
│   └── hallucination/            ← Deep Verification subsystem
│       ├── assessor.py           ← PUBLIC ENTRY POINT (sole one)
│       ├── models/               ← Evidence, VerificationResult, FusionResult, Report, ...
│       ├── verifiers/            ← 9 specialized verifiers
│       ├── analyzers/            ← Taxonomy, Severity, Agreement, Outlier, Coverage, ...
│       ├── calibration/          ← Platt/Isotonic/Beta/Temperature/Histogram/Ensemble
│       ├── correlation/          ← Similarity, Dedup, ConnectedComponents, Reducer
│       ├── fusion/               ← NoisyOr/Weighted/Bayesian/DempsterShafer/Logistic
│       ├── metrics/              ← AUC, Brier, ECE, Drift, Reliability, Entropy, ...
│       ├── pipeline/             ← 6-stage Evidence→Verification→...→Report
│       ├── registry/             ← Plugin registry (no need to edit existing code to add new)
│       ├── factory/              ← Factory with instance caching
│       ├── strategy/             ← fast/balanced/paranoid/custom
│       ├── cache/                ← LRU+TTL: evidence, verifier, calibration, stats
│       ├── benchmarks/           ← Internal AUC/ECE/monotonicity benchmarks
│       ├── docs/                 ← architecture.md, api.md, verifier.md, ...
│       └── tests/                ← 13 test cases, 100% pass
│
├── safety/               ← EthicalFilter, PromptGuard, InjectionDetector, ActiveLearning
├── learning/             ← FeedbackCollector, OnlineLearner, RouterUpdater, ExperienceBuffer
├── execution/            ← InferenceRequest, ResponseBuilder, ResponseFormatter, EngineOutput
└── utils/                ← Hash, Graph, Math, Probability, Timer, Thread, Serialization
```

### 1.5 Quick Start

```python
from thinking_engine import ThinkingEngine

# Initialize (stub bridge — no real model needed to run)
engine = ThinkingEngine()

# Phase 1: compile prompt
out = engine.process(
    "Viết hàm Python tính giai thừa",
    history=[],
    think_mode=False,           # True = activates Council Debate (8 experts)
)
print(out.intent)               # {'intent': 'code', 'lang': 'vi', ...}
print(out.expert_boost)         # {0: 0.04, 1: 0.48, 2: 0.27, ...}
print(len(out.messages))        # 3 (system + plan + user)

# Phase 2: after the LLM generates a response, assess Hallucination + Reflection
final = engine.finish(
    "Viết hàm Python tính giai thừa",
    generated_text="def factorial(n): return 1 if n<=1 else n*factorial(n-1)",
)
print(final.hallucination_report['risk_level'])   # 'none' / 'low' / 'medium' / 'high' / 'critical'
print(final.hallucination_report['risk_score'])   # float [0.0, 1.0]
print(final.hallucination_report['top_issues'])   # List[str]

# Feedback loop (improves router + calibration over time)
engine.submit_feedback("Viết hàm Python...", "def factorial...", rating=0.95)
```

### 1.6 Using the Hallucination Assessor Standalone

```python
from thinking_engine.reflection.hallucination import Assessor
from thinking_engine.config import HallucinationConfig

assessor = Assessor(config=HallucinationConfig(strategy="paranoid"))
report = assessor.assess(
    answer="2 + 2 = 5.",
    context={"tool_results": [{"tool": "calculator", "output": "4", "ok": True}]},
)
print(report.risk_level)    # RiskLevel.CRITICAL
print(report.top_issues)    # ["'2 + 2 = 5' but actual result is 4"]
```

### 1.7 Registering a New Verifier (Plugin Architecture)

```python
from thinking_engine.reflection.hallucination.registry import VerifierRegistry

class MyEmbeddingVerifier:
    name = "embedding"
    def verify(self, claim, evidence, context):
        # ... embedding similarity check ...
        return {"verifier": self.name, "score": 0.85, "confidence": 0.9, "issues": []}

registry = VerifierRegistry()
registry.register("embedding", MyEmbeddingVerifier)
# No need to touch assessor.py or strategy/*.py
```

### 1.8 Comparison with `engine_v1.py`

| Aspect | engine_v1.py | thinking_engine/ |
|---|---|---|
| Size | ~3,775 lines / 1 file | ~13,212 lines / ~277 files (see note on counts above) |
| Architecture | Monolith (ThinkingEngineV1) | 3-layer: Infrastructure/Cognition/Verification |
| Hallucination layer | SelfReflection.critique() inline | 6-stage deep pipeline, 9 verifiers |
| Fusion | None | 5 methods (noisy_or, bayesian, ...) |
| Calibration | 1 online Platt | 5 methods + ensemble |
| Search algorithms | BFS, DFS, A*, MCTS | + Beam, IDA*, Bidirectional |
| Tests (hallucination subsystem) | 1 inline self-test block | 13 structured tests, 100% pass |
| Plugin | None | Registry + Factory pattern |
| Dependency separation | All in 1 class | Verification layer does not depend on Cognition |
| Import sweep | N/A | 277/277 modules, 0 errors |

### 1.9 Running Tests

```bash
pytest thinking_engine/reflection/hallucination/tests/ -v
```

The hallucination subsystem currently has **13 test cases, all passing** as of this writing — these cover the 9 verifiers, 5 fusion methods, and 5 calibration methods described in §1.3. This is the only part of `thinking_engine/` with a dedicated, structured test suite so far; most other subpackages (`reasoning/`, `planning/`, `memory/`, etc.) do not yet have equivalent automated coverage. Expanding test coverage across the rest of the package is tracked in the [Roadmap](#roadmap).

---

## Part 2 — `modeling/` (Inference Engine)

**Pure-Python hybrid transformer inference engine.** NumPy is the primary compute backend; Triton (GPU) and Numba (CPU JIT) are optional accelerators that fall back silently to NumPy when unavailable. No required dependency beyond NumPy.

> **Version discipline**: this package keeps the **V1** label throughout — no version bumps, every change is a fix or feature that stays within V1's scope.

### 2.1 Directory Structure

```text
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
└── testing/                   # pytest suite — 5 test files
    ├── test_attention.py
    ├── test_quant.py
    ├── test_cache.py
    ├── test_sampling.py
    └── test_inference.py
```

### 2.2 Architecture Overview

**One-way data-flow:** `runtime → layers → ops → kernels`. No upward calls. `device.py` is the single source of truth for "what can this machine do" — every other module (`dtypes.py`, `kernels/`, `layers/block.py`) queries `device.py` instead of sniffing hardware itself. `constants.py` is the single source of truth for every named constant; `config.py` only re-exports, never redefines.

**Compute dtype:** `COMPUTE_DTYPE` is **not** a frozen module-level alias. It lives inside `dtypes.py` as a mutable variable `_COMPUTE_DTYPE`, accessed through `get_compute_dtype()` / `set_compute_dtype()`. `config.py` re-exports these two functions instead of detecting the dtype itself — guaranteeing that calling `set_compute_dtype()` anywhere propagates the change immediately to every importer.

```python
from modeling.dtypes import get_compute_dtype, set_compute_dtype
set_compute_dtype(np.float32, lock=True)   # lock=True blocks further overrides
```

**Kernel dispatch order** (in `ops/tensor_ops.mm()`, `layers/attention.py`, `layers/mlp.py`):
1. **Triton** (GPU, requires CUDA + the `triton` package)
2. **Numba** (CPU JIT, requires the `numba` package)
3. **NumPy** (always available — final fallback)

Import failures on any kernel are swallowed silently (`except Exception: pass`) — the NumPy path always runs. `device.py::detect_hardware_capability()` only runs its probes (CUDA via CuPy/PyTorch, Triton, Numba, AVX2 — via `/proc/cpuinfo` on Linux, via `sysctl` on macOS) **once**, lazily, cached in a module-level singleton.

### 2.3 Model Core (`transformer.py`)

**`DracoTransformerV1`**: standard forward pass: embedding → N × `TransformerBlock` → RMSNorm → LM head + an auxiliary MTP head (for speculative decoding). The single most important detail in `forward()`:

```python
rope_offset = cache.get_pos(batch_idx)   # capture ONCE before any layer's update()
for block in self.blocks:
    x, aux = block.forward(..., rope_offset=rope_offset)
```

Every layer in the same forward pass must use the **same** RoPE offset. If each layer instead called `cache.get_pos()` on its own after preceding layers had already called `update()` (advancing `_cache_pos`), layer *i* would receive an offset skewed by layer index rather than by actual token position.

**`generate()`** supports simultaneously: prefix-cache hits (full/partial/no-logits), Mirostat v2 or top-k/top-p, single-token speculative decoding (MTP), speculative tree decoding, repetition penalty (including a dedicated penalty for the reject branch), adaptive temperature, stream callback, WAL, periodic checkpointing, health monitoring, precision manager, and per-block Engram commits.

**Invariants that must be preserved when modifying this loop:**
- Every forward pass that writes KV must advance `cache_pos` exactly **once** — a double-forward or a missed write is the most common source of silent corruption in this subsystem.
- The speculative **accept** branch must not forward the already-accepted token a second time (already fixed — see §Speculative Decoding).
- The **reject** branch must restore both the `cache` snapshot and the `engram_cache` snapshot, and must restore `mu` back to `mu_pre_verify` before resampling.
- `_try_commit_block()` only commits an Engram block when that block sits entirely inside the current access window; if it has already been evicted from the sliding window, only the pointer is advanced (via `advance_committed_end()`, lock-protected) without compression.

**`TransformerBridge`**: production bridge between the NumPy backend and `llama.cpp` (via `llama-cpp-python` + a GGUF file). `export_gguf()` uses `GGUFExporter` (see `quant/gguf_loader.py`) and then automatically switches to `BACKEND_LLAMA`. If `llama.generate()`'s streaming call fails (older API), it raises a clear `RuntimeError` instead of falling back silently — the caller needs to know to upgrade `llama-cpp-python`, rather than silently receiving wrong output.

### 2.4 Layers

**`GQAttention`** (`layers/attention.py`): Grouped Query Attention with `n_rep = n_heads // n_kv_heads`. Integrates three optional features, all **backward-compatible when left unset** (setting nothing = original GQA behavior):

| Parameter | Effect when set |
|---|---|
| `mla` (`MLAProjection`) | Compresses K/V to `latent_dim` before writing to cache, expands back after reading |
| `is_global` | Flag decided by `HybridAttentionConfig` — full-history vs sliding-window (the difference lives in cache configuration, not in a separate code path) |
| `engram` (`EngramCache`) | Blends exact (sliding-window) attention with cross-attention into compressed memory |
| `sparsity_thresh` | Prefill-only: zeroes the contribution of key positions whose max attention weight (across all heads) falls below the threshold |

Order inside `forward()`: RoPE → (MLA compress, if present) → `cache.update()` → `cache.get()` → (MLA expand, if present) → attention (Triton fused if available, else NumPy) → sparsity skip (if prefill + enabled) → Engram blend (if any Engram block exists) → output projection.

**`TransformerBlock`** (`layers/block.py`): the **SOLE definition** of `TransformerBlock` — `transformer.py` imports from here, it does not redefine it. Pre-norm: `x + attn(rms_norm(x))`, then `x + moe(rms_norm(x))`. `hybrid_config` and `mla` are optional constructor parameters, both defaulting to `None` (= original behavior).

**`MoELayer`** (`layers/moe.py`): top-k routing with the router always kept in FP32 (RED zone — never quantized). Two fast paths:
- **Stacked einsum** (float experts) — used when *no* expert is a `TernaryLinear`/`QuantizedLinear`.
- **Per-expert dispatch** — fallback when experts are ternary/quantized, since einsum doesn't work on non-ndarray weights.

The returned aux dict contains: `importance_loss`, `load_loss`, `aux_total` (training signal), plus `z_loss` (router Z-loss, for logging/monitoring only, **does not** affect the routing decision) and `capacity_util` (Expert-Choice-style capacity tracking, monitoring-only at inference time).

**`HybridAttentionConfig`** (`layers/hybrid_attention.py`): classifies each layer as **global** (full-history) or **local** (SWA). Default: `[0, n_layers // 2, n_layers - 1]` — first, middle, and last layer act as "relay stations". `best_engram_layer()` returns the last global layer — the best candidate for Engram compression, since it holds the most semantically-condensed representation.

**`MLAProjection`** (`layers/attention_mla.py`): compresses KV down to `latent_dim < head_dim`. Initialized via QR decomposition (orthonormal columns) so that `W_kc @ W_ke ≈ I` on the top-latent_dim subspace — compress→expand is nearly lossless for the retained subspace, with loss coming only from the discarded dimensions. **Note**: without training from scratch, the actual reconstruction error is ~30–50%; suitable for memory-constrained inference that accepts a quality trade-off for a longer context window, not suitable when high accuracy is required.

### 2.5 KV Cache Subsystem (`kv_cache/`)

**`KVCache`** — SWA-Sink ring buffer: sliding-window attention with a sink token region (`SINK_TOKENS`, default 4 — never evicted). Slot index: `abs_pos < sink` → kept at its literal position; otherwise → ring buffer `sink + (abs_pos - sink) % ring_cap`.

**Hard constructor validation**: `window <= sink` raises `ValueError` immediately — the ring buffer needs at least 1 slot beyond the sink region.

Supports two parallel storage modes:
- **Float** (default `float16`) — stored directly.
- **KV-Q** (`use_kv_quant=True`) — INT8 values + FP16 per-vector scale. The scale is always computed in **float32** before being cast to float16, because a `1e-5` epsilon could underflow to 0 in float16 (min positive normal ~6e-5), causing division by zero. Measured relative error < 2%, saving ~43% memory versus float16.

Snapshots come in two types:
- `"full"` — copies the entire slab, O(window) memory.
- `"delta"` — records only modified (layer, slot) pairs with their values **before the write**, auto-escalating to full when the delta list exceeds a threshold.

`step()` is a documented no-op kept for API compatibility — `update()` is the only method that advances `_cache_pos`.

**Eviction hook** (`on_evict` callback, optional): fired right before a ring slot is overwritten, letting `EngramCache` compress a token about to be lost before it disappears permanently.

**`EngramCache`** — three-tier compressed memory:

```text
Tier 0  Exact KV    → KVCache ring (most recent WINDOW tokens)
Tier 1  Engram      → every block_size tokens compressed into 1 summary vector
Tier 2  ToC         → mean of n_toc_blocks Engram vectors → 1 "chapter" vector
```

Enables O(1)-ish coarse lookup: the query first matches cosine similarity against ToC chapter vectors, then scores the candidate list in detail via pure-NumPy BLAS `argpartition` at O(n).

`attend()` returns a **tuple** `(eng_out, eff_alpha)` — the caller must unpack both. `eff_alpha` is a dynamic blend weight: it decreases as similarity to the query increases (the more confident the retrieval, the lower the weight given to exact attention, making room for compressed memory).

Block storage uses a **plain `list`**, not `collections.deque` — because random-access reads (`self._blocks[i]`) happen continuously in `_rebuild_toc_nolock()` and `retrieve_for_layer()`, while eviction (`pop(0)`) only happens rarely, when `max_blocks` is exceeded. `deque.pop(0)` is O(1) but random-indexing is O(n) — the opposite of what's actually needed here.

**Concurrency**: `_last_committed_end` is only ever written inside `_add_block()` (under lock) or via `advance_committed_end(end_pos)` (also lock-protected, idempotent, monotonic — never moves the pointer backward). `commit_block()` has an early-out that runs **without a lock** to avoid wasting compression work on obviously-stale blocks, but this is **not** the authoritative duplicate-prevention mechanism — `_add_block()` is where the re-check happens atomically, under the lock.

**`PrefixCache`** — LRU prompt cache: `get()` always returns a **4-tuple** `(snap, prefix_len, last_logits, engram_snap)`, even for legacy entries stored in a 3-tuple or 4-tuple format (automatically padded with `None`). Hash key: SHA-256 of `token_ids.tobytes() + rope_theta`.

**`SnapshotStack`** — nested rollback: `push()` / `pop()` / `commit()` (keep without restoring) / `rollback_to(level)`. Used for beam search or multi-level speculative rollback.

### 2.6 Quantization (`quant/`)

Three modes, applied under a **GREEN/RED zone** policy:

| Zone | Component | Policy |
|---|---|---|
| RED | Attention Q/K/V/O, Router, Embedding, LM head, Norm | Always kept FP32/FP16 |
| GREEN | MoE Expert FFN (W_g, W_u, W_d), Shared expert | Safe to ternarize/quantize |

- **`int8`/`int4`** (`QuantizedLinear`) — applied to **both** attention and FFN experts.
- **`ternary`** (`TernaryLinear`, BitNet 1.58b) — applied **only** to FFN experts (GREEN zone). Attention and the router remain float32. Addition-only forward: `Y = (X @ pos_mask.T - X @ neg_mask.T) * scale`, with no floating-point multiplication in the mathematical formula (though the internal BLAS gemm still multiplies — the true zero-multiply benefit is only achieved on the Triton GPU path).

`quantize_model_weights(model, quant, group_size=128)` is the single entry point, dispatching on the `quant` string; raises `ValueError` for anything other than `'int8'`, `'int4'`, `'ternary'`.

### 2.7 Speculative Decoding

Three independent mechanisms, each toggleable through `generate()`:

**6.1 Single-token MTP** (`MTPHead` + `SpeculativeDecoder`): a lightweight auxiliary head predicts the token *following* the one just generated. If confidence exceeds `SPEC_THRESH` (0.80), that token is proposed as "spec_pending". At the next step, the verifier samples from the real distribution; if it matches → **accept** (no need to re-forward — KV was already written at draft time), if not → **reject** (restore snapshot, resample using a distribution with `_apply_rep_penalty_reject` applied — using `actual_cnt = cnt - 1` for the rejected token, since `freq[]` was artificially +1'd from the proposal).

`SpeculativeDecoder` tracks an accept-rate EMA with **hysteresis**: automatically disables speculative decoding when EMA < `disable_thresh` (0.3), automatically re-enables when EMA > `disable_thresh × 1.5` — preventing constant oscillation around the threshold.

**6.2 Speculative tree** (`SpeculativeTreeDecoder`): expands in both width (`tree_width` candidates per level) and depth (`tree_depth` levels); each candidate is actually forwarded (not draft-only) so its KV gets written, verified by sampling from the parent distribution, recursively finding the longest accepted chain. Key invariants:
- **EOS acceptance does not restore the branch snapshot** — because `model.forward()` already wrote the EOS token's KV into the cache; restoring would erase that entry and cause a position desync for any subsequent generation.
- The winning chain is **truncated to the `max_tokens` budget BEFORE replay** — otherwise the cache would end up with more entries than tokens actually committed to `ids`, creating "phantom attention positions".
- Replay always uses `add_noise=False` — because `add_noise=True` makes the MoE router add Gumbel noise, making each forward call stochastic; replaying with noise would produce logits different from the original forward pass used to make the accept/reject decision.
- `best_logits` after truncation is always taken from the **actual last replayed token** (not from the chain's end point before it was cut).

**6.3 Medusa** (`MedusaHeads` + `MedusaDecoder`): structurally different from the two mechanisms above: Medusa forwards **once** and then uses `n_heads` parallel linear projections to predict tokens +1, +2, ..., +n_heads simultaneously (much faster than tree speculative's sequential forwards, but lower quality since each head learns independently instead of reusing the MTP hidden state).

`try_medusa()` verifies each draft sequentially: if a draft token matches a sample from the current verifier distribution → accept, forward that token (writing KV), update the snapshot; if it doesn't match → reject, restore the cache to the most recent snapshot, replay the already-accepted portion with `add_noise=False` to reconstruct a consistent `current_logits`, then stop.

> ⚠️ **Currently under audit**: the reject branch of `try_medusa()` is currently being empirically verified to confirm that already-accepted tokens are not forwarded a second time, corrupting `cache_pos`/KV. Before modifying/optimizing this subsystem, run `pytest testing/test_inference.py::TestMedusaDecoder -v` and cross-check the actual number of `model.forward()` calls against the expected token count.

### 2.8 Runtime Lifecycle (`runtime/`)

| Module | Role |
|---|---|
| `tensor_pool.py` | Reusable buffer pool, `preallocate()` to avoid malloc on the hot path, `secure_clear()` zeroes buffers before release |
| `health.py` | NaN/Inf, saturation, expert collapse, adversarial expert (Trojan-style dominance detection), self-correction signal (EMA confidence below threshold for N consecutive steps) |
| `precision.py` | `DynamicPrecisionManager` — advisory dtype escalation/de-escalation based on overflow EMA, with a hard VRAM budget cap and independent per-expert dtype recommendations |
| `wal.py` | Binary per-token Write-Ahead Log, flushes every `WAL_FLUSH_INTERVAL` (16) tokens, TOCTOU-safe (checks `_closed` inside the lock) |
| `scheduler.py` | `ContinuousBatchingScheduler` — each request occupies one cache slot, resets the slot when the request finishes |
| `environment.py` | `ExecutionEnvironment` — separates "thinking" (sampling/penalty, always CPU) from "inference" (forward pass, uses the optimal backend) |
| `session.py` | `GenerationSession` — serializable state, `secure_clear()` zeroes `freq`/`pos`/`generated` to defend against Cold Boot Attacks, context-manager pattern |

**About `secure_clear()` in `session.py`**: there is an internal helper `_zero_string()` that uses ctypes pointer arithmetic into CPython string internals to attempt zeroing raw string memory. This technique is **unsafe and not recommended for use** — CPython's string layout is not guaranteed stable across versions, and this approach is currently not invoked on any active code path within `secure_clear()` (which only zeroes list/dict values via ordinary Python assignment, a far safer approach). Treat this as dead code that needs cleanup or clear isolation, not as a security mechanism actually being relied upon.

### 2.9 Sampling (`sampling/`)

- **Mirostat v2** (`mirostat.py`) — updates `mu` with the **minus sign**, matching the original Basu 2020 formulation: `mu ← mu - eta * (surprise - tau)` (negative feedback: higher-than-target surprise → mu decreases → the acceptance region narrows).
- **Top-k / top-p / min-p** (`sampler.py`) — applies temperature first, then top-k filtering, then top-p (nucleus) cumulative cutoff, then min-p relative threshold if set.
- **Penalties** (`penalties.py`) — repetition (decays by positional distance), frequency (linear in count), presence (flat, only requires ≥1 appearance). All return a **copy**, never mutating the input.

### 2.10 Running Tests

```bash
pytest modeling/testing/ -v
```

The suite is organized by subsystem across the **5 test files** listed in §2.1 (`test_attention.py`, `test_cache.py`, `test_quant.py`, `test_sampling.py`, `test_inference.py`). `test_inference.py` uses a tiny config (`d_model=64, n_layers=2, n_experts=4, vocab_size=256`) to keep the suite running quickly on CPU, while still covering every major code path: ternary, MLA, hybrid attention, Medusa, self-correction, KV-Q.

A clean baseline (all passing) is a mandatory prerequisite before any bug-fix pass — don't modify code while the baseline has unrelated failing tests.

### 2.11 Architectural Constraints to Respect When Extending This Package

1. **No upward imports** — `runtime/` must not be imported from `transformer.py`; `layers/` must not import from `runtime/` at module level (only via `TYPE_CHECKING` or lazy imports inside functions).
2. **`__init__.py` must match the actual package structure** — if a new file is added to a subpackage, its corresponding `__init__.py` must be updated, and so must the root `modeling/__init__.py` if that API needs to be public.
3. **Never create a new frozen `COMPUTE_DTYPE` alias** in any module — always call `get_compute_dtype()`.
4. **Every forward pass that writes KV must advance `cache_pos` exactly once** — this is the most fragile invariant across the entire speculative decoding subsystem.
5. **Changes to `kv_cache/`, `speculative.py`, `medusa.py`, `transformer.py` (the MTP path)** require empirical verification before modification if that subsystem has a history of prior fixes — high regression risk, especially on paths related to `cache_pos`/KV desync.

---

## Part 3 — Bridge Between the Two Packages

`thinking_engine/` does not depend on `numpy` or `transformer.py` directly anywhere except at a single point: `interfaces/llm.py`. The entire bridge:

```python
# engine.py, inside ThinkingEngine.__init__:
raw_bridge   = load_default_bridge(numpy_model=numpy_model)   # tries to import modeling.transformer.TransformerBridge
adapter      = TransformerBridgeAdapter(raw_bridge)            # matches the real API <-> the API the engine needs
locked_bridge = LockedBridge(adapter, bridge_lock)              # thread-locks it for ThreadPoolExecutor
```

### 3.1 Real `TransformerBridge` API (directly cross-checked against `transformer.py`)

| Real Method/Property | Signature |
|---|---|
| `bridge.backend` | `@property -> "numpy"` or `"llama.cpp"` |
| `bridge.set_intent_boost(arr)` | takes `np.ndarray[n_experts]`, stores into `self._intent_boost` |
| `bridge.set_intent_bias(arr)` | takes `np.ndarray[vocab_size]`, stores into `self._intent_bias` |
| `bridge.generate(prompt_ids, max_new_tokens, temp, top_p, min_p, eos_id, eos_ids, use_speculative_tree, spec_tree_width, spec_tree_depth, deterministic, rep_alpha, temp_inertia, snap_delta_threshold, debug, stream_cb, profiler, stop_event, checkpoint_every, checkpoint_path, wal, ...)` | Reads `self._intent_boost`/`self._intent_bias` internally — does NOT accept them as kwargs |

`TransformerBridge` **does not have** `is_connected()`, `expert_boost_to_array()`, `build_intent_bias()`, `to_generate_kwargs()` — these are methods `thinking_engine` needs but the real bridge doesn't provide, so `TransformerBridgeAdapter` (`interfaces/llm.py`) acts as the conversion layer:

```python
class TransformerBridgeAdapter:
    def is_connected(self) -> bool: ...              # True when the backend is a real "numpy"/"llama.cpp"
    def expert_boost_to_array(self, {0: 0.5, ...}) -> np.ndarray: ...
    def build_intent_bias(self, [151643, ...]) -> np.ndarray: ...
    def generate(self, prompt_ids, **kwargs) -> List[int]: ...              # direct proxy
    def generate_from_engine_output(self, prompt_ids, engine_out, ...):     # calls set_intent_boost/bias then generate()
```

9 call sites in `reasoning/debate/council.py`, `reasoning/cognitive/{abduction,counterfactual}.py`, `reasoning/thinking/tree_of_thought.py`, `planning/{goal_decomposer,plan_decomposer}.py`, `memory/summarization.py` call `bridge.is_connected()` before using `bridge.generate()` — all of them receive the adapter (not the raw TransformerBridge) via `EngineComponents.bridge`, so these calls work correctly with both the real bridge and `StubLLMBridge`.

### 3.2 Plugging a Real Model into `thinking_engine`

```python
from thinking_engine import ThinkingEngine
from modeling.transformer import DracoTransformerV1 

model = DracoTransformerV1(config=my_model_config)
engine = ThinkingEngine(numpy_model=model) 
```

If `modeling.transformer` fails to import (not installed, missing dependency), `load_default_bridge()` automatically falls back to `StubLLMBridge` — the entire pipeline still runs (no real tokens generated, but all routing/reasoning/hallucination-assessment logic still operates for testing purposes).

### 3.3 Dependencies

- **`thinking_engine/`**: no dependency beyond stdlib — all math/graph/probability helpers are written from scratch in `utils/`. `numpy` is only lazily imported (inside a function, not at the top level) in `interfaces/llm.py` when `TransformerBridgeAdapter` needs to build an array — so `thinking_engine/` can be installed and tested even without `numpy`/`modeling.transformer` present in the environment.
- **`modeling/`**: no required dependency beyond NumPy. Triton and Numba are optional accelerators that fall back silently when unavailable.

---

## Roadmap

This is a direction, not a committed timeline — it's maintained by one person alongside everything else, so dates are intentionally not given.

**Phase 1 — Optimization & Stabilization (current focus)**
- Reduce latency and improve verifier accuracy in the `thinking_engine/` pipeline; refine the 12-stage orchestration.
- Fix the known issues currently tracked from internal audits (missing classes, unwired components, thread-safety gaps in `runtime/`).
- Improve KV-cache efficiency, quantization stability, and speculative-decoding throughput in `modeling/`.
- Expand automated test coverage beyond the hallucination subsystem (see §1.9).

**Phase 2 — Core Expansion (not started)**
- A dedicated long-term Memory Engine that persists context across sessions.
- A graph-based Mapping Tool for relating internal representations across modalities/reasoning paths.
- A lightweight, pure-Python/NumPy Trainer Engine for fine-tuning Draco models without external ML-framework dependencies.

**Phase 3 — Full-Scale Training (not started)**
- Use the Trainer Engine above to train or continue pre-training Draco models on curated data, with reproducible logging.

**Phase 4 — Draco V2 ("Omni") (long-term research direction, not started)**
- Exploratory direction toward multi-modal support (text, vision, audio). This is speculative and far out; it is **not** a near-term commitment, and no V2 work will begin before V1's current feature set is stable and well-tested.

## Contributing

This project genuinely needs outside eyes — it's too large for one person to finish or fully audit alone. Contributions, bug reports, and skeptical code review are all welcome, including (especially) reports that point out something the README overstates or gets wrong.

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Make your changes, with tests where practical.
4. Run the existing suites (`pytest modeling/testing/ -v` and `pytest thinking_engine/reflection/hallucination/tests/ -v`) and confirm they still pass before submitting.
5. Open a pull request describing what changed and why.

If you're not sure where to start, opening an issue with questions or a review of an existing module is just as useful as a code change.

---

## License

GPL v3 © 2026 Draco Studio and DUCNGUYEN-creator

---
Built with ❤️ in Vietnam — Draco Studio