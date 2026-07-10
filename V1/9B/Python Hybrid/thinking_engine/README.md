# DracoAI Thinking Engine v1.0.0

**Draco Studio — DUCNGUYEN-creator — GPL v3**

A package split off from `engine_v1.py` (a ~3,775-line monolith) into a modular architecture of **277 modules / 13,212 lines / 47 folders**, following the Hybrid Architecture design exactly:

```text
Engine = Infrastructure + Cognition + Verification
```

## 3-Layer Architecture

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
                                        └─ hallucination/    ← In-depth
```

## 12-Stage Pipeline (Perception → Output)

| # | Stage | Module | Task |
|---|-------|--------|----------|
| 1 | Perception | `perception/` | Sanitize, rewrite, intent, entity, metaphor, difficulty |
| 2 | Memory Retrieval | `memory/` | Rerank, compress, episodic recall |
| 3 | Knowledge Retrieval | `knowledge/` | KG path, RAG, analogy, counterfactual, abduction |
| 4 | Planning | `planning/` | Tool injection, goal/plan decomposition |
| 5 | Reasoning | `reasoning/` | ToT, Council Debate, Self-Consistency, MCTS |
| 6 | Tool Execution | `tools/` | Safe AST eval, sandbox, tool registry |
| 7 | Reasoning Loop | `reasoning/thinking/` | Chain-of-thought verification, recursive reflection |
| 8 | Verifier Layer | `reasoning/thinking/chain_verifier.py` | CoT soundness check |
| 9 | Hallucination Assessment | `reflection/hallucination/` | In-depth 6-stage pipeline |
| 10 | Reflection | `reflection/critic.py` | Recursive critique, post-generation check |
| 11 | Answer Rewriter | `reflection/answer_rewriter.py` | Rewrite trigger if risk is high |
| 12 | Output | `perception/prompt/compiler.py` | Compile [PLAN][THOUGHT][FINAL ANSWER] |

## Hallucination Subsystem (In-depth)

```text
Evidence → Verification → Calibration → Correlation → Fusion → Risk → Report
```

### 9 Verifiers (the heart of the system)

| Verifier | Checks | Confidence |
|----------|----------|------------|
| retrieval | Claim matches retrieved evidence | High when trust_score is high |
| contradiction | Claim contradicts evidence (negation-flip, antonym) | Medium |
| consistency | Claim agrees with multiple independent reasoning paths | Increases with n_paths |
| numerical | Arithmetic expression is correct (deterministic, via SafeASTEvaluator) | Extremely high at 0.9 |
| symbolic | Logical proposition (tautology/contradiction) is valid | 0.85 when applicable |
| citation | `[hexid]` citation exists in CitationTracker | 0.9 vs registry |
| planner | Claim matches committed subgoals/plan | 0.4 soft signal |
| tool | Claim matches actual tool output (most trusted source) | 0.85 |
| reasoning | Claim is traceable from the selected reasoning trace | 0.55 |

### Calibration Methods

`platt` · `isotonic` · `beta` · `temperature` · `histogram` + `ensemble`

### 5 Fusion Methods

| Method | Characteristics | Default |
|---|---|---|
| noisy_or | One strong signal dominates all others | ✓ |
| weighted | Weighted average, easy to interpret | N/A |
| bayesian | Sequential update, distinguishes positive/negative signals | N/A |
| dempster_shafer | Handles abstain (mass-on-uncertainty) precisely | N/A |
| logistic | Weighted sum of logits, does not compress extreme probabilities | N/A |

### 3 Strategy Tiers (per architecture documentation)

| Strategy | Verifiers | Use case |
|---|---|---|
| fast | 2 (retrieval + contradiction) | INTENT_CHAT, low-latency |
| balanced | 6 | DEFAULT for every request |
| paranoid | 9 (all, automatically expands as more are registered) | High-stakes |

## Folder Structure

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
│       ├── assessor.py           ← PUBLIC ENTRY POINT (the only one)
│       ├── models/               ← Evidence, VerificationResult, FusionResult, Report, ...
│       ├── verifiers/            ← 9 specialized verifiers
│       ├── analyzers/            ← Taxonomy, Severity, Agreement, Outlier, Coverage, ...
│       ├── calibration/          ← Platt/Isotonic/Beta/Temperature/Histogram/Ensemble
│       ├── correlation/          ← Similarity, Dedup, ConnectedComponents, Reducer
│       ├── fusion/               ← NoisyOr/Weighted/Bayesian/DempsterShafer/Logistic
│       ├── metrics/              ← AUC, Brier, ECE, Drift, Reliability, Entropy, ...
│       ├── pipeline/             ← 6-stage Evidence→Verification→...→Report
│       ├── registry/             ← Plugin registry (no need to modify old code when adding new ones)
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

## Quick Start

```python
from thinking_engine import ThinkingEngine

# Initialize (stub bridge — no real model needed to run)
engine = ThinkingEngine()

# Phase 1: compile prompt
out = engine.process(
    "Write a Python function to compute a factorial",
    history=[],
    think_mode=False,           # True = activate Council Debate (8 experts)
)
print(out.intent)               # {'intent': 'code', 'lang': 'vi', ...}
print(out.expert_boost)         # {0: 0.04, 1: 0.48, 2: 0.27, ...}
print(len(out.messages))        # 3 (system + plan + user)

# Phase 2: after the LLM generates a response, evaluate Hallucination + Reflection
final = engine.finish(
    "Write a Python function to compute a factorial",
    generated_text="def factorial(n): return 1 if n<=1 else n*factorial(n-1)",
)
print(final.hallucination_report['risk_level'])   # 'none' / 'low' / 'medium' / 'high' / 'critical'
print(final.hallucination_report['risk_score'])   # float [0.0, 1.0]
print(final.hallucination_report['top_issues'])   # List[str]

# Feedback loop (improves router + calibration over time)
engine.submit_feedback("Write a Python function...", "def factorial...", rating=0.95)
```

## Using the Hallucination Assessor Standalone

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

## Registering a New Verifier (Plugin Architecture)

```python
from thinking_engine.reflection.hallucination.registry import VerifierRegistry

class MyEmbeddingVerifier:
    name = "embedding"
    def verify(self, claim, evidence, context):
        # ... embedding similarity check ...
        return {"verifier": self.name, "score": 0.85, "confidence": 0.9, "issues": []}

registry = VerifierRegistry()
registry.register("embedding", MyEmbeddingVerifier)
# No need to modify assessor.py or strategy/*.py
```

## Origin & Comparison with engine_v1.py

| Aspect | engine_v1.py | thinking_engine/ |
|---|---|---|
| Size | ~3,775 lines / 1 file | 13,212 lines / 278 files |
| Architecture | Monolith (ThinkingEngineV1) | 3-layer: Infrastructure/Cognition/Verification |
| Hallucination Layer | SelfReflection.critique() inline | In-depth 6-stage pipeline, 9 verifiers |
| Fusion | None | 5 methods (noisy_or, bayesian, ...) |
| Calibration | 1 online Platt | 5 methods + ensemble |
| Search algorithms | BFS, DFS, A*, MCTS | + Beam, IDA*, Bidirectional |
| Tests | 1 inline self-test block | 13 structured tests, 100% pass |
| Plugin | None | Registry + Factory pattern |
| Dependency separation | Everything in 1 class | Verification layer does not depend on Cognition |
| Import sweep | N/A | 277/277 modules, 0 errors |

## Connection with `modeling/transformer.py` (the real TransformerBridge)

`thinking_engine/` does not depend directly on `numpy` or `transformer.py` anywhere except at a single point: `interfaces/llm.py`. This is the entire bridge:

```python
# engine.py, inside ThinkingEngine.__init__:
raw_bridge   = load_default_bridge(numpy_model=numpy_model)   # attempts to import modeling.transformer.TransformerBridge
adapter      = TransformerBridgeAdapter(raw_bridge)            # matches the real API <-> the API the engine needs
locked_bridge = LockedBridge(adapter, bridge_lock)              # locks the stream for ThreadPoolExecutor
```

### The Real TransformerBridge API (cross-checked directly against transformer.py)

| Real Method/Property | Signature |
|---|---|
| `bridge.backend` | `@property -> "numpy"` or `"llama.cpp"` |
| `bridge.set_intent_boost(arr)` | receives `np.ndarray[n_experts]`, stores into `self._intent_boost` |
| `bridge.set_intent_bias(arr)` | receives `np.ndarray[vocab_size]`, stores into `self._intent_bias` |
| `bridge.generate(prompt_ids, max_new_tokens, temp, top_p, min_p, eos_id, eos_ids, use_speculative_tree, spec_tree_width, spec_tree_depth, deterministic, rep_alpha, temp_inertia, snap_delta_threshold, debug, stream_cb, profiler, stop_event, checkpoint_every, checkpoint_path, wal, ...)` | Reads `self._intent_boost`/`self._intent_bias` internally — does NOT receive them as kwargs |

`TransformerBridge` has no `is_connected()`, `expert_boost_to_array()`, `build_intent_bias()`, or `to_generate_kwargs()` — these are methods that `thinking_engine` needs but the real bridge does not provide, so `TransformerBridgeAdapter` (`interfaces/llm.py`) steps in as the conversion layer:

```python
class TransformerBridgeAdapter:
    def is_connected(self) -> bool: ...              # True when the backend is a real "numpy"/"llama.cpp"
    def expert_boost_to_array(self, {0: 0.5, ...}) -> np.ndarray: ...
    def build_intent_bias(self, [151643, ...]) -> np.ndarray: ...
    def generate(self, prompt_ids, **kwargs) -> List[int]: ...              # direct proxy
    def generate_from_engine_output(self, prompt_ids, engine_out, ...):     # calls set_intent_boost/bias, then generate()
```

9 locations in `reasoning/debate/council.py`, `reasoning/cognitive/{abduction,counterfactual}.py`, `reasoning/thinking/tree_of_thought.py`, `planning/{goal_decomposer,plan_decomposer}.py`, and `memory/summarization.py` call `bridge.is_connected()` before using `bridge.generate()` — all of them receive the adapter (not the raw TransformerBridge) via `EngineComponents.bridge`, so these calls work correctly with both the real bridge and `StubLLMBridge`.

### How to Plug in a Real Model

```python
from thinking_engine import ThinkingEngine
from modeling.transformer import DracoTransformerV1 

model = DracoTransformerV1(config=my_model_config)
engine = ThinkingEngine(numpy_model=model) 
```

If `modeling.transformer` cannot be imported (not installed, missing dependency), `load_default_bridge()` automatically falls back to `StubLLMBridge` — the entire pipeline still runs (no real tokens are generated, but all routing/reasoning/hallucination-assessment logic still works for testing).

## Main Dependencies (Besides stdlib)

None — all math/graph/probability helpers are written from scratch in `utils/`. `numpy` is only lazily imported (inside a function, not at the top level) in `interfaces/llm.py` when `TransformerBridgeAdapter` needs to construct an array — so `thinking_engine/` can be installed and tested even without `numpy`/`modeling.transformer` present in the environment.

## License

GPL v3 © 2026 Draco Studio and DUCNGUYEN-creator