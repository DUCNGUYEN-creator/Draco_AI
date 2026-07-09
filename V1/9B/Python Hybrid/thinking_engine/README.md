# DracoAI Thinking Engine v1.0.0

**Draco Studio — DUCNGUYEN-creator — GPL v3**

Package tách từ `engine_v1.py` (~3,775 dòng monolith) thành kiến trúc modular **277 module / 13,212 dòng / 47 thư mục** theo đúng thiết kế Hybrid Architecture:

```
Engine = Infrastructure + Cognition + Verification
```

---

## Kiến trúc 3 tầng

```
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
                                        └─ hallucination/    ← Chuyên sâu
```

---

## Pipeline 12 tầng (Perception → Output)

| # | Stage | Module | Nhiệm vụ |
|---|-------|--------|----------|
| 1 | Perception | `perception/` | Sanitize, rewrite, intent, entity, metaphor, difficulty |
| 2 | Memory Retrieval | `memory/` | Rerank, compress, episodic recall |
| 3 | Knowledge Retrieval | `knowledge/` | KG path, RAG, analogy, counterfactual, abduction |
| 4 | Planning | `planning/` | Tool injection, goal/plan decomposition |
| 5 | Reasoning | `reasoning/` | ToT, Council Debate, Self-Consistency, MCTS |
| 6 | Tool Execution | `tools/` | Safe AST eval, sandbox, tool registry |
| 7 | Reasoning Loop | `reasoning/thinking/` | Chain-of-thought verification, recursive reflection |
| 8 | Verifier Layer | `reasoning/thinking/chain_verifier.py` | CoT soundness check |
| 9 | Hallucination Assessment | `reflection/hallucination/` | 6-stage pipeline chuyên sâu |
| 10 | Reflection | `reflection/critic.py` | Recursive critique, post-generation check |
| 11 | Answer Rewriter | `reflection/answer_rewriter.py` | Rewrite trigger nếu risk cao |
| 12 | Output | `perception/prompt/compiler.py` | Compile [PLAN][THOUGHT][FINAL ANSWER] |

---

## Hallucination Subsystem (Chuyên sâu)

```
Evidence → Verification → Calibration → Correlation → Fusion → Risk → Report
```

### 9 Verifiers (trái tim hệ thống)

| Verifier | Kiểm tra | Confidence |
|----------|----------|-----------|
| `retrieval` | Claim khớp evidence retrieved | Cao khi trust_score cao |
| `contradiction` | Claim mâu thuẫn evidence (negation-flip, antonym) | Trung bình |
| `consistency` | Claim đồng thuận với nhiều đường reasoning độc lập | Tăng theo n_paths |
| `numerical` | Biểu thức số học đúng (deterministic, qua SafeASTEvaluator) | 0.9 cực cao |
| `symbolic` | Mệnh đề logic (tautology/contradiction) hợp lệ | 0.85 khi applicable |
| `citation` | `[hexid]` citation tồn tại trong CitationTracker | 0.9 vs registry |
| `planner` | Claim khớp subgoals/kế hoạch đã cam kết | 0.4 soft signal |
| `tool` | Claim khớp output tool thực tế (source tin cậy nhất) | 0.85 |
| `reasoning` | Claim traceable từ reasoning trace đã chọn | 0.5 |

### 5 Phương pháp Calibration

`platt` · `isotonic` · `beta` · `temperature` · `histogram` + `ensemble`

### 5 Phương pháp Fusion

| Phương pháp | Đặc tính | Default |
|-------------|----------|---------|
| `noisy_or` | Một tín hiệu mạnh áp đảo toàn bộ | ✓ |
| `weighted` | Trung bình có trọng số, dễ giải thích | |
| `bayesian` | Update tuần tự, phân biệt tín hiệu âm/dương | |
| `dempster_shafer` | Xử lý abstain (mass-on-uncertainty) chính xác | |
| `logistic` | Tổng logit có trọng số, không nén xác suất cực | |

### 3 Strategy tiers (theo tài liệu kiến trúc)

| Strategy | Verifiers | Use case |
|----------|-----------|----------|
| `fast` | 2 (retrieval + contradiction) | INTENT_CHAT, low-latency |
| `balanced` | 6 | DEFAULT cho mọi request |
| `paranoid` | 9 (tất cả, tự động mở rộng khi đăng ký thêm) | High-stakes |

---

## Cấu trúc thư mục

```
thinking_engine/
├── __init__.py           ← ThinkingEngine, EngineConfig
├── engine.py             ← ThinkingEngine (top-level facade)
├── pipeline.py           ← Global Pipeline 12-stage orchestrator
├── state.py              ← ThinkingState (carrier xuyên suốt pipeline)
├── config.py             ← EngineConfig (toàn bộ tuning knobs)
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
│       ├── assessor.py           ← PUBLIC ENTRY POINT (duy nhất)
│       ├── models/               ← Evidence, VerificationResult, FusionResult, Report, ...
│       ├── verifiers/            ← 9 verifier chuyên biệt
│       ├── analyzers/            ← Taxonomy, Severity, Agreement, Outlier, Coverage, ...
│       ├── calibration/          ← Platt/Isotonic/Beta/Temperature/Histogram/Ensemble
│       ├── correlation/          ← Similarity, Dedup, ConnectedComponents, Reducer
│       ├── fusion/               ← NoisyOr/Weighted/Bayesian/DempsterShafer/Logistic
│       ├── metrics/              ← AUC, Brier, ECE, Drift, Reliability, Entropy, ...
│       ├── pipeline/             ← 6-stage Evidence→Verification→...→Report
│       ├── registry/             ← Plugin registry (không sửa code cũ khi thêm mới)
│       ├── factory/              ← Factory với instance caching
│       ├── strategy/             ← fast/balanced/paranoid/custom
│       ├── cache/                ← LRU+TTL: evidence, verifier, calibration, stats
│       ├── benchmarks/           ← AUC/ECE/monotonicity benchmarks nội bộ
│       ├── docs/                 ← architecture.md, api.md, verifier.md, ...
│       └── tests/                ← 13 test cases, 100% pass
│
├── safety/               ← EthicalFilter, PromptGuard, InjectionDetector, ActiveLearning
├── learning/             ← FeedbackCollector, OnlineLearner, RouterUpdater, ExperienceBuffer
├── execution/            ← InferenceRequest, ResponseBuilder, ResponseFormatter, EngineOutput
└── utils/                ← Hash, Graph, Math, Probability, Timer, Thread, Serialization
```

---

## Quick Start

```python
from thinking_engine import ThinkingEngine

# Khởi tạo (stub bridge — không cần model thật để chạy)
engine = ThinkingEngine()

# Phase 1: compile prompt
out = engine.process(
    "Viết hàm Python tính giai thừa",
    history=[],
    think_mode=False,           # True = kích hoạt Council Debate (8 expert)
)
print(out.intent)               # {'intent': 'code', 'lang': 'vi', ...}
print(out.expert_boost)         # {0: 0.04, 1: 0.48, 2: 0.27, ...}
print(len(out.messages))        # 3 (system + plan + user)

# Phase 2: sau khi LLM tạo ra response, đánh giá Hallucination + Reflection
final = engine.finish(
    "Viết hàm Python tính giai thừa",
    generated_text="def factorial(n): return 1 if n<=1 else n*factorial(n-1)",
)
print(final.hallucination_report['risk_level'])   # 'none' / 'low' / 'medium' / 'high' / 'critical'
print(final.hallucination_report['risk_score'])   # float [0.0, 1.0]
print(final.hallucination_report['top_issues'])   # List[str]

# Feedback loop (cải thiện router + calibration theo thời gian)
engine.submit_feedback("Viết hàm Python...", "def factorial...", rating=0.95)
```

### Dùng Hallucination Assessor độc lập

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

### Đăng ký verifier mới (plugin architecture)

```python
from thinking_engine.reflection.hallucination.registry import VerifierRegistry

class MyEmbeddingVerifier:
    name = "embedding"
    def verify(self, claim, evidence, context):
        # ... embedding similarity check ...
        return {"verifier": self.name, "score": 0.85, "confidence": 0.9, "issues": []}

registry = VerifierRegistry()
registry.register("embedding", MyEmbeddingVerifier)
# Không cần sửa assessor.py hay strategy/*.py
```

---

## Nguồn gốc & So sánh với engine_v1.py

| Aspect | engine_v1.py | thinking_engine/ |
|--------|-------------|-----------------|
| Kích thước | ~3,775 dòng / 1 file | 13,212 dòng / 278 file |
| Kiến trúc | Monolith (ThinkingEngineV1) | 3-layer: Infrastructure/Cognition/Verification |
| Tầng Hallucination | `SelfReflection.critique()` inline | 6-stage deep pipeline, 9 verifiers |
| Fusion | Không có | 5 phương pháp (noisy_or, bayesian, ...) |
| Calibration | 1 Platt online | 5 phương pháp + ensemble |
| Search algorithms | BFS, DFS, A*, MCTS | + Beam, IDA*, Bidirectional |
| Tests | 1 self-test block inline | 13 structured tests, 100% pass |
| Plugin | Không có | Registry + Factory pattern |
| Dependency separation | Tất cả trong 1 class | Tầng Verification không phụ thuộc Cognition |
| Import sweep | N/A | 277/277 module, 0 lỗi |

---

## Liên kết với `modeling/transformer.py` (TransformerBridge thật)

`thinking_engine/` không phụ thuộc `numpy` hay `transformer.py` trực tiếp ở bất kỳ đâu ngoài **một điểm duy nhất**: `interfaces/llm.py`. Đây là toàn bộ cầu nối:

```python
# engine.py, trong ThinkingEngine.__init__:
raw_bridge   = load_default_bridge(numpy_model=numpy_model)   # thử import modeling.transformer.TransformerBridge
adapter      = TransformerBridgeAdapter(raw_bridge)            # khớp API thật <-> API engine cần
locked_bridge = LockedBridge(adapter, bridge_lock)              # khóa luồng cho ThreadPoolExecutor
```

### API thật của `TransformerBridge` (đã đối chiếu trực tiếp với `transformer.py`)

| Method/Property thật | Chữ ký |
|---|---|
| `bridge.backend` | `@property -> "numpy" \| "llama.cpp"` |
| `bridge.set_intent_boost(arr)` | nhận `np.ndarray[n_experts]`, lưu vào `self._intent_boost` |
| `bridge.set_intent_bias(arr)` | nhận `np.ndarray[vocab_size]`, lưu vào `self._intent_bias` |
| `bridge.generate(prompt_ids, max_new_tokens, temp, top_p, min_p, eos_id, eos_ids, use_speculative_tree, spec_tree_width, spec_tree_depth, deterministic, rep_alpha, temp_inertia, snap_delta_threshold, debug, stream_cb, profiler, stop_event, checkpoint_every, checkpoint_path, wal, ...)` | Đọc `self._intent_boost`/`self._intent_bias` nội bộ — **KHÔNG** nhận chúng làm kwargs |

`TransformerBridge` **không có** `is_connected()`, `expert_boost_to_array()`, `build_intent_bias()`, `to_generate_kwargs()` — đây là những method `thinking_engine` cần nhưng bridge thật không cung cấp, nên `TransformerBridgeAdapter` (`interfaces/llm.py`) đứng ra làm lớp chuyển đổi:

```python
class TransformerBridgeAdapter:
    def is_connected(self) -> bool: ...              # True khi backend là "numpy"/"llama.cpp" thật
    def expert_boost_to_array(self, {0: 0.5, ...}) -> np.ndarray: ...
    def build_intent_bias(self, [151643, ...]) -> np.ndarray: ...
    def generate(self, prompt_ids, **kwargs) -> List[int]: ...              # proxy trực tiếp
    def generate_from_engine_output(self, prompt_ids, engine_out, ...):     # gọi set_intent_boost/bias rồi generate()
```

**9 vị trí** trong `reasoning/debate/council.py`, `reasoning/cognitive/{abduction,counterfactual}.py`, `reasoning/thinking/tree_of_thought.py`, `planning/{goal_decomposer,plan_decomposer}.py`, `memory/summarization.py` gọi `bridge.is_connected()` trước khi dùng `bridge.generate()` — toàn bộ đều nhận `adapter` (không phải `TransformerBridge` thô) qua `EngineComponents.bridge`, nên các lời gọi này hoạt động đúng với cả bridge thật lẫn `StubLLMBridge`.

### Cách cắm model thật vào

```python
from thinking_engine import ThinkingEngine
from modeling.transformer import DracoTransformerV1  # file transformer.py của bạn

model = DracoTransformerV1(config=my_model_config)
engine = ThinkingEngine(numpy_model=model)   # tự động dùng backend="numpy" thật
```

Nếu `modeling.transformer` không import được (chưa cài, thiếu dependency), `load_default_bridge()` tự rơi về `StubLLMBridge` — toàn bộ pipeline vẫn chạy được (không sinh token thật, nhưng mọi logic routing/reasoning/hallucination-assessment vẫn hoạt động để test).

---

## Dependency chính (ngoài stdlib)

Không có — toàn bộ math/graph/probability helpers được viết từ đầu trong `utils/`. `numpy` chỉ được import lười (bên trong hàm, không phải top-level) tại `interfaces/llm.py` khi `TransformerBridgeAdapter` cần dựng mảng — nên `thinking_engine/` cài đặt và chạy test được ngay cả khi không có `numpy`/`modeling.transformer` trong môi trường.

## License

GPL v3 © 2026 Draco Studio and DUCNGUYEN-creator
