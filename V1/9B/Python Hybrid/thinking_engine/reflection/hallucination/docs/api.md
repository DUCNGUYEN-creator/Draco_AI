# Hallucination API Reference

## Entry point

```python
from thinking_engine.reflection.hallucination import Assessor
from thinking_engine.config import HallucinationConfig

cfg = HallucinationConfig(
    strategy="balanced",        # fast | balanced | paranoid | custom
    fusion_method="noisy_or",   # weighted | noisy_or | bayesian | dempster_shafer | logistic
    calibration_method="platt", # platt | isotonic | beta | temperature | histogram | ensemble
    risk_thresholds={"low":0.15, "medium":0.35, "high":0.60, "critical":0.85},
)
assessor = Assessor(config=cfg)

report = assessor.assess(
    answer="The model's generated text...",
    context={
        "reasoning_path": [...],      # List[str] from KnowledgeGraph
        "rag_docs": [...],            # List[dict] from RAGPipeline
        "memory_summary": "...",      # str from MemoryRetrieval
        "tool_results": [...],        # List[dict] from ToolExecutor
        "subgoals": [...],            # List[str] from PlanDecomposer
        "debate_opinions": {...},     # Dict[int,str] from MultiAgentDebate
        "known_citation_ids": [...],  # List[str] from CitationTracker
    },
    strategy_name="balanced",   # optional per-call override
)

print(report.risk_level)    # RiskLevel enum
print(report.risk_score)    # float 0.0 (safe) .. 1.0 (critical)
print(report.top_issues)    # List[str] — most important issues
print(report.as_dict())     # JSON-safe dict for logging/PromptCompiler
```

## Registering a custom verifier

```python
from thinking_engine.reflection.hallucination.registry import VerifierRegistry

class MyVerifier:
    name = "my_verifier"
    def verify(self, claim, evidence, context):
        return {"verifier": self.name, "score": 0.8, "confidence": 0.7, "issues": []}

registry = VerifierRegistry()
registry.register("my_verifier", MyVerifier)
```
