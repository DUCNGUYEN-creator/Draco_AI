# Hallucination Subsystem Architecture
## thinking_engine.reflection.hallucination v1.0.0

### Vị trí trong kiến trúc tổng thể

```
Infrastructure (Memory, RAG, Planner, Tools)
        ↓
Cognition (Reasoning, Planning)
        ↓
Verification ← bạn đang ở đây
  └── reflection/
        ├── self_reflection.py    # Style/coherence critique
        ├── consistency.py        # Answer vs reasoning trace
        ├── confidence.py         # Signal combination
        ├── confidence_calibrator.py  # Always-on lightweight calibrator
        ├── answer_rewriter.py    # Rewrite trigger + instruction
        ├── critic.py             # Post-generation orchestrator
        └── hallucination/        # THIS PACKAGE ← Deep risk assessment
```

### Pipeline 6 giai đoạn

```
Evidence → Verification → Calibration → Correlation → Fusion → Risk → Report
```

| Stage | Module | Nhiệm vụ |
|-------|--------|----------|
| Evidence | pipeline/evidence_pipeline.py | Thu thập EvidenceBundle từ KG/RAG/memory/tools |
| Verification | pipeline/verification_pipeline.py | Chạy verifier ensemble |
| Calibration | pipeline/calibration_pipeline.py | Calibrate per-verifier scores |
| Correlation | pipeline/correlation_pipeline.py | Khử trùng lặp, tính verifier weights |
| Fusion | pipeline/fusion_pipeline.py | Kết hợp signals thành 1 risk score |
| Risk → Report | pipeline/report_pipeline.py | Lắp ráp HallucinationReport |

### Quy tắc dependency chính

- `reflection/` phụ thuộc `hallucination/` (đọc Report) — **ĐÚNG**
- `hallucination/` **KHÔNG** phụ thuộc `reflection/` — **ĐÚNG**
- `hallucination/` **KHÔNG** sinh reasoning — **ĐÚNG**
- `assessor.py` là entry point DUY NHẤT từ bên ngoài — **ĐÚNG**

### Strategy tiers

| Strategy | Verifiers | Fusion | Calibration | Use case |
|----------|-----------|--------|-------------|----------|
| fast | 2 | noisy_or | temperature | INTENT_CHAT, low-latency |
| balanced | 6 | noisy_or | platt | DEFAULT |
| paranoid | 9 (tất cả) | ensemble | ensemble | High-stakes |
| custom | configurable | configurable | configurable | Expert use |
