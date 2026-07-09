# Verifier Reference

## Built-in verifiers (9)

| Name | Kind | What it checks | Confidence when strong |
|------|------|---------------|----------------------|
| retrieval | RETRIEVAL | Claim lexically supported by retrieved evidence | High when trust_score high |
| contradiction | CONTRADICTION | Claim actively negates evidence (negation-flip, antonyms) | Medium |
| consistency | CONSISTENCY | Claim agrees with majority of independent reasoning paths | Grows with n_paths |
| numerical | NUMERICAL | Arithmetic expressions in claim are mathematically correct | 0.9 (very high — deterministic) |
| symbolic | SYMBOLIC | Formal logic claims (tautology/contradiction) are valid | 0.85 when applicable |
| citation | CITATION | [hexid] markers reference actually-retrieved documents | 0.9 vs registry, 0.2 without |
| planner | PLANNER | Claim overlaps with the committed plan/subgoals | 0.4 (soft signal) |
| tool | TOOL | Claim reflects actual tool execution output | 0.85 (high-trust ground truth) |
| reasoning | REASONING | Claim traceable to selected reasoning trace | 0.5 (moderate) |

## Verifier score semantics

`score = 1.0` → fully supported  
`score = 0.0` → not supported / contradicted  
`confidence = 0.0` → verifier abstains (not enough signal for this claim type)

**High confidence, low score** = strong hallucination signal  
**Low confidence, any score** = verifier doesn't have enough information, contribute little to fusion
